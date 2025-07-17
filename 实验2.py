#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于MindX SDK的目标检测实验二
YOLOv3目标检测完整实现
"""

import os
import time
import argparse
import ast
import shutil
import numpy as np
import mindspore as ms
from mindspore import nn, Tensor, load_checkpoint, load_param_into_net
from mindspore import ops
from mindspore import dataset
from mindspore.mindrecord import FileWriter
from mindspore.dataset import vision
from mindspore.common.initializer import initializer
from mindspore.train import CheckpointConfig, ModelCheckpoint, LossMonitor, Model
from mindspore.communication import init
from mindspore import set_seed

# 设置随机种子
set_seed(1)

# 配置类
class ConfigYOLOV3ResNet18:
    # YOLOv3模型Config参数
    img_shape = [352, 640]
    feature_shape = [32, 3, 352, 640]
    num_classes = 3
    nms_max_num = 50
    _NUM_BOXES = 50
    
    backbone_input_shape = [64, 64, 128, 256]
    backbone_shape = [64, 128, 256, 512]
    backbone_layers = [2, 2, 2, 2]
    backbone_stride = [1, 2, 2, 2]
    
    ignore_threshold = 0.5
    obj_threshold = 0.3
    nms_threshold = 0.4
    
    anchor_scales = [(5, 3), (10, 13), (16, 30), (33, 23), (30, 61), (62, 45), (59, 119), (116, 90), (156, 198)]
    out_channel = int(len(anchor_scales) / 3 * (num_classes + 5))

# 权重初始化
def weight_variable():
    return ms.common.initializer.TruncatedNormal(0.02)

# 基础层定义
class _conv2d(nn.Cell):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(_conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 
                             kernel_size=kernel_size, stride=stride, 
                             padding=0, pad_mode='same',
                             weight_init=weight_variable())
    
    def construct(self, x):
        x = self.conv(x)
        return x

def _fused_bn(channels, momentum=0.99):
    return nn.BatchNorm2d(channels, momentum=momentum)

def _conv_bn_relu(in_channel, out_channel, ksize, stride=1, padding=0, 
                 dilation=1, alpha=0.1, momentum=0.99, pad_mode="same"):
    return nn.SequentialCell([
        nn.Conv2d(in_channel, out_channel, kernel_size=ksize, 
                 stride=stride, padding=padding, dilation=dilation, 
                 pad_mode=pad_mode),
        nn.BatchNorm2d(out_channel, momentum=momentum),
        nn.LeakyReLU(alpha)
    ])

# ResNet基础块
class BasicBlock(nn.Cell):
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1, momentum=0.99):
        super(BasicBlock, self).__init__()
        self.conv1 = _conv2d(in_channels, out_channels, 3, stride=stride)
        self.bn1 = _fused_bn(out_channels, momentum=momentum)
        self.conv2 = _conv2d(out_channels, out_channels, 3)
        self.bn2 = _fused_bn(out_channels, momentum=momentum)
        self.relu = nn.ReLU()
        self.down_sample_layer = None
        self.downsample = (in_channels != out_channels)
        
        if self.downsample:
            self.down_sample_layer = _conv2d(in_channels, out_channels, 1, stride=stride)
    
    def construct(self, x):
        identity = x
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        
        if self.downsample:
            identity = self.down_sample_layer(identity)
        
        out = ops.add(x, identity)
        out = self.relu(out)
        
        return out

# ResNet网络
class ResNet(nn.Cell):
    def __init__(self, block, layer_nums, in_channels, out_channels, 
                 strides=None, num_classes=80):
        super(ResNet, self).__init__()
        
        if not len(layer_nums) == len(in_channels) == len(out_channels) == 4:
            raise ValueError("the length of layer_num, inchannel, outchannel list must be 4!")
        
        self.conv1 = _conv2d(3, 64, 7, stride=2)
        self.bn1 = _fused_bn(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')
        
        self.layer1 = self._make_layer(block, layer_nums[0], 
                                      in_channel=in_channels[0], 
                                      out_channel=out_channels[0], 
                                      stride=strides[0])
        self.layer2 = self._make_layer(block, layer_nums[1], 
                                      in_channel=in_channels[1], 
                                      out_channel=out_channels[1], 
                                      stride=strides[1])
        self.layer3 = self._make_layer(block, layer_nums[2], 
                                      in_channel=in_channels[2], 
                                      out_channel=out_channels[2], 
                                      stride=strides[2])
        self.layer4 = self._make_layer(block, layer_nums[3], 
                                      in_channel=in_channels[3], 
                                      out_channel=out_channels[3], 
                                      stride=strides[3])
        
        self.num_classes = num_classes
        if num_classes:
            self.reduce_mean = ops.ReduceMean(keep_dims=True)
            self.end_point = nn.Dense(out_channels[3], num_classes, has_bias=True,
                                    weight_init=weight_variable(),
                                    bias_init=weight_variable())
            self.squeeze = lambda out: ops.squeeze(out, axis=(2, 3))
    
    def _make_layer(self, block, layer_num, in_channel, out_channel, stride):
        layers = []
        
        resblk = block(in_channel, out_channel, stride=stride)
        layers.append(resblk)
        
        for _ in range(1, layer_num):
            resblk = block(out_channel, out_channel, stride=1)
            layers.append(resblk)
        
        return nn.SequentialCell(layers)
    
    def construct(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        c1 = self.maxpool(x)
        
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        
        out = c5
        if self.num_classes:
            out = self.reduce_mean(c5, (2, 3))
            out = self.squeeze(out)
            out = self.end_point(out)
        
        return c3, c4, out

def resnet18(class_num=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], 
                 [64, 64, 128, 256], [64, 128, 256, 512], 
                 [1, 2, 2, 2], num_classes=class_num)

# YOLO块
class YoloBlock(nn.Cell):
    def __init__(self, in_channels, out_chls, out_channels):
        super(YoloBlock, self).__init__()
        out_chls_2 = out_chls * 2
        
        self.conv0 = _conv_bn_relu(in_channels, out_chls, ksize=1)
        self.conv1 = _conv_bn_relu(out_chls, out_chls_2, ksize=3)
        self.conv2 = _conv_bn_relu(out_chls_2, out_chls, ksize=1)
        self.conv3 = _conv_bn_relu(out_chls, out_chls_2, ksize=3)
        self.conv4 = _conv_bn_relu(out_chls_2, out_chls, ksize=1)
        self.conv5 = _conv_bn_relu(out_chls, out_chls_2, ksize=3)
        self.conv6 = nn.Conv2d(out_chls_2, out_channels, kernel_size=1, stride=1, has_bias=True)
    
    def construct(self, x):
        c1 = self.conv0(x)
        c2 = self.conv1(c1)
        c3 = self.conv2(c2)
        c4 = self.conv3(c3)
        c5 = self.conv4(c4)
        c6 = self.conv5(c5)
        out = self.conv6(c6)
        return c5, out

# YOLOv3网络
class YOLOv3(nn.Cell):
    def __init__(self, feature_shape, backbone_shape, backbone, out_channel):
        super(YOLOv3, self).__init__()
        self.out_channel = out_channel
        self.net = backbone
        self.backblock0 = YoloBlock(backbone_shape[-1], out_chls=backbone_shape[-2], 
                                   out_channels=out_channel)
        
        self.conv1 = _conv_bn_relu(in_channel=backbone_shape[-2], 
                                  out_channel=backbone_shape[-2]//2, ksize=1)
        self.upsample1 = ops.ResizeNearestNeighbor((feature_shape[2]//16, 
                                                   feature_shape[3]//16))
        self.backblock1 = YoloBlock(in_channels=backbone_shape[-2]+backbone_shape[-3],
                                   out_chls=backbone_shape[-3], 
                                   out_channels=out_channel)
        
        self.conv2 = _conv_bn_relu(in_channel=backbone_shape[-3], 
                                  out_channel=backbone_shape[-3]//2, ksize=1)
        self.upsample2 = ops.ResizeNearestNeighbor((feature_shape[2]//8, 
                                                   feature_shape[3]//8))
        self.backblock2 = YoloBlock(in_channels=backbone_shape[-3]+backbone_shape[-4],
                                   out_chls=backbone_shape[-4], 
                                   out_channels=out_channel)
        self.concat = lambda x: ops.concat(x, axis=1)
    
    def construct(self, x):
        feature_map1, feature_map2, feature_map3 = self.net(x)
        
        con1, big_object_output = self.backblock0(feature_map3)
        
        con1 = self.conv1(con1)
        ups1 = self.upsample1(con1)
        con1 = self.concat((ups1, feature_map2))
        
        con2, medium_object_output = self.backblock1(con1)
        con2 = self.conv2(con2)
        ups2 = self.upsample2(con2)
        con3 = self.concat((ups2, feature_map1))
        _, small_object_output = self.backblock2(con3)
        
        return big_object_output, medium_object_output, small_object_output

# 检测块
class DetectionBlock(nn.Cell):
    def __init__(self, scale, config):
        super(DetectionBlock, self).__init__()
        self.config = config
        
        if scale == 's':
            idx = (0, 1, 2)
        elif scale == 'm':
            idx = (3, 4, 5)
        elif scale == 'l':
            idx = (6, 7, 8)
        else:
            raise KeyError("Invalid scale value for DetectionBlock")
        
        self.anchors = Tensor([self.config.anchor_scales[i] for i in idx], ms.float32)
        self.num_anchors_per_scale = 3
        self.num_attrib = 4 + 1 + self.config.num_classes
        self.ignore_threshold = 0.5
        self.lambda_coord = 1
        
        self.sigmoid = nn.Sigmoid()
        self.tile = lambda x, y: ops.tile(x, y)
        self.concat = lambda x: ops.concat(x, axis=-1)
        self.reshape = ops.Reshape()
        self.input_shape = Tensor(tuple(config.img_shape[::-1]), ms.float32)
    
    def construct(self, x):
        num_batch = ops.shape(x)[0]
        grid_size = ops.shape(x)[2:4]
        
        prediction = ops.reshape(x, (num_batch, 
                                   self.num_anchors_per_scale,
                                   self.num_attrib,
                                   grid_size[0],
                                   grid_size[1]))
        prediction = ops.transpose(prediction, (0, 3, 4, 1, 2))
        
        range_x = range(grid_size[1])
        range_y = range(grid_size[0])
        grid_x = ops.Cast()(ops.tuple_to_array(range_x), ms.float32)
        grid_y = ops.Cast()(ops.tuple_to_array(range_y), ms.float32)
        
        grid_x = self.tile(ops.reshape(grid_x, (1, 1, -1, 1, 1)), 
                          (1, grid_size[0], 1, 1, 1))
        grid_y = self.tile(ops.reshape(grid_y, (1, -1, 1, 1, 1)), 
                          (1, 1, grid_size[1], 1, 1))
        grid = self.concat((grid_x, grid_y))
        
        box_xy = prediction[:, :, :, :, :2]
        box_wh = prediction[:, :, :, :, 2:4]
        box_confidence = prediction[:, :, :, :, 4:5]
        box_probs = prediction[:, :, :, :, 5:]
        
        box_xy = (self.sigmoid(box_xy) + grid) / ops.Cast()(ops.tuple_to_array(
            (grid_size[1], grid_size[0])), ms.float32)
        box_wh = ops.exp(box_wh) * self.anchors / self.input_shape
        box_confidence = self.sigmoid(box_confidence)
        box_probs = self.sigmoid(box_probs)
        
        if self.training:
            return grid, prediction, box_xy, box_wh
        return box_xy, box_wh, box_confidence, box_probs

# IoU计算
class Iou(nn.Cell):
    def __init__(self):
        super(Iou, self).__init__()
    
    def construct(self, box1, box2):
        box1_xy = box1[:, :, :, :, :, :2]
        box1_wh = box1[:, :, :, :, :, 2:4]
        
        box1_mins = box1_xy - box1_wh / Tensor(2.0)
        box1_maxs = box1_xy + box1_wh / Tensor(2.0)
        
        box2_xy = box2[:, :, :, :, :, :2]
        box2_wh = box2[:, :, :, :, :, 2:4]
        
        box2_mins = box2_xy - box2_wh / Tensor(2.0)
        box2_maxs = box2_xy + box2_wh / Tensor(2.0)
        
        intersect_mins = ops.maximum(box1_mins, box2_mins)
        intersect_maxs = ops.minimum(box1_maxs, box2_maxs)
        
        intersect_wh = ops.maximum(intersect_maxs - intersect_mins, Tensor(0.0))
        
        intersect_area = ops.squeeze(intersect_wh[:, :, :, :, :, 0:1], -1) * \
                        ops.squeeze(intersect_wh[:, :, :, :, :, 1:2], -1)
        box1_area = ops.squeeze(box1_wh[:, :, :, :, :, 0:1], -1) * \
                   ops.squeeze(box1_wh[:, :, :, :, :, 1:2], -1)
        box2_area = ops.squeeze(box2_wh[:, :, :, :, :, 0:1], -1) * \
                   ops.squeeze(box2_wh[:, :, :, :, :, 1:2], -1)
        
        iou = intersect_area / (box1_area + box2_area - intersect_area)
        return iou

# YOLO损失块
class YoloLossBlock(nn.Cell):
    def __init__(self, scale, config):
        super(YoloLossBlock, self).__init__()
        self.config = config
        
        if scale == 's':
            idx = (0, 1, 2)
        elif scale == 'm':
            idx = (3, 4, 5)
        elif scale == 'l':
            idx = (6, 7, 8)
        else:
            raise KeyError("Invalid scale value for DetectionBlock")
        
        self.anchors = Tensor([self.config.anchor_scales[i] for i in idx], ms.float32)
        self.ignore_threshold = Tensor(self.config.ignore_threshold, ms.float32)
        self.concat = lambda x: ops.concat(x, axis=-1)
        self.iou = Iou()
        self.cross_entropy = ops.SigmoidCrossEntropyWithLogits()
        self.reduce_sum = ops.ReduceSum()
        self.reduce_max = ops.ReduceMax(keep_dims=False)
        self.input_shape = Tensor(tuple(config.img_shape[::-1]), ms.float32)
    
    def construct(self, grid, prediction, pred_xy, pred_wh, y_true, gt_box):
        object_mask = y_true[:, :, :, :, 4:5]
        class_probs = y_true[:, :, :, :, 5:]
        
        grid_shape = ops.shape(prediction)[1:3]
        grid_shape = ops.Cast()(ops.tuple_to_array(grid_shape[::-1]), ms.float32)
        
        pred_boxes = self.concat((pred_xy, pred_wh))
        true_xy = y_true[:, :, :, :, :2] * grid_shape - grid
        true_wh = y_true[:, :, :, :, 2:4]
        true_wh = ops.select(ops.equal(true_wh, 0.0),
                           ops.fill(ops.DType()(true_wh), ops.shape(true_wh), 1.0),
                           true_wh)
        true_wh = ops.log(true_wh / self.anchors * self.input_shape)
        box_loss_scale = 2 - y_true[:, :, :, :, 2:3] * y_true[:, :, :, :, 3:4]
        
        gt_shape = ops.shape(gt_box)
        gt_box = ops.reshape(gt_box, (gt_shape[0], 1, 1, 1, gt_shape[1], gt_shape[2]))
        
        iou = self.iou(ops.ExpandDims()(pred_boxes, -2), gt_box)
        best_iou = self.reduce_max(iou, -1)
        
        ignore_mask = best_iou < self.ignore_threshold
        ignore_mask = ops.Cast()(ignore_mask, ms.float32)
        ignore_mask = ops.ExpandDims()(ignore_mask, -1)
        ignore_mask = ops.stop_gradient(ignore_mask)
        
        xy_loss = object_mask * box_loss_scale * self.cross_entropy(
            prediction[:, :, :, :, :2], true_xy)
        wh_loss = object_mask * box_loss_scale * 0.5 * ops.square(
            true_wh - prediction[:, :, :, :, 2:4])
        confidence_loss = self.cross_entropy(prediction[:, :, :, :, 4:5], object_mask)
        confidence_loss = object_mask * confidence_loss + \
                         (1 - object_mask) * confidence_loss * ignore_mask
        class_loss = object_mask * self.cross_entropy(prediction[:, :, :, :, 5:], class_probs)
        
        xy_loss = self.reduce_sum(xy_loss, ())
        wh_loss = self.reduce_sum(wh_loss, ())
        confidence_loss = self.reduce_sum(confidence_loss, ())
        class_loss = self.reduce_sum(class_loss, ())
        
        loss = xy_loss + wh_loss + confidence_loss + class_loss
        return loss / ops.shape(prediction)[0]

# 完整的YOLOv3网络
class yolov3_resnet18(nn.Cell):
    def __init__(self, config):
        super(yolov3_resnet18, self).__init__()
        self.config = config
        
        self.feature_map = YOLOv3(feature_shape=self.config.feature_shape,
                                backbone=ResNet(BasicBlock,
                                              self.config.backbone_layers,
                                              self.config.backbone_input_shape,
                                              self.config.backbone_shape,
                                              self.config.backbone_stride,
                                              num_classes=None),
                                backbone_shape=self.config.backbone_shape,
                                out_channel=self.config.out_channel)
        
        self.detect_1 = DetectionBlock('l', self.config)
        self.detect_2 = DetectionBlock('m', self.config)
        self.detect_3 = DetectionBlock('s', self.config)
    
    def construct(self, x):
        big_object_output, medium_object_output, small_object_output = self.feature_map(x)
        output_big = self.detect_1(big_object_output)
        output_me = self.detect_2(medium_object_output)
        output_small = self.detect_3(small_object_output)
        return output_big, output_me, output_small

# 数据预处理函数
def preprocess_fn(image, box, file, is_training):
    config_anchors = []
    temp = ConfigYOLOV3ResNet18.anchor_scales
    for i in temp:
        config_anchors += list(i)
    
    anchors = np.array([float(x) for x in config_anchors]).reshape(-1, 2)
    do_hsv = False
    max_boxes = ConfigYOLOV3ResNet18._NUM_BOXES
    num_classes = ConfigYOLOV3ResNet18.num_classes
    
    def _rand(a=0., b=1.):
        return np.random.rand() * (b - a) + a
    
    def _preprocess_true_boxes(true_boxes, anchors, in_shape=None):
        num_layers = anchors.shape[0] // 3
        anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        true_boxes = np.array(true_boxes, dtype='float32')
        input_shape = np.array(in_shape, dtype='int32')
        
        boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2.
        boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
        true_boxes[..., 0:2] = boxes_xy / input_shape[::-1]
        true_boxes[..., 2:4] = boxes_wh / input_shape[::-1]
        
        grid_shapes = [input_shape // 32, input_shape // 16, input_shape // 8]
        y_true = [np.zeros((grid_shapes[l][0], grid_shapes[l][1], 
                           len(anchor_mask[l]), 5 + num_classes), 
                          dtype='float32') for l in range(num_layers)]
        
        anchors = np.expand_dims(anchors, 0)
        anchors_max = anchors / 2.
        anchors_min = -anchors_max
        valid_mask = boxes_wh[..., 0] >= 1
        wh = boxes_wh[valid_mask]
        
        if len(wh) >= 1:
            wh = np.expand_dims(wh, -2)
            boxes_max = wh / 2.
            boxes_min = -boxes_max
            intersect_min = np.maximum(boxes_min, anchors_min)
            intersect_max = np.minimum(boxes_max, anchors_max)
            intersect_wh = np.maximum(intersect_max - intersect_min, 0.)
            intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
            box_area = wh[..., 0] * wh[..., 1]
            anchor_area = anchors[..., 0] * anchors[..., 1]
            iou = intersect_area / (box_area + anchor_area - intersect_area)
            best_anchor = np.argmax(iou, axis=-1)
            
            for t, n in enumerate(best_anchor):
                for l in range(num_layers):
                    if n in anchor_mask[l]:
                        i = np.floor(true_boxes[t, 0] * grid_shapes[l][1]).astype('int32')
                        j = np.floor(true_boxes[t, 1] * grid_shapes[l][0]).astype('int32')
                        k = anchor_mask[l].index(n)
                        c = true_boxes[t, 4].astype('int32')
                        y_true[l][j, i, k, 0:4] = true_boxes[t, 0:4]
                        y_true[l][j, i, k, 4] = 1.
                        y_true[l][j, i, k, 5 + c] = 1.
        
        pad_gt_box0 = np.zeros(shape=[ConfigYOLOV3ResNet18._NUM_BOXES, 4], dtype=np.float32)
        pad_gt_box1 = np.zeros(shape=[ConfigYOLOV3ResNet18._NUM_BOXES, 4], dtype=np.float32)
        pad_gt_box2 = np.zeros(shape=[ConfigYOLOV3ResNet18._NUM_BOXES, 4], dtype=np.float32)
        
        mask0 = np.reshape(y_true[0][..., 4:5], [-1])
        gt_box0 = np.reshape(y_true[0][..., 0:4], [-1, 4])
        gt_box0 = gt_box0[mask0 == 1]
        pad_gt_box0[:gt_box0.shape[0]] = gt_box0
        
        mask1 = np.reshape(y_true[1][..., 4:5], [-1])
        gt_box1 = np.reshape(y_true[1][..., 0:4], [-1, 4])
        gt_box1 = gt_box1[mask1 == 1]
        pad_gt_box1[:gt_box1.shape[0]] = gt_box1
        
        mask2 = np.reshape(y_true[2][..., 4:5], [-1])
        gt_box2 = np.reshape(y_true[2][..., 0:4], [-1, 4])
        gt_box2 = gt_box2[mask2 == 1]
        pad_gt_box2[:gt_box2.shape[0]] = gt_box2
        
        return y_true[0], y_true[1], y_true[2], pad_gt_box0, pad_gt_box1, pad_gt_box2
    
    def _infer_data(img_data, input_shape, box):
        w, h = img_data.size
        input_h, input_w = input_shape
        scale = min(float(input_w) / float(w), float(input_h) / float(h))
        nw = int(w * scale)
        nh = int(h * scale)
        
        img_data = img_data.resize((nw, nh), Image.Resampling.BICUBIC)
        new_image = np.zeros((input_h, input_w, 3), np.float32)
        new_image.fill(128)
        img_data = np.array(img_data)
        if len(img_data.shape) == 2:
            img_data = np.expand_dims(img_data, axis=-1)
            img_data = np.concatenate([img_data, img_data, img_data], axis=-1)
        
        dh = int((input_h - nh) / 2)
        dw = int((input_w - nw) / 2)
        new_image[dh:(nh + dh), dw:(nw + dw), :] = img_data
        new_image /= 255.
        new_image = np.transpose(new_image, (2, 0, 1))
        new_image = np.expand_dims(new_image, 0)
        return new_image, np.array([h, w], np.float32), box
    
    def _data_aug(image, box, is_training, jitter=0.3, hue=0.1, sat=1.5, val=1.5, 
                  image_size=(352, 640)):
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        iw, ih = image.size
        ori_image_shape = np.array([ih, iw], np.int32)
        h, w = image_size
        
        if not is_training:
            return _infer_data(image, image_size, box)
        
        flip = _rand() < .5
        box_data = np.zeros((max_boxes, 5))
        flag = 0
        
        while True:
            new_ar = float(w) / float(h) * _rand(1 - jitter, 1 + jitter) / \
                    _rand(1 - jitter, 1 + jitter)
            scale = _rand(0.25, 2)
            
            if new_ar < 1:
                nh = int(scale * h)
                nw = int(nh * new_ar)
            else:
                nw = int(scale * w)
                nh = int(nw / new_ar)
            
            dx = int(_rand(0, w - nw))
            dy = int(_rand(0, h - nh))
            flag = flag + 1
            
            if len(box) >= 1:
                t_box = box.copy()
                np.random.shuffle(t_box)
                t_box[:, [0, 2]] = t_box[:, [0, 2]] * float(nw) / float(iw) + dx
                t_box[:, [1, 3]] = t_box[:, [1, 3]] * float(nh) / float(ih) + dy
                if flip:
                    t_box[:, [0, 2]] = w - t_box[:, [2, 0]]
                t_box[:, 0:2][t_box[:, 0:2] < 0] = 0
                t_box[:, 2][t_box[:, 2] > w] = w
                t_box[:, 3][t_box[:, 3] > h] = h
                box_w = t_box[:, 2] - t_box[:, 0]
                box_h = t_box[:, 3] - t_box[:, 1]
                t_box = t_box[np.logical_and(box_w > 1, box_h > 1)]
                if len(t_box) >= 1:
                    box = t_box
                    break
        
        box_data[:len(box)] = box
        image = image.resize((nw, nh), Image.Resampling.BICUBIC)
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image = new_image
        
        if flip:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        
        gray = _rand() < .25
        if gray:
            image = image.convert('L').convert('RGB')
        
        image = np.array(image)
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
            image = np.concatenate([image, image, image], axis=-1)
        
        hue = _rand(-hue, hue)
        sat = _rand(1, sat) if _rand() < .5 else 1 / _rand(1, sat)
        val = _rand(1, val) if _rand() < .5 else 1 / _rand(1, val)
        image_data = image / 255.
        
        bbox_true_1, bbox_true_2, bbox_true_3, gt_box1, gt_box2, gt_box3 = \
            _preprocess_true_boxes(box_data, anchors, image_size)
        
        return image_data, bbox_true_1, bbox_true_2, bbox_true_3, \
               ori_image_shape, gt_box1, gt_box2, gt_box3
    
    if is_training:
        images, bbox_1, bbox_2, bbox_3, image_shape, gt_box1, gt_box2, gt_box3 = \
            _data_aug(image, box, is_training)
        return images, bbox_1, bbox_2, bbox_3, gt_box1, gt_box2, gt_box3
    
    images, shape, anno = _data_aug(image, box, is_training)
    return images, shape, anno, file

# XML处理函数
from xml.dom.minidom import parse
import xml.dom.minidom
from PIL import Image

def xy_local(collection, element):
    xy = collection.getElementsByTagName(element)[0]
    xy = xy.childNodes[0].data
    return xy

def filter_valid_data(image_dir):
    label_id = {'person': 0, 'face': 1, 'mask': 2}
    all_files = os.listdir(image_dir)
    
    image_dict = {}
    image_files = []
    
    for i in all_files:
        if (i[-3:] == 'jpg' or i[-4:] == 'jpeg') and i not in image_dict:
            image_files.append(i)
            label = []
            xml_path = os.path.join(image_dir, i[:-3] + 'xml')
            
            if not os.path.exists(xml_path):
                label = [[0, 0, 0, 0, 0]]
                image_dict[i] = label
                continue
            
            DOMTree = xml.dom.minidom.parse(xml_path)
            collection = DOMTree.documentElement
            object_ = collection.getElementsByTagName("object")
            
            for m in object_:
                temp = []
                name = m.getElementsByTagName('name')[0]
                class_num = label_id[name.childNodes[0].data]
                bndbox = m.getElementsByTagName('bndbox')[0]
                xmin = xy_local(bndbox, 'xmin')
                ymin = xy_local(bndbox, 'ymin')
                xmax = xy_local(bndbox, 'xmax')
                ymax = xy_local(bndbox, 'ymax')
                temp.extend([int(xmin), int(ymin), int(xmax), int(ymax), class_num])
                label.append(temp)
            
            image_dict[i] = label
    
    return image_files, image_dict

def data_to_mindrecord_byte_image(image_dir, mindrecord_dir, prefix, file_num):
    mindrecord_path = os.path.join(mindrecord_dir, prefix)
    writer = FileWriter(mindrecord_path, file_num)
    image_files, image_anno_dict = filter_valid_data(image_dir)
    
    yolo_json = {
        "image": {"type": "bytes"},
        "annotation": {"type": "int32", "shape": [-1, 5]},
        "file": {"type": "string"},
    }
    writer.add_schema(yolo_json, "yolo_json")
    
    for image_name in image_files:
        image_path = os.path.join(image_dir, image_name)
        with open(image_path, 'rb') as f:
            img = f.read()
        annos = np.array(image_anno_dict[image_name], dtype=np.int32)
        row = {"image": img, "annotation": annos, "file": image_name}
        writer.write_raw_data([row])
    
    writer.commit()

def create_yolo_dataset(mindrecord_dir, batch_size=32, repeat_num=1, device_num=1, 
                       rank=0, is_training=True, num_parallel_workers=8):
    ds = dataset.MindDataset(mindrecord_dir, columns_list=["image", "annotation", "file"],
                           num_shards=device_num, shard_id=rank,
                           num_parallel_workers=num_parallel_workers, shuffle=is_training)
    
    decode = vision.Decode()
    ds = ds.map(operations=decode, input_columns=["image"])
    
    compose_map_func = (lambda image, annotation, file: 
                       preprocess_fn(image, annotation, file, is_training))
    
    if is_training:
        hwc_to_chw = vision.HWC2CHW()
        ds = ds.map(operations=compose_map_func, input_columns=["image", "annotation", "file"],
                   output_columns=["image", "bbox_1", "bbox_2", "bbox_3", "gt_box1", "gt_box2", "gt_box3"],
                   num_parallel_workers=num_parallel_workers)
        ds = ds.project(["image", "bbox_1", "bbox_2", "bbox_3", "gt_box1", "gt_box2", "gt_box3"])
        ds = ds.map(operations=hwc_to_chw, input_columns=["image"],
                   num_parallel_workers=num_parallel_workers)
        ds = ds.batch(batch_size, drop_remainder=True)
        ds = ds.repeat(repeat_num)
    else:
        ds = ds.map(operations=compose_map_func, input_columns=["image", "annotation", "file"],
                   output_columns=["image", "image_shape", "annotation", "file"],
                   num_parallel_workers=num_parallel_workers)
        ds = ds.project(["image", "image_shape", "annotation", "file"])
    
    return ds

# YOLOv3损失网络
class YoloWithLossCell(nn.Cell):
    def __init__(self, network, config):
        super(YoloWithLossCell, self).__init__()
        self.yolo_network = network
        self.config = config
        self.loss_big = YoloLossBlock('l', self.config)
        self.loss_me = YoloLossBlock('m', self.config)
        self.loss_small = YoloLossBlock('s', self.config)
    
    def construct(self, x, y_true_0, y_true_1, y_true_2, gt_0, gt_1, gt_2):
        yolo_out = self.yolo_network(x)
        loss_l = self.loss_big(yolo_out[0][0], yolo_out[0][1], yolo_out[0][2], 
                              yolo_out[0][3], y_true_0, gt_0)
        loss_m = self.loss_me(yolo_out[1][0], yolo_out[1][1], yolo_out[1][2], 
                             yolo_out[1][3], y_true_1, gt_1)
        loss_s = self.loss_small(yolo_out[2][0], yolo_out[2][1], yolo_out[2][2], 
                               yolo_out[2][3], y_true_2, gt_2)
        return loss_l + loss_m + loss_s

# 训练包装器
class TrainingWrapper(nn.Cell):
    def __init__(self, network, optimizer, sens=1.0):
        super(TrainingWrapper, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        self.weights = ms.ParameterTuple(network.trainable_params())
        self.optimizer = optimizer
        self.grad = ops.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens
        self.reducer_flag = False
        self.grad_reducer = None
        self.parallel_mode = ms.get_auto_parallel_context("parallel_mode")
        
        if self.parallel_mode in [ms.ParallelMode.DATA_PARALLEL, 
                                 ms.ParallelMode.HYBRID_PARALLEL]:
            self.reducer_flag = True
            if self.reducer_flag:
                mean = ms.get_auto_parallel_context("gradients_mean")
                degree = ms.get_auto_parallel_context("device_num")
                self.grad_reducer = nn.DistributedGradReducer(optimizer.parameters, 
                                                             mean, degree)
    
    def construct(self, *args):
        weights = self.weights
        loss = self.network(*args)
        sens = ops.fill(ops.DType()(loss), ops.shape(loss), self.sens)
        grads = self.grad(self.network, weights)(*args, sens)
        
        if self.reducer_flag:
            grads = self.grad_reducer(grads)
        
        return ops.depend(loss, self.optimizer(grads))

# 评估相关类
class YoloBoxScores(nn.Cell):
    def __init__(self, config):
        super(YoloBoxScores, self).__init__()
        self.input_shape = Tensor(np.array(config.img_shape), ms.float32)
        self.num_classes = config.num_classes
    
    def construct(self, box_xy, box_wh, box_confidence, box_probs, image_shape):
        batch_size = ops.shape(box_xy)[0]
        x = box_xy[:, :, :, :, 0:1]
        y = box_xy[:, :, :, :, 1:2]
        box_yx = ops.concat((y, x), -1)
        w = box_wh[:, :, :, :, 0:1]
        h = box_wh[:, :, :, :, 1:2]
        box_hw = ops.concat((h, w), -1)
        
        new_shape = ops.round(image_shape * ops.ReduceMin()(self.input_shape / image_shape))
        offset = (self.input_shape - new_shape) / 2.0 / self.input_shape
        scale = self.input_shape / new_shape
        box_yx = (box_yx - offset) * scale
        box_hw = box_hw * scale
        
        box_min = box_yx - box_hw / 2.0
        box_max = box_yx + box_hw / 2.0
        boxes = ops.concat((box_min[:, :, :, :, 0:1],
                          box_min[:, :, :, :, 1:2],
                          box_max[:, :, :, :, 0:1],
                          box_max[:, :, :, :, 1:2]), -1)
        image_scale = ops.tile(image_shape, (1, 2))
        boxes = boxes * image_scale
        boxes = ops.reshape(boxes, (batch_size, -1, 4))
        boxes_scores = box_confidence * box_probs
        boxes_scores = ops.reshape(boxes_scores, (batch_size, -1, self.num_classes))
        return boxes, boxes_scores

class YoloWithEval(nn.Cell):
    def __init__(self, network, config):
        super(YoloWithEval, self).__init__()
        self.yolo_network = network
        self.box_score_0 = YoloBoxScores(config)
        self.box_score_1 = YoloBoxScores(config)
        self.box_score_2 = YoloBoxScores(config)
    
    def construct(self, x, image_shape):
        yolo_output = self.yolo_network(x)
        boxes_0, boxes_scores_0 = self.box_score_0(*yolo_output[0], image_shape)
        boxes_1, boxes_scores_1 = self.box_score_1(*yolo_output[1], image_shape)
        boxes_2, boxes_scores_2 = self.box_score_2(*yolo_output[2], image_shape)
        
        boxes = ops.concat((boxes_0, boxes_1, boxes_2), 1)
        boxes_scores = ops.concat((boxes_scores_0, boxes_scores_1, boxes_scores_2), 1)
        return boxes, boxes_scores, image_shape

# 评估指标计算
def calc_iou(bbox_pred, bbox_ground):
    x1 = bbox_pred[0]
    y1 = bbox_pred[1]
    width1 = bbox_pred[2] - bbox_pred[0]
    height1 = bbox_pred[3] - bbox_pred[1]
    
    x2 = bbox_ground[0]
    y2 = bbox_ground[1]
    width2 = bbox_ground[2] - bbox_ground[0]
    height2 = bbox_ground[3] - bbox_ground[1]
    
    endx = max(x1 + width1, x2 + width2)
    startx = min(x1, x2)
    width = width1 + width2 - (endx - startx)
    
    endy = max(y1 + height1, y2 + height2)
    starty = min(y1, y2)
    height = height1 + height2 - (endy - starty)
    
    if width <= 0 or height <= 0:
        iou = 0
    else:
        area = width * height
        area1 = width1 * height1
        area2 = width2 * height2
        iou = area * 1. / (area1 + area2 - area)
    
    return iou

def apply_nms(all_boxes, all_scores, thres, max_boxes):
    x1 = all_boxes[:, 0]
    y1 = all_boxes[:, 1]
    x2 = all_boxes[:, 2]
    y2 = all_boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    
    order = all_scores.argsort()[::-1]
    keep = []
    
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        if len(keep) >= max_boxes:
            break
        
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= thres)[0]
        order = order[inds + 1]
    
    return keep

def metrics(pred_data):
    config = ConfigYOLOV3ResNet18()
    num_classes = config.num_classes
    count_corrects = [1e-6 for _ in range(num_classes)]
    count_grounds = [1e-6 for _ in range(num_classes)]
    count_preds = [1e-6 for _ in range(num_classes)]
    
    for i, sample in enumerate(pred_data):
        gt_anno = sample["annotation"]
        box_scores = sample['box_scores']
        boxes = sample['boxes']
        mask = box_scores >= config.obj_threshold
        
        boxes_ = []
        scores_ = []
        classes_ = []
        max_boxes = config.nms_max_num
        
        for c in range(num_classes):
            class_boxes = np.reshape(boxes, [-1, 4])[np.reshape(mask[:, c], [-1])]
            class_box_scores = np.reshape(box_scores[:, c], [-1])[np.reshape(mask[:, c], [-1])]
            nms_index = apply_nms(class_boxes, class_box_scores, config.nms_threshold, max_boxes)
            class_boxes = class_boxes[nms_index]
            class_box_scores = class_box_scores[nms_index]
            classes = np.ones_like(class_box_scores, 'int32') * c
            
            boxes_.append(class_boxes)
            scores_.append(class_box_scores)
            classes_.append(classes)
        
        boxes = np.concatenate(boxes_, axis=0)
        classes = np.concatenate(classes_, axis=0)
        
        count_correct = [1e-6 for _ in range(num_classes)]
        count_ground = [1e-6 for _ in range(num_classes)]
        count_pred = [1e-6 for _ in range(num_classes)]
        
        for anno in gt_anno:
            count_ground[anno[4]] += 1
        
        for box_index, box in enumerate(boxes):
            bbox_pred = [box[1], box[0], box[3], box[2]]
            count_pred[classes[box_index]] += 1
            
            for anno in gt_anno:
                class_ground = anno[4]
                if classes[box_index] == class_ground:
                    iou = calc_iou(bbox_pred, anno)
                    if iou >= 0.5:
                        count_correct[class_ground] += 1
                        break
        
        count_corrects = [count_corrects[i] + count_correct[i] for i in range(num_classes)]
        count_preds = [count_preds[i] + count_pred[i] for i in range(num_classes)]
        count_grounds = [count_grounds[i] + count_ground[i] for i in range(num_classes)]
    
    precision = np.array([count_corrects[ix] / count_preds[ix] for ix in range(num_classes)])
    recall = np.array([count_corrects[ix] / count_grounds[ix] for ix in range(num_classes)])
    return precision, recall

# 网络参数初始化
def init_net_param(network, init_value='ones'):
    params = network.trainable_params()
    for p in params:
        if isinstance(p.data, Tensor) and 'beta' not in p.name and \
           'gamma' not in p.name and 'bias' not in p.name:
            p.set_data(initializer(init_value, p.data.shape, p.data.dtype))

# 训练主函数
def main(args_opt):
    device_target = ms.get_context("device_target")
    mode = ms.GRAPH_MODE
    
    if device_target == "GPU":
        ms.set_context(mode=mode, device_target="GPU", device_id=args_opt.device_id)
    elif device_target == "Ascend":
        ms.set_context(mode=mode, device_target="Ascend", device_id=args_opt.device_id)
    else:
        ms.set_context(mode=mode, device_target="CPU")
    
    if args_opt.distribute:
        device_num = args_opt.device_num
        ms.reset_auto_parallel_context()
        ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL,
                                   gradients_mean=True, device_num=device_num)
        init()
        rank = args_opt.device_id % device_num
    else:
        rank = 0
        device_num = 1
    
    loss_scale = float(args_opt.loss_scale)
    dataset = create_yolo_dataset(args_opt.mindrecord_file,
                                batch_size=args_opt.batch_size,
                                device_num=device_num, rank=rank)
    dataset_size = dataset.get_dataset_size()
    print('The epoch size:', dataset_size)
    print("Create dataset done!")
    
    net = yolov3_resnet18(ConfigYOLOV3ResNet18())
    net = YoloWithLossCell(net, ConfigYOLOV3ResNet18())
    init_net_param(net, "XavierUniform")
    
    ckpt_config = CheckpointConfig(
        save_checkpoint_steps=dataset_size * args_opt.save_checkpoint_epochs,
        keep_checkpoint_max=args_opt.keep_checkpoint_max)
    ckpoint_cb = ModelCheckpoint(prefix="yolov3", directory=args_opt.ckpt_dir, 
                               config=ckpt_config)
    
    if args_opt.pre_trained and os.path.exists(args_opt.pre_trained):
        if args_opt.pre_trained_epoch_size <= 0:
            raise KeyError("pre_trained_epoch_size must be greater than 0.")
        param_dict = load_checkpoint(args_opt.pre_trained)
        load_param_into_net(net, param_dict)
        print("Load pre_trained")
    
    lr = args_opt.lr
    opt = nn.Adam(filter(lambda x: x.requires_grad, net.get_parameters()), 
                 lr, loss_scale=loss_scale)
    net = TrainingWrapper(net, opt, loss_scale)
    
    callback = [LossMonitor(10 * dataset_size), ckpoint_cb]
    model = Model(net)
    dataset_sink_mode = args_opt.dataset_sink_mode
    
    print("Start train YOLOv3, the first epoch will be slower because of graph compilation.")
    model.train(args_opt.epoch_size, dataset, callbacks=callback,
               dataset_sink_mode=dataset_sink_mode)

# 预测函数
def tobox(boxes, box_scores):
    config = ConfigYOLOV3ResNet18()
    num_classes = config.num_classes
    mask = box_scores >= config.obj_threshold
    boxes_ = []
    scores_ = []
    classes_ = []
    max_boxes = config.nms_max_num
    
    for c in range(num_classes):
        class_boxes = np.reshape(boxes, [-1, 4])[np.reshape(mask[:, c], [-1])]
        class_box_scores = np.reshape(box_scores[:, c], [-1])[np.reshape(mask[:, c], [-1])]
        nms_index = apply_nms(class_boxes, class_box_scores, config.nms_threshold, max_boxes)
        class_boxes = class_boxes[nms_index]
        class_box_scores = class_box_scores[nms_index]
        classes = np.ones_like(class_box_scores, 'int32') * c
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)
    
    boxes = np.concatenate(boxes_, axis=0)
    classes = np.concatenate(classes_, axis=0)
    scores = np.concatenate(scores_, axis=0)
    return boxes, classes, scores

def yolo_eval(cfg_test):
    ds = create_yolo_dataset(cfg_test.mindrecord_file, batch_size=1, is_training=False)
    config = ConfigYOLOV3ResNet18()
    net = yolov3_resnet18(config)
    eval_net = YoloWithEval(net, config)
    
    print("Load Checkpoint!")
    print(cfg_test.ckpt_path)
    param_dict = load_checkpoint(cfg_test.ckpt_path)
    load_param_into_net(net, param_dict)
    
    eval_net.set_train(False)
    total = ds.get_dataset_size()
    print("\n========================================")
    print("total images num:", total)
    print("Processing, please wait a moment.")
    
    pred_data = []
    num_class = {0: 'person', 1: 'face', 2: 'mask'}
    
    for data in ds.create_dict_iterator(output_numpy=True):
        img_np = data['image']
        image_shape = data['image_shape']
        annotation = data['annotation']
        image_file = data['file']
        image_file = image_file.tobytes().decode('ascii')
        
        eval_net.set_train(False)
        output = eval_net(Tensor(img_np), Tensor(image_shape))
        
        for batch_idx in range(img_np.shape[0]):
            boxes = output[0].asnumpy()[batch_idx]
            box_scores = output[1].asnumpy()[batch_idx]
            boxes, classes, scores = tobox(boxes, box_scores)
            
            pred_data.append({
                "boxes": boxes,
                "classes": classes,
                "scores": scores,
                "annotation": annotation[batch_idx]
            })
    
    precision, recall = metrics(pred_data)
    print("Precision:", precision)
    print("Recall:", recall)
    return pred_data

# 主程序入口
if __name__ == "__main__":
    import argparse
    from easydict import EasyDict as edict
    
    # 训练配置
    cfg = edict({
        "distribute": False,
        "device_id": 0,
        "device_num": 1,
        "dataset_sink_mode": True,
        "lr": 0.001,
        "epoch_size": 10,
        "batch_size": 32,
        "loss_scale": 1024,
        "pre_trained": './pre_trained/pre.ckpt', #预训练模型路径 
        "pre_trained_epoch_size": 0,
        "ckpt_dir": "./ckpt", #检查点保存目录 
        "save_checkpoint_epochs": 1,
        "keep_checkpoint_max": 5,
        "train_url": './output', #训练输出目录 
    })
    
    data_path = './data/' #总数据集路径 
    mindrecord_dir_train = os.path.join(data_path, 'mindrecord/train')
    prefix = "yolo.mindrecord"
    cfg.mindrecord_file = os.path.join(mindrecord_dir_train, prefix)
    
    if not os.path.exists(mindrecord_dir_train):
        os.makedirs(mindrecord_dir_train)
    
    image_dir = os.path.join(data_path, "train")
    if not os.path.exists(cfg.mindrecord_file + "0"):
        print("Create Mindrecord...")
        data_to_mindrecord_byte_image(image_dir, mindrecord_dir_train, prefix, 1)
        print("Create Mindrecord Done!")
    
    main(cfg)