import copy
import types

import PIL
import cv2
import torch
from joblib import Parallel, delayed
from skimage.transform import resize
from skimage.feature import hog
import torchvision
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FasterRCNN
import torchvision.transforms.functional as F
from torchvision.ops.misc import FrozenBatchNorm2d
import numpy as np

from detection import datasets


def flip_horizontal(image, targets):
    new_targets = copy.deepcopy(targets)
    boxes = new_targets['boxes']
    widths = boxes[:, 2] - boxes[:, 0]
    boxes[:, 0] = image.shape[2] - boxes[:, 0] - widths
    boxes[:, 2] = boxes[:, 0] + widths
    new_targets['boxes'] = boxes
    return image[:, ::-1], targets


def apply_rotation(image, targets, angle):
    """Rotates image and targets and expands image to avoid cropping"""
    h, w, _ = image.shape

    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)

    cos = np.cos(np.deg2rad(angle))
    sin = np.sin(np.deg2rad(angle))

    # find width and height after rotation
    new_w = int(h * abs(sin) + w * abs(cos))
    new_h = int(h * abs(cos) + w * abs(sin))

    # translate rotation matrix to image center within the new bounds
    matrix[0, 2] += (new_w - w) / 2
    matrix[1, 2] += (new_h - h) / 2

    # rotate image with the new bounds and translated rotation matrix
    rotated_image = cv2.warpAffine(image, matrix, (new_w, new_h))

    new_targets = copy.deepcopy(targets)
    boxes = new_targets['boxes']

    # translate boxes to new image
    boxes[:, [0, 2]] += (new_w - w) / 2
    boxes[:, [1, 3]] += (new_h - h) / 2

    left_upper = torch.cat((boxes[:, [0, 1]].T, torch.ones(1, boxes.shape[0])))
    right_lower = torch.cat((boxes[:, [2, 3]].T, torch.ones(1, boxes.shape[0])))
    right_upper = torch.cat((boxes[:, [2, 1]].T, torch.ones(1, boxes.shape[0])))
    left_lower = torch.cat((boxes[:, [0, 3]].T, torch.ones(1, boxes.shape[0])))

    box_matrix = torch.from_numpy(cv2.getRotationMatrix2D((new_w / 2, new_h / 2), angle, 1.0).astype(np.float32))
    new_left_upper = box_matrix @ left_upper
    new_right_upper = box_matrix @ right_upper
    new_left_lower = box_matrix @ left_lower
    new_right_lower = box_matrix @ right_lower

    xs = torch.vstack((new_left_upper[0], new_right_upper[0], new_left_lower[0], new_right_lower[0]))
    ys = torch.vstack((new_left_upper[1], new_right_upper[1], new_left_lower[1], new_right_lower[1]))

    x_min = xs.min(dim=0).values
    y_min = ys.min(dim=0).values
    x_max = xs.max(dim=0).values
    y_max = ys.max(dim=0).values

    new_boxes = torch.stack((x_min, y_min, x_max, y_max), dim=1)
    new_targets['boxes'] = new_boxes
    return rotated_image, new_targets


class HOG:
    def __init__(self, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3), block_norm='L2-Hys',
                 gamma_corr=False, resize_to=(64, 64), center_crop=False):
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.block_norm = block_norm
        self.gamma_corr = gamma_corr
        self.resize_to = resize_to
        self.center_crop = center_crop

    def extract_all(self, dataset, cpus=1, horizontal_flip=False, rotations=None):
        """Extracts features from all samples in dataset with optional image augmentation"""
        pool = Parallel(n_jobs=cpus)
        res = pool(delayed(self.extract_from_sample)(np.array(img), targets) for img, targets in dataset)
        if horizontal_flip:
            res.extend(
                pool(delayed(self.extract_from_sample)(flip_horizontal(np.array(img), targets))
                     for img, targets in dataset)
            )
        if rotations:
            for angle in rotations:
                res.extend(
                    pool(delayed(self.extract_from_sample)(apply_rotation(np.array(img), targets, angle))
                         for img, targets in dataset)
                )
                res.extend(
                    pool(delayed(self.extract_from_sample)(apply_rotation(*flip_horizontal(np.array(img), targets),
                                                                          angle))
                         for img, targets in dataset)
                )

        res = [pair for pairs in res for pair in pairs]
        descriptors, labels = list(map(list, zip(*res)))
        return descriptors, labels

    def extract_from_sample(self, image, targets):
        res = []
        for box, label in zip(targets['boxes'], targets['labels']):
            x0, y0, x1, y1 = box.int()
            cropped = image[y0:y1, x0:x1]
            fd = self.extract(cropped)
            res.append((fd, label.item()))
        return res

    def extract(self, image):
        im_resized = resize(image, self.resize_to, anti_aliasing=True)
        fd = hog(im_resized, self.orientations, self.pixels_per_cell, self.cells_per_block, self.block_norm,
                 transform_sqrt=self.gamma_corr, multichannel=True)
        if self.center_crop:
            h, w, _ = image.shape
            im_resized2 = resize(image[h // 4:3 * h // 4, w // 4:3 * w // 4], self.resize_to, anti_aliasing=True)
            fd2 = hog(im_resized2, self.orientations, self.pixels_per_cell, self.cells_per_block, self.block_norm,
                      transform_sqrt=self.gamma_corr, multichannel=True)
            fd = np.concatenate([fd, fd2])
        return fd


def get_resnet_features(name, trainable_layers=3, norm_layer=FrozenBatchNorm2d, pretrained=True):
    model = torchvision.models.resnet.__dict__[name](norm_layer=norm_layer, pretrained=pretrained)
    if trainable_layers < 0 or trainable_layers > 5:
        raise ValueError(f"Trainable layers should be in the range [0,5], got {trainable_layers}")
    layers_to_train = ["layer4", "layer3", "layer2", "layer1", "conv1"][:trainable_layers]
    if trainable_layers == 5:
        layers_to_train.append("bn1")
    for name, parameter in model.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    model._forward_impl = types.MethodType(_forward_impl, model)
    return model


def load_model(name, num_classes, trainable_layers=3, pretrained=True):
    if name == 'rcnn_resnet50_fpn':
        backbone = torchvision.models.detection.backbone_utils.resnet_fpn_backbone('resnet50',
                                                                                   trainable_layers=trainable_layers,
                                                                                   pretrained=pretrained)
        model = FasterRCNN(backbone, num_classes=num_classes)
    elif name == 'rcnn_resnet101_fpn':
        backbone = torchvision.models.detection.backbone_utils.resnet_fpn_backbone('resnet101',
                                                                                   trainable_layers=trainable_layers,
                                                                                   pretrained=pretrained)
        model = FasterRCNN(backbone, num_classes=num_classes)
    elif name == 'rcnn_resnet34_fpn':
        backbone = torchvision.models.detection.backbone_utils.resnet_fpn_backbone('resnet34',
                                                                                   trainable_layers=trainable_layers,
                                                                                   pretrained=pretrained)
        model = FasterRCNN(backbone, num_classes=num_classes)
    elif name == 'rcnn_resnet18_fpn':
        backbone = torchvision.models.detection.backbone_utils.resnet_fpn_backbone('resnet18',
                                                                                   trainable_layers=trainable_layers,
                                                                                   pretrained=pretrained)
        model = FasterRCNN(backbone, num_classes=num_classes)
    elif name == 'rcnn_resnet18':
        backbone = get_resnet_features('resnet18', trainable_layers=trainable_layers, pretrained=pretrained)
        backbone.out_channels = 512
        anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)
        model = FasterRCNN(backbone, num_classes=num_classes, rpn_anchor_generator=anchor_generator,
                           box_roi_pool=roi_pooler)
    elif name == 'rcnn_resnet50':
        backbone = get_resnet_features('resnet50', trainable_layers=trainable_layers, pretrained=pretrained)
        backbone.out_channels = 2048
        anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)
        model = FasterRCNN(backbone, num_classes=num_classes, rpn_anchor_generator=anchor_generator,
                           box_roi_pool=roi_pooler)
    elif name == 'rcnn_resnet101':
        backbone = get_resnet_features('resnet18', trainable_layers=trainable_layers, pretrained=pretrained)
        backbone.out_channels = 512
        anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)
        model = FasterRCNN(backbone, num_classes=num_classes, rpn_anchor_generator=anchor_generator,
                           box_roi_pool=roi_pooler)
    elif name == 'rcnn_mobilenet_v2':
        backbone = torchvision.models.mobilenet_v2(pretrained=True).features
        backbone.out_channels = 1280
        anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)
        model = FasterRCNN(backbone, num_classes=num_classes, rpn_anchor_generator=anchor_generator,
                           box_roi_pool=roi_pooler)
    elif name == 'rcnn_vgg19_bn':
        backbone = torchvision.models.vgg19_bn(pretrained=True).features
        backbone.out_channels = 512
        anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)
        model = FasterRCNN(backbone, num_classes=num_classes, rpn_anchor_generator=anchor_generator,
                           box_roi_pool=roi_pooler)
    else:
        raise NotImplementedError('Currently supported model names: rcnn_resnet50_fpn and rcnn_resnet101_fpn')
    return model
