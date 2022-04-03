import numpy as np
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from skimage.transform import resize
from skimage.feature import hog
import torchvision
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FasterRCNN, FastRCNNPredictor
import torchvision.transforms.functional as F

from detection import datasets, utils


class HOG:
    def __init__(self, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3), block_norm='L2-Hys',
                 gamma_corr=False, resize_to=(64, 64)):
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.block_norm = block_norm
        self.gamma_corr = gamma_corr
        self.resize_to = resize_to

    def extract_all(self, dataset, cpus=1):
        """Extracts features from all samples from dataset"""
        descriptors = []
        labels = []
        if isinstance(dataset, datasets.FathomNetDataset):
            def process_func(img, box, label):
                x0, y0, x1, y1 = box.int()
                cropped = F.crop(img, y0, x0, y1 - y0, x1 - x0)
                fd = self.extract(cropped)
                return fd, label.item()

            res = [pair for pairs in Parallel(n_jobs=cpus)(delayed(self.extract_from_sample)(img, targets)
                                                           for img, targets in dataset) for pair in pairs]
        elif isinstance(dataset, datasets.FathomNetCroppedDataset):
            res = Parallel(n_jobs=cpus)(
                delayed(lambda img, label: (self.extract(img), label))(img, label) for img, label in dataset)
        else:
            raise NotImplementedError("Dataset of wrong type.")
        descriptors, labels = list(map(list, zip(*res)))
        return descriptors, labels

    def extract_from_sample(self, image, targets):
        res = []
        for box, label in zip(targets['boxes'], targets['labels']):
            x0, y0, x1, y1 = box.int()
            cropped = F.crop(image, y0, x0, y1 - y0, x1 - x0)
            fd = self.extract(cropped)
            res.append((fd, label.item()))
        return res

    def extract(self, image):
        im_resized = resize(image, self.resize_to, anti_aliasing=True)
        fd = hog(im_resized, self.orientations, self.pixels_per_cell, self.cells_per_block, self.block_norm,
                 transform_sqrt=self.gamma_corr, multichannel=True)
        return fd


def load_model(name, num_classes, pretrained=False, progress=True):
    if name == 'rcnn_resnet50_fpn':
        backbone = torchvision.models.detection.backbone_utils.resnet_fpn_backbone('resnet50', pretrained=pretrained)
        model = FasterRCNN(backbone, num_classes=num_classes)
    elif name == 'rcnn_resnet101_fpn':
        backbone = torchvision.models.detection.backbone_utils.resnet_fpn_backbone('resnet101', pretrained=pretrained)
        model = FasterRCNN(backbone, num_classes=num_classes)
    elif name == 'rcnn_resnet34_fpn':
        backbone = torchvision.models.detection.backbone_utils.resnet_fpn_backbone('resnet34', pretrained=pretrained)
        model = FasterRCNN(backbone, num_classes=num_classes)
    elif name == 'rcnn_resnet18_fpn':
        backbone = torchvision.models.detection.backbone_utils.resnet_fpn_backbone('resnet18', pretrained=pretrained)
        model = FasterRCNN(backbone, num_classes=num_classes)
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
