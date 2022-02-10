from skimage.transform import resize
from skimage.feature import hog
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class HOG:
    def __init__(self, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3), block_norm='L2-Hys', gamma_corr=False, resize_to=(64, 64)):
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.block_norm = block_norm
        self.gamma_corr = gamma_corr
        self.resize_to = resize_to

    def extract_all(self, dataset):
        """Extracts features from all samples in dataset"""
        descriptors = []
        labels = []
        for im, label in dataset:
            fd = self.extract(im)
            descriptors.append(fd)
            labels.append(label)
        return descriptors, labels

    def extract(self, image):
        im_resized = resize(image, self.resize_to, anti_aliasing=True)
        fd = hog(im_resized, self.orientations, self.pixels_per_cell, self.cells_per_block, self.block_norm,
                 transform_sqrt=self.gamma_corr)
        return fd


def load_model(name, num_classes, pretrained=False, progress=True):
    if name == 'rcnn_resnet50_fpn':
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained, progress=progress)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    else:
        raise NotImplementedError('Currently supported model names: rcnn_resnet50_fpn')
    return model
