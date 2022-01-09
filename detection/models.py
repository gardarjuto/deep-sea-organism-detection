import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def load_model(name, num_classes, pretrained=False, progress=True):
    if name == 'rcnn_resnet50_fpn':
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained, progress=progress)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    else:
        raise NotImplementedError
    return model
