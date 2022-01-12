import numpy as np
import torch
from torchvision.ops import box_iou


class FathomNetEvaluator:
    def __init__(self, dataset, device, iou_thresh):
        self.dataset = dataset
        self.device = device
        self.iou_thresh = iou_thresh
        self.num_classes = len(dataset.label_mapping)
        self.metrics_by_class = {
            cls: {
                'TP': [],
                'FP': [],
                'conf': [],
                'tot_GT': 0
            }
            for cls in range(1, self.num_classes + 1)
        }

    def update(self, targets, predictions):
        target_boxes_all = targets[0]['boxes']
        target_labels_all = targets[0]['labels']
        pred_boxes_all = predictions[0]['boxes']
        pred_labels_all = predictions[0]['labels']
        pred_scores_all = predictions[0]['scores']

        for cls in range(1, self.num_classes + 1):
            true_boxes = target_boxes_all[target_labels_all == cls]
            pred_boxes = pred_boxes_all[pred_labels_all == cls]
            pred_scores = pred_scores_all[pred_labels_all == cls]

            iou = box_iou(true_boxes, pred_boxes)
            seen = np.zeros(len(true_boxes))
            if true_boxes.numel() == 0:
                self.metrics_by_class[cls]['TP'].extend([0] * len(pred_boxes))
                self.metrics_by_class[cls]['FP'].extend([1] * len(pred_boxes))
                self.metrics_by_class[cls]['conf'].extend(pred_scores.tolist())
                continue
            # Order predictions in descending order of max iou
            pred_order = torch.argsort(iou.max(dim=0).values, descending=True)
            for j in pred_order:
                i = torch.argmax(iou[:, j])
                if iou[i, j] >= self.iou_thresh and seen[i] == 0:
                    # True positive
                    seen[i] = 1
                    self.metrics_by_class[cls]['TP'].append(1)
                    self.metrics_by_class[cls]['FP'].append(0)
                    self.metrics_by_class[cls]['conf'].append(pred_scores[j].item())
                else:
                    # False positive
                    self.metrics_by_class[cls]['TP'].append(0)
                    self.metrics_by_class[cls]['FP'].append(1)
                    self.metrics_by_class[cls]['conf'].append(pred_scores[j].item())
            # True positives + False negatives
            self.metrics_by_class[cls]['tot_GT'] += len(true_boxes)

    def prec_rec_for_class(self, cls):
        # Calculate cumulative metrics by descending confidence order
        conf_order = np.argsort(self.metrics_by_class[cls]['conf'])[::-1]
        cum_tp = np.cumsum(np.array(self.metrics_by_class[cls]['TP'])[conf_order])
        cum_fp = np.cumsum(np.array(self.metrics_by_class[cls]['FP'])[conf_order])
        if len(cum_tp) + len(cum_fp) == 0:
            raise ZeroDivisionError("No predictions for class")
        if self.metrics_by_class[cls]['tot_GT'] == 0:
            raise ZeroDivisionError("No instances in test set")
        precision = cum_tp / (cum_tp + cum_fp)
        recall = cum_tp / self.metrics_by_class[cls]['tot_GT']
        return precision, recall

    def summarise(self, method="101"):
        res = {}
        for cls in self.metrics_by_class:
            try:
                precision, recall = self.prec_rec_for_class(cls)
            except ZeroDivisionError as e:
                res[self.dataset.get_class_name(cls)] = e
                continue

            if method == "101":
                # Implements the MS COCO calculation of AP
                # Calculate interpolated precision, then compute AP as the mean of 101 evenly spaced sample points
                precision_ip = np.maximum.accumulate(precision[::-1])[::-1]
                sample_points = np.linspace(0., 1., 101)
                AP = np.mean(np.append(precision, 0.0)[np.searchsorted(recall, sample_points)])
            elif method == "all_points":
                # Implements the PASCAL VOC 2010 challenge interpolation
                # Calculate all points interpolated precision, then compute AP as AUC
                precision_ip = np.maximum.accumulate(precision[::-1])[::-1]
                AP = np.sum(np.diff(recall, prepend=0.0) * precision_ip)
            else:
                raise NotImplementedError("Only supports methods '101' and 'all_points'")
            res[self.dataset.get_class_name(cls)] = AP
        res['mAP'] = np.nanmean(list(val for val in res.values() if not isinstance(val, str)))
        return res
