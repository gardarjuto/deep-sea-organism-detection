import numpy as np
import torch
from matplotlib import pyplot as plt
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
                'tot_GT': 0,
                'intersection_area': [],
                'union_area': []
            }
            for cls in range(1, self.num_classes + 1)
        }

    def update(self, targets, predictions, im_width, im_height):
        assert im_width > im_height
        target_boxes_all = targets['boxes']
        target_labels_all = targets['labels']
        pred_boxes_all = predictions['boxes']
        pred_labels_all = predictions['labels']
        pred_scores_all = predictions['scores']

        for cls in range(1, self.num_classes + 1):
            true_boxes = target_boxes_all[target_labels_all == cls]
            pred_boxes = pred_boxes_all[pred_labels_all == cls]
            pred_scores = pred_scores_all[pred_labels_all == cls]

            # Calculate intersection and union area
            gt_bitmap = np.zeros((im_height, im_width))
            pred_bitmap = np.zeros((im_height, im_width))

            for box in true_boxes:
                box = box.to(torch.long)
                gt_bitmap[box[1]:box[3], box[0]:box[1]] = 1
            for box in pred_boxes:
                box = box.to(torch.long)
                pred_bitmap[box[1]:box[3], box[0]:box[1]] = 1

            area_of_intersect = np.sum((gt_bitmap + pred_bitmap) > 1)
            area_of_union = np.sum((gt_bitmap + pred_bitmap) > 0)
            self.metrics_by_class[cls]['intersection_area'].append(area_of_intersect)
            self.metrics_by_class[cls]['union_area'].append(area_of_union)

            # Calculate TP and FP
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

    def iou_for_class(self, cls):
        if sum(self.metrics_by_class[cls]['union_area']) == 0:
            raise ZeroDivisionError("No instances or predictions in test set")
        iou1 = sum(self.metrics_by_class[cls]['intersection_area']) / sum(self.metrics_by_class[cls]['union_area'])
        iou2 = np.mean(np.array(self.metrics_by_class[cls]['intersection_area']) / np.array(self.metrics_by_class[cls]['union_area']))
        return iou1, iou2

    def summarise(self, method="101"):
        AP_res = {}
        IoU_res = {}
        for cls in self.metrics_by_class:
            try:
                IoU_res[self.dataset.get_class_name(cls)] = self.iou_for_class(cls)
            except ZeroDivisionError as e:
                IoU_res[self.dataset.get_class_name(cls)] = e
            try:
                precision, recall = self.prec_rec_for_class(cls)
            except ZeroDivisionError as e:
                AP_res[self.dataset.get_class_name(cls)] = e
                continue

            if method == "101":
                # Implements the MS COCO calculation of AP
                # Calculate interpolated precision, then compute AP as the mean of 101 evenly spaced sample points
                precision_ip = np.maximum.accumulate(precision[::-1])[::-1]
                sample_points = np.linspace(0., 1., 101)
                AP = np.mean(np.append(precision_ip, 0.0)[np.searchsorted(recall, sample_points)])
            elif method == "all_points":
                # Implements the PASCAL VOC 2010 challenge interpolation
                # Calculate all points interpolated precision, then compute AP as AUC
                precision_ip = np.maximum.accumulate(precision[::-1])[::-1]
                AP = np.sum(np.diff(recall, prepend=0.0) * precision_ip)
            else:
                raise NotImplementedError("Only supports methods '101' and 'all_points'")
            AP_res[self.dataset.get_class_name(cls)] = AP
        AP_res['mAP'] = np.mean(list(val if not isinstance(val, ZeroDivisionError) and np.isfinite(val) else 0.0
                                     for val in AP_res.values()))
        IoU_res['mIoU'] = (np.mean(list(val[0] for val in IoU_res.values() if not isinstance(val, ZeroDivisionError))),
                           np.mean(list(val[1] for val in IoU_res.values() if not isinstance(val, ZeroDivisionError))))
        return AP_res, IoU_res

    def plot_precision_recall(self, interpolate=True):
        """Returns a grid with one plot for each class"""
        nrows = int(np.sqrt(len(self.metrics_by_class)))
        ncols = int(np.ceil(len(self.metrics_by_class) / nrows))
        fig, axes = plt.subplots(nrows, ncols, figsize=(15, 15), sharex=True, sharey=True)

        for ax, cls in zip(axes.ravel() if nrows > 1 or ncols > 1 else [axes], self.metrics_by_class):
            try:
                precision, recall = self.prec_rec_for_class(cls)
            except ZeroDivisionError as e:
                ax.set_title(e)
                continue
            ax.plot(recall, precision, '--', label='original', color='grey')
            if interpolate:
                precision_ip = np.maximum.accumulate(precision[::-1])[::-1]
                ax.step(recall, precision_ip, label='interpolated', color='black')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.legend()
            ax.grid()
            ax.title.set_text(self.dataset.get_class_name(cls))
        fig.add_subplot(111, frameon=False)
        plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        plt.xlabel("Recall", fontsize=14)
        plt.ylabel("Precision", fontsize=14)
        return axes

    def get_hard_negatives(self, targets, predictions):
        target_boxes = targets['boxes']
        pred_boxes = predictions['boxes']
        pred_labels = predictions['labels']
        pred_scores = predictions['scores']

        if target_boxes.numel() == 0:
            return pred_boxes[pred_labels > 0], pred_scores[pred_labels > 0]
        elif pred_boxes.numel() == 0:
            return torch.empty(0), torch.empty(0)

        ious = box_iou(target_boxes, pred_boxes)
        negative_idx = (ious.max(dim=0).values < self.iou_thresh) & (pred_labels > 0)

        return pred_boxes[negative_idx], pred_scores[negative_idx]
