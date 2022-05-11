import math
import os
import warnings
import PIL.Image
import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from joblib import delayed, Parallel
from skimage.transform import pyramid_gaussian
from torchvision.ops import box_iou
from torchvision.transforms import transforms as T, functional as F
from torchvision.utils import draw_bounding_boxes
import logging

from detection import utils, evaluation, datasets


def train_one_epoch(model, loader, device, optimizer, epoch, n_epochs, log_every=None):
    """Trains the neural model for one epoch."""
    model.train()

    total_loss_classifier = 0.0
    total_loss_box_reg = 0.0
    total_loss_objectness = 0.0
    total_loss_rpn_box_reg = 0.0

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1e-3
        warmup_iters = min(1000, len(loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    for i, (images, targets) in enumerate(loader, start=1):
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        loss_value = losses.item()

        if not math.isfinite(loss_value):
            logging.error(f"Loss is {loss_value}, stopping training")
            logging.error(loss_dict)
            return -1

        total_loss_classifier += loss_dict['loss_classifier']
        total_loss_box_reg += loss_dict['loss_box_reg']
        total_loss_objectness += loss_dict['loss_objectness']
        total_loss_rpn_box_reg += loss_dict['loss_rpn_box_reg']

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        if log_every and i % log_every == 0:
            logging.info(f"Epoch [{epoch}/{n_epochs}]  [{i}/{len(loader)}]  LR={optimizer.param_groups[0]['lr']}  " +
                         ", ".join([f"{loss_type}={loss.item():.3f}" for loss_type, loss in loss_dict.items()]))

    logging.info(f'Summary:')
    logging.info(f'\tloss_classifier (mean): {total_loss_classifier.item() / len(loader):.3f}, '
                 f'loss_box_reg: {total_loss_box_reg.item() / len(loader):.3f}, '
                 f'loss_objectness: {total_loss_objectness.item() / len(loader):.3f}, '
                 f'loss_rpn_box_reg (mean): {total_loss_rpn_box_reg.item() / len(loader):.3f}')


@torch.inference_mode()
def visualise_prediction(model, device, img_name, dataset, show_ground_truth=True):
    """Visualise the predictions of a model with bounding boxes"""
    model.eval()
    idx = dataset.index_of(img_name)
    img, targets = dataset[idx]
    img = [img.to(device)]
    targets = {k: v.to(device) for k, v in targets.items()}
    prediction = model(img)[0]
    boxes = prediction['boxes']
    labels = ['pred_(' + dataset.get_class_name(lab) + ')' for lab in prediction['labels'].tolist()]
    colours = ['green'] * len(prediction['boxes'])
    if show_ground_truth:
        boxes = torch.cat((targets['boxes'], boxes), dim=0)
        labels = ['true_(' + dataset.get_class_name(lab) + ')' for lab in targets['labels'].tolist()] + labels
        colours = ['red'] * len(targets['boxes']) + colours
    image255 = Image.open(os.path.join(dataset.root, 'images', img_name)).convert('RGB')
    image255 = F.pil_to_tensor(image255)
    res = draw_bounding_boxes(image255, boxes, labels, colours, width=3)
    return F.to_pil_image(res)


@torch.inference_mode()
def evaluate(model, loader, device, epoch, iou_thresh=0.5, log_every=None, output_dir=None, plot_pc=False,
             save_to_file=None):
    """Evaluate the model on a test set. Produces an optional precision-recall plot"""
    model.eval()

    evaluator = evaluation.FathomNetEvaluator(dataset=loader.dataset, device=device, iou_thresh=iou_thresh)

    for i, (images, targets) in enumerate(loader, start=1):
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        predictions = model(images)
        predictions = [{k: v.to(device) for k, v in t.items()} for t in predictions]

        evaluator.update(targets[0], predictions[0], *images[0].shape[:-3:-1])

        if log_every and i % log_every == 0:
            logging.info(f"Test [{i}/{len(loader)}]")

    AP_res, IoU_res = evaluator.summarise(method="101")
    logging.info(f"Summary (Average Precision @ {iou_thresh}): mAP={AP_res['mAP']:.3f}, "
                 + ", ".join([f"{key}={val}" if not isinstance(val, ZeroDivisionError)
                              else f"{key}={val}" for key, val in AP_res.items() if key != 'mAP']))
    logging.info(f"Summary (IoU): mIoU=({IoU_res['mIoU1']:.3f}, {IoU_res['mIoU2']:.3f}), "
                 + ", ".join([f"{key}=({val}" if not isinstance(val, ZeroDivisionError)
                              else f"{key}={val}" for key, val in IoU_res.items() if
                              key != 'mIoU1' and key != 'mIoU2']))
    if plot_pc:
        axes = evaluator.plot_precision_recall(interpolate=True)
        plt.savefig(os.path.join(output_dir, f"precision_recall_e{epoch}.png"), dpi=300)
    if save_to_file:
        joblib.dump(evaluator, save_to_file)
        logging.info(f"Saved evaluator object to file {save_to_file}")
    return AP_res, IoU_res


def selective_search_roi(image, resize_height=250, quality=False):
    """Produces region proposals using the selective search algorithm."""
    scale_factor = resize_height / image.shape[0]
    resize_width = int(image.shape[1] * scale_factor)
    image = cv2.resize(image, (resize_width, resize_height))

    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    if quality:
        ss.switchToSelectiveSearchQuality()
    else:
        ss.switchToSelectiveSearchFast()

    bboxes = (ss.process() / scale_factor).astype(int)
    return bboxes


def get_detections_ss(image, resize_height=250):
    """Calls the selective search function. Exists for backward compatability."""
    bboxes = selective_search_roi(image, resize_height=resize_height)
    bboxes = bboxes[(bboxes[:, 2] > 3) & (bboxes[:, 3] > 3)]
    return bboxes


def get_detections(svm, feature_extractor, image, downscale=1.25, min_w=(50, 50, 3), step_size=(20, 20, 3),
                   visualise=False):
    """DEPRECATED. Use get_detections_ss instead."""
    warnings.warn("Use get_detections_ss instead", DeprecationWarning)
    total = 0
    n_detect = 0
    detections = {cl: [] for cl in svm.classes_}
    confidence = {cl: [] for cl in svm.classes_}

    image = np.transpose(image.numpy(), (1, 2, 0))

    scale = 0
    for im_scaled in pyramid_gaussian(image, downscale=downscale, multichannel=True):
        curr_detections = {cl: [] for cl in svm.classes_}
        curr_confidence = {cl: [] for cl in svm.classes_}
        scale_factor = downscale ** scale

        # If the width or height of the scaled image is less than
        # the width or height of the window, then end the iterations.
        if im_scaled.shape[0] < min_w[0] or im_scaled.shape[1] < min_w[1]:
            break
        for x, y, window in utils.sliding_window(im_scaled, min_w, step_size):
            pred, conf = get_classification(svm, feature_extractor, window)
            total += 1
            if pred > 0:
                n_detect += 1
                detections[pred[0]].append(
                    [int(x * scale_factor), int(y * scale_factor),
                     int((x + min_w[0]) * scale_factor), int((y + min_w[1]) * scale_factor)])
                confidence[pred[0]].append(conf)

                curr_detections[pred[0]].append(
                    [x, y, x + min_w[0], y + min_w[1]])
                curr_confidence[pred[0]].append(conf)
        # Move the the next scale
        scale += 1

    return detections, confidence


def get_classification(clf, sample):
    """Run classifier on sample. If using a DummyClassifier then return a random confidence."""
    pred = clf.predict(sample)
    if hasattr(clf, 'decision_function'):
        conf = clf.decision_function(sample).max()
    else:
        conf = np.random.random()
    return pred, conf


def get_predictions(obj_clf, feature_extractor, image, ss_height=250, bg_clf=None):
    """Get bounding box classifications from an image."""
    if isinstance(image, PIL.Image.Image):
        image = np.array(image)
    bboxes = get_detections_ss(image, resize_height=ss_height)

    detections = {cl: [] for cl in obj_clf.classes_}
    confidence = {cl: [] for cl in obj_clf.classes_}
    for (x, y, w, h) in bboxes:
        window = image[y:y + h, x:x + w]
            
        fd = feature_extractor.extract(window).reshape(1, -1)

        if bg_clf and bg_clf.predict(fd) == 0:
            continue
        pred, conf = get_classification(obj_clf, fd)
        if pred[0] > 0 and conf > 0:
            detections[pred[0]].append([x, y, x + w, y + h])
            confidence[pred[0]].append(conf)

    boxes = []
    labels = []
    scores = []
    for cls in detections:
        if not detections[cls]:
            continue
        filtered_boxes, filtered_confidence = utils.non_maxima_suppression(np.array(detections[cls]),
                                                                           np.array(confidence[cls]))
        boxes.extend(filtered_boxes)
        labels.extend([cls for _ in range(len(filtered_boxes))])
        scores.extend(filtered_confidence)
    predictions = {'boxes': torch.tensor(np.array(boxes).reshape(-1, 4)), 'labels': torch.tensor(labels),
                   'scores': torch.tensor(scores)}
    return predictions


def evaluate_classifier(clf, feature_extractor, dataset, iou_thresh=0.5, ss_height=250, output_dir=None, plot_pc=False,
                        cpus=1, save_to_file=None):
    """Evaluates a classical classifier on a test set. Supports parallel execution."""
    evaluator = evaluation.FathomNetEvaluator(dataset=dataset, device='cpu', iou_thresh=iou_thresh)
    logging.info(f"SVM has classes {clf.classes_}")

    prediction_list = Parallel(n_jobs=cpus)(
        delayed(lambda targets, *args: (targets, get_predictions(*args), image.size))(targets, clf, feature_extractor,
                                                                                      image, ss_height)
        for image, targets in dataset)
    for targets, predictions, shape in prediction_list:
        evaluator.update(targets, predictions, *shape)

    AP_res, IoU_res = evaluator.summarise(method="101")
    logging.info(f"Summary (Average Precision @ {iou_thresh}): mAP={AP_res['mAP']:.3f}, "
                 + ", ".join([f"{key}={val}" if not isinstance(val, ZeroDivisionError)
                              else f"{key}={val}" for key, val in AP_res.items() if key != 'mAP']))
    logging.info(f"Summary (IoU): mIoU=({IoU_res['mIoU1']:.3f}, {IoU_res['mIoU2']:.3f}), "
                 + ", ".join([f"{key}=({val}" if not isinstance(val, ZeroDivisionError)
                              else f"{key}={val}" for key, val in IoU_res.items() if
                              key != 'mIoU1' and key != 'mIoU2']))
    if plot_pc:
        axes = evaluator.plot_precision_recall(interpolate=True)
        plt.savefig(os.path.join(output_dir, f"precision_recall_svm.png"), dpi=300)
    if save_to_file:
        joblib.dump(evaluator, save_to_file)
        logging.info(f"Saved evaluator object to file {save_to_file}")
    return AP_res, IoU_res


def mine_hard_negatives(clf, feature_extractor, dataset, iou_thresh=0.5, max_per_img=None, cpus=1):
    """Perform mining on a dataset to extract negatives. Supports parallel execution."""
    hard_negatives = Parallel(n_jobs=cpus)(
        delayed(mine_single_img)(clf, feature_extractor, img, targets, iou_thresh=iou_thresh, limit=max_per_img)
        for img, targets in dataset)

    return np.array([fd for sublist in hard_negatives for fd in sublist], dtype=np.float32)


def mine_single_img(clf, feature_extractor, image, targets, iou_thresh=0.5, limit=None):
    """Perform hard negative mining on a single image"""
    if isinstance(image, PIL.Image.Image):
        image = np.array(image)
    predictions = get_predictions(clf, feature_extractor, image)
    pred_boxes = predictions['boxes']
    pred_labels = predictions['labels']
    pred_scores = predictions['scores']

    if targets['boxes'].numel() == 0:
        hn, hn_scores = pred_boxes[pred_labels > 0], pred_scores[pred_labels > 0]
    elif pred_boxes.numel() == 0:
        hn, hn_scores = torch.empty((0, 4)), torch.empty(0)
    else:
        ious = box_iou(targets['boxes'], pred_boxes)
        negative_idx = (ious.max(dim=0).values < iou_thresh) & (pred_labels > 0)
        hn, hn_scores = pred_boxes[negative_idx], pred_scores[negative_idx]

    hn_filtered = hn
    if limit:
        n = min(limit, len(hn_scores))
        top_n_idx = np.argpartition(hn_scores, -n)[-n:]
        hn_filtered = hn[top_n_idx]

    hard_negatives = []
    for box in hn_filtered:
        x0, y0, x1, y1 = box.int()
        cropped_img = image[y0:y1, x0:x1]
        fd = feature_extractor.extract(cropped_img)
        hard_negatives.append(fd)
    return hard_negatives


def evaluate_two_stage(bg_clf, obj_clf, feature_extractor, dataset, iou_thresh=0.5, ss_height=250, output_dir=None,
                       plot_pc=False, cpus=1):
    """Evaluate a two-stage classical classifier. Supports parallel execution."""
    evaluator = evaluation.FathomNetEvaluator(dataset=dataset, device='cpu', iou_thresh=iou_thresh)
    logging.info(f"Object classifier has classes {obj_clf.classes_}")

    prediction_list = Parallel(n_jobs=cpus)(
        delayed(lambda targets, *args: (targets, get_predictions(*args)))(targets, obj_clf, feature_extractor, image,
                                                                          ss_height, bg_clf)
        for image, targets in dataset)

    for targets, predictions in prediction_list:
        evaluator.update(targets, predictions)

    res = evaluator.summarise(method="101")
    logging.info(f"Summary (Average Precision @ {iou_thresh}): mAP={res['mAP']:.3f}, "
                 + ", ".join([f"{key}={val}" if not isinstance(val, ZeroDivisionError)
                              else f"{key}={val}" for key, val in res.items() if key != 'mAP']))
    if plot_pc:
        axes = evaluator.plot_precision_recall(interpolate=True)
        plt.savefig(os.path.join(output_dir, f"precision_recall_svm.png"), dpi=300)
    return res
