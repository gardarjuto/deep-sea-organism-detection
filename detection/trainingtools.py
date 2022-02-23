import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from matplotlib import patches
from skimage.transform import pyramid_gaussian
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from torchvision.transforms import transforms as T, functional as F
from torchvision.utils import draw_bounding_boxes
import logging

from detection import utils, evaluation, datasets


def train_one_epoch(model, loader, device, optimiser, epoch, n_epochs, log_every=None, scaler=None):
    model.train()

    total_loss_classifier = 0.0
    total_loss_box_reg = 0.0
    total_loss_objectness = 0.0
    total_loss_rpn_box_reg = 0.0

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimiser, start_factor=warmup_factor, total_iters=warmup_iters
        )

    for i, (images, targets) in enumerate(loader, start=1):
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        loss_value = losses.item()

        if not math.isfinite(loss_value):
            logging.error(f"Loss is {loss_value}, stopping training")
            logging.error(loss_dict)
            sys.exit(1)

        total_loss_classifier += loss_dict['loss_classifier']
        total_loss_box_reg += loss_dict['loss_box_reg']
        total_loss_objectness += loss_dict['loss_objectness']
        total_loss_rpn_box_reg += loss_dict['loss_rpn_box_reg']

        optimiser.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimiser)
            scaler.update()
        else:
            losses.backward()
            optimiser.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        if log_every and i % log_every == 0:
            logging.info(f"Epoch [{epoch}/{n_epochs}]  [{i}/{len(loader)}]  LR={optimiser.param_groups[0]['lr']}  " +
                         ", ".join([f"{loss_type}={loss.item():.3f}" for loss_type, loss in loss_dict.items()]))

    logging.info(f'Summary:')
    logging.info(f'\tloss_classifier (mean): {total_loss_classifier.item() / len(loader):.3f}, '
                 f'loss_box_reg: {total_loss_box_reg.item() / len(loader):.3f}, '
                 f'loss_objectness: {total_loss_objectness.item() / len(loader):.3f}, '
                 f'loss_rpn_box_reg (mean): {total_loss_rpn_box_reg.item() / len(loader):.3f}')


@torch.inference_mode()
def visualise_prediction(model, device, img_name, dataset, show_ground_truth=True):
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
def evaluate(model, loader, device, epoch, iou_thresh=0.5, log_every=None, output_dir=None, plot_pc=True):
    model.eval()

    evaluator = evaluation.FathomNetEvaluator(dataset=loader.dataset.dataset, device=device, iou_thresh=iou_thresh)

    for i, (images, targets) in enumerate(loader, start=1):
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        predictions = model(images)
        predictions = [{k: v.to(device) for k, v in t.items()} for t in predictions]

        evaluator.update(targets[0], predictions[0])

        if log_every and i % log_every == 0:
            logging.info(f"Test [{i}/{len(loader)}]")

    res = evaluator.summarise(method="101")
    logging.info(f"Summary (Average Precision @ {iou_thresh}): mAP={res['mAP']:.3f}, "
                 + ", ".join([f"{key}={val:.3f}" for key, val in res.items() if key != 'mAP']))
    if plot_pc:
        axes = evaluator.plot_precision_recall(interpolate=True)
        plt.savefig(os.path.join(output_dir, f"precision_recall_e{epoch}.png"), dpi=300)


def train_svm(descriptors, labels, num_classes, max_iter=1000):
    if num_classes != len(set(labels)):
        raise RuntimeError(f"Expected {num_classes} classes, got {len(set(labels))}.")
    clf = make_pipeline(StandardScaler(), LinearSVC(max_iter=max_iter))
    clf.fit(descriptors, labels)
    return clf


def get_detections(svm, feature_extractor, image, downscale=1.25, min_w=(50, 50, 3), step_size=(20, 20, 3), visualise=False):
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
            if conf > 0:
                detections[pred[0]].append(
                    [int(x * scale_factor), int(y * scale_factor),
                     int((x + min_w[0]) * scale_factor), int((y + min_w[1]) * scale_factor)])
                confidence[pred[0]].append(conf)

                curr_detections[pred[0]].append(
                    [x, y, x + min_w[0], y + min_w[1]])
                curr_confidence[pred[0]].append(conf)
        # Move the the next scale
        scale += 1

        if visualise:
            fig, ax = plt.subplots()
            plt.imshow(im_scaled)

            for cl in detections:
                for i, ((x0, y0, x1, y1), conf) in enumerate(zip(detections[cl], confidence[cl])):
                    # add bounding boxes to the image
                    box = patches.Rectangle(
                        (x0, y0), x1-x0, y1-y0, edgecolor="red", facecolor="none"
                    )

                    ax.add_patch(box)

                    rx, ry = box.get_xy()
                    cx = rx + box.get_width() / 2.0
                    cy = ry + box.get_height() / 8.0
                    l = ax.annotate(
                        f"{cl}, {conf:.1f}",
                        (cx, cy),
                        fontsize=8,
                        fontweight="bold",
                        color="white",
                        ha='center',
                        va='center'
                    )
                    l.set_bbox(
                        dict(facecolor='red', alpha=0.5, edgecolor='red')
                    )

            plt.axis('off')
            plt.show()

    return detections, confidence


def get_classification(svm, feature_extractor, image):
    fd = feature_extractor.extract(image)
    pred = svm.predict(fd.reshape(1, -1))
    conf = svm.decision_function(fd.reshape(1, -1)).max()
    return pred, conf


def evaluate_classifier(
        clf, feature_extractor, loader, iou_thresh=0.5, downscale=1.25, min_w=(50, 50, 3), step_size=(20, 20, 3),
        log_every=None, output_dir=None, plot_pc=True, visualise=False):
    evaluator = evaluation.FathomNetEvaluator(dataset=loader.dataset, device='cpu', iou_thresh=iou_thresh)
    logging.info(f"SVM has classes {clf.classes_}")
    for i, (images, targets) in enumerate(loader):
        detections, conf_scores = get_detections(clf, feature_extractor, images[0], downscale=downscale, min_w=min_w,
                                                 step_size=step_size, visualise=visualise)
        boxes = []
        labels = []
        scores = []
        for cls in detections:
            if not detections[cls]:
                continue
            filtered_boxes, filtered_confidence = utils.non_maxima_suppression(np.array(detections[cls]),
                                                                               np.array(conf_scores[cls]))
            boxes.extend(filtered_boxes)
            labels.extend([cls for _ in range(len(filtered_boxes))])
            scores.extend(filtered_confidence)
        predictions = {'boxes': torch.tensor(np.array(boxes)), 'labels': torch.tensor(labels),
                       'scores': torch.tensor(scores)}

        evaluator.update(targets[0], predictions)

        if log_every and i % log_every == 0:
            logging.info(f"Test [{i}/{len(loader)}]")

    res = evaluator.summarise(method="101")
    logging.info(f"Summary (Average Precision @ {iou_thresh}): mAP={res['mAP']:.3f}, "
                 + ", ".join([f"{key}={val:.3f}" if not isinstance(val, ZeroDivisionError)
                              else f"{key}={val}" for key, val in res.items() if key != 'mAP']))
    if plot_pc:
        axes = evaluator.plot_precision_recall(interpolate=True)
        plt.savefig(os.path.join(output_dir, f"precision_recall_svm.png"), dpi=300)


def mine_hard_negatives(
        svm, feature_extractor, loader, iou_thresh=0.5, downscale=1.25, min_w=(50, 50, 3), step_size=(20, 20, 3),
        max_per_img=100):
    evaluator = evaluation.FathomNetEvaluator(dataset=loader.dataset, device='cpu', iou_thresh=iou_thresh)
    hard_negatives = []

    for i, (img, targets) in enumerate(loader):
        print(f"Mining from image {i}/{len(loader)}")
        detections, conf_scores = get_detections(svm, feature_extractor, img[0], downscale=downscale, min_w=min_w,
                                                 step_size=step_size)
        boxes = []
        labels = []
        scores = []
        for cls in detections:
            if not detections[cls]:
                continue
            filtered_boxes, filtered_confidence = utils.non_maxima_suppression(np.array(detections[cls]),
                                                                               np.array(conf_scores[cls]))
            boxes.extend(filtered_boxes)
            labels.extend([cls for _ in range(len(filtered_boxes))])
            scores.extend(filtered_confidence)
        predictions = {'boxes': torch.tensor(np.array(boxes)), 'labels': torch.tensor(labels),
                       'scores': torch.tensor(scores)}
        hn, hn_scores = evaluator.get_hard_negatives(targets[0], predictions)
        n = min(max_per_img, len(hn_scores))
        top_n_idx = np.argpartition(hn_scores, -n)[-n:]

        for box in hn[top_n_idx]:
            x0, y0, x1, y1 = box.int()
            cropped_img = F.crop(img[0], y0, x0, y1 - y0, x1 - x0)
            fd = feature_extractor.extract(cropped_img)
            hard_negatives.append(fd)
    return hard_negatives
