import os
import logging
import sys
import random

import cv2
import numpy as np
import torch
import torch.cuda
import torch.distributed as dist
from matplotlib import pyplot as plt, patches
from matplotlib.colors import SymLogNorm
from skimage.util import view_as_windows


def initialise_distributed(args):
    """Initialises environment for distributed training on GPUs."""
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        logging.info("Not using distributed mode")
        args.distributed = False
        return
    args.rank = int(os.environ["RANK"])
    args.world_size = int(os.environ["WORLD_SIZE"])
    args.gpu = int(os.environ["LOCAL_RANK"])
    args.distributed = True
    torch.cuda.set_device(args.gpu)
    dist.init_process_group(backend='nccl', init_method=args.dist_url, world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier(device_ids=[args.gpu])


def get_world_size():
    if "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"])
    return 1


def initialise_logging(args):
    """Initialise logging environment."""
    if not is_master_process():
        logging.disable()
        return
    level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(level, int):
        raise ValueError(f"Invalid log level: {args.log_level}")
    if args.log_file:
        logging.basicConfig(filename=args.log_file, filemode='w', level=level, format='[%(asctime)s] %(message)s',
                            datefmt='%I:%M:%S %p')
    else:
        logging.basicConfig(stream=sys.stdout, level=level, format='[%(asctime)s] %(message)s', datefmt='%I:%M:%S %p')


def collate_fn(batch):
    return tuple(zip(*batch))


def is_master_process():
    if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
        return False
    return True


def save_state(checkpoint, output_dir, epoch):
    torch.save(checkpoint, os.path.join(output_dir, 'checkpoint.file'))
    torch.save(checkpoint, os.path.join(output_dir, f'model_{epoch}.file'))


def tensor_encode_id(img_id):
    """
    Encodes a FathomNet image id like '00a6db92-5277-4772-b019-5b89c6af57c3' as a tensor
    of shape torch.Size([4]) of four integers in the range [0, 2^32-1].
    """
    hex_str = img_id.replace('-', '')
    length = len(hex_str) // 4
    img_id_enc = tuple(int(hex_str[i * length: (i + 1) * length], 16) for i in range(4))
    return torch.tensor(img_id_enc)


def tensor_decode_id(img_id_enc):
    """Inverse function of tensor_encode_id"""
    ints = img_id_enc.tolist()
    img_id = ''.join([hex(part)[2:].zfill(8) for part in ints])
    for ind in [8, 13, 18, 23]:
        img_id = img_id[:ind] + '-' + img_id[ind:]
    return img_id


def sliding_window(image, win_size, step_size):
    """Implements sliding window over an image."""
    for j, row in enumerate(view_as_windows(image, win_size, step_size)):
        for i, col in enumerate(row):
            x, y = i * step_size[1], j * step_size[0]
            for window in col:
                yield x, y, window


def iou_one_to_many(box, boxes):
    """Computes one-to-many intersection over union"""
    x0y0 = np.maximum(boxes[:, :2], box[:2])
    x1y1 = np.minimum(boxes[:, 2:], box[2:])

    intersection = (x1y1[:, 0] - x0y0[:, 0]) * (x1y1[:, 1] - x0y0[:, 1])
    intersection[(x0y0[:, 0] > x1y1[:, 0]) | (x0y0[:, 1] > x1y1[:, 1])] = 0

    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    ious = intersection / (area + areas - intersection)
    return ious


def non_maxima_suppression(boxes, confidence_scores, threshold=0.5):
    """Performes non-maxima suppression filtering on bounding boxes."""
    filtered_boxes = []
    filtered_confidence = []
    while boxes.size > 0:
        i = confidence_scores.argmax()
        filtered_boxes.append(boxes[i])
        filtered_confidence.append(confidence_scores[i])
        ious = iou_one_to_many(boxes[i], boxes)
        boxes = boxes[ious <= threshold]
        confidence_scores = confidence_scores[ious <= threshold]
    return filtered_boxes, filtered_confidence


def plot_confusion_matrix(conf_mat, filename, labels=None, log_scale=False, show_values=False, cmap='viridis'):
    """Plots a confusion matrix with or without labels."""
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(conf_mat, cmap=cmap, norm=SymLogNorm(10) if log_scale else None, extent=[0, 1, 0, 1], origin='lower',
                   interpolation="nearest")
    fig.colorbar(im, ax=ax)
    if labels:
        ax.set_xticks(np.linspace(0.5 / len(labels), 1 - 0.5 / len(labels), len(labels)))
        ax.set_xticklabels(labels, rotation='vertical', fontsize=8)
        ax.set_yticks(np.linspace(0.5 / len(labels), 1 - 0.5 / len(labels), len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    if show_values:
        for (j, i), label in np.ndenumerate(conf_mat):
            ax.text((i + 0.5) / len(conf_mat), (j + 0.5) / len(conf_mat[0]), f"{label:.02f}", ha='center', va='center',
                    fontsize=8)

    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.savefig(filename, facecolor='w', bbox_inches='tight', dpi=200)


def make_deterministic(seed):
    """Forces a deterministic run."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    cv2.setRNGSeed(0)


def visualise_image(image, predictions=None, ground_truths=None, name_mapping=None, conf_thresh=None, c_pred='red',
                    c_gt='yellow'):
    """Visualises image with predicted and ground truth bounding boxes"""
    fig, ax = plt.subplots(figsize=(10, 10))
    if isinstance(image, torch.Tensor):
        ax.imshow(image.permute(1, 2, 0))
    else:
        ax.imshow(image)
    if predictions:
        for box, label, score in zip(predictions['boxes'], predictions['labels'], predictions['scores']):
            if conf_thresh and score < conf_thresh:
                continue
            # Create a Rectangle patch
            rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor=c_pred,
                                     facecolor='none')
            # Add the patch to the Axes
            ax.add_patch(rect)
            label = label.item()
            if name_mapping:
                label = name_mapping(label)
            plt.text(box[0] + 10, box[3] + 10, label + str(score), size=10, color='white',
                     ha="left", va="top",
                     bbox=dict(boxstyle="square",
                               ec=c_pred,
                               fc=c_pred,
                               alpha=0.5
                               )
                     )
    if ground_truths:
        for box, label in zip(ground_truths['boxes'], ground_truths['labels']):
            # Create a Rectangle patch
            rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor=c_gt,
                                     facecolor='none')
            # Add the patch to the Axes
            ax.add_patch(rect)
            label = label.item()
            if name_mapping:
                label = name_mapping(label)
            plt.text(box[0] + 10, box[3] + 10, label, size=10,
                     ha="left", va="top",
                     bbox=dict(boxstyle="square",
                               ec=c_gt,
                               fc=c_gt,
                               alpha=0.2
                               )
                     )


def log_summary(AP_res, IoU_res, iou_thresh, epoch=None, n_epochs=None):
    if epoch is not None and n_epochs is not None:
        logging.info(f"Summary [{epoch}/{n_epochs}]")
    else:
        logging.info(f"Summary")
    logging.info(f"\t(Average Precision @ {iou_thresh}): mAP={AP_res['mAP']:.3f}, "
                 + ", ".join([f"{key}={val}" if not isinstance(val, ZeroDivisionError)
                              else f"{key}={val}" for key, val in AP_res.items() if key != 'mAP']))
    logging.info(f"\t(IoU): mIoU=({IoU_res['mIoU']}), "
                 + ", ".join([f"{key}=({val}" if not isinstance(val, ZeroDivisionError)
                              else f"{key}={val}" for key, val in IoU_res.items() if key != 'mIoU']))
