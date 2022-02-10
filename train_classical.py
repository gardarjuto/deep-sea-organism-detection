import argparse
import json
import random
import numpy as np
import torch
from detection import models, utils, datasets, evaluation
from detection import trainingtools
import logging


def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description="Marine Organism Object Detection Training", add_help=add_help)

    parser.add_argument("--train-path", default="data/train", type=str, help="path to training dataset")
    parser.add_argument("--test-path", default="data/test", type=str, help="path to test dataset")
    parser.add_argument("--class-file", default="classes.json", type=str, help="path to class definitions")
    parser.add_argument("--dataset", default="FathomNet", type=str, help="dataset name")
    parser.add_argument("--val-split", "--tr", default=0.2, type=float,
                        help="proportion of training dataset to use for validation")
    parser.add_argument("-j", "--workers", default=4, type=int, metavar="N",
                        help="number of data loading workers (default: 4)")
    parser.add_argument("--output-dir", default=".", type=str, help="path to save outputs")
    parser.add_argument("--data-augmentation", default="hflip", type=str,
                        help="data augmentation policy (default: hflip)")
    parser.add_argument("--iou-thresh", default=0.5, type=float, help="IoU threshold for evaluation")
    parser.add_argument("--log-file", "--lf", default=None, type=str,
                        help="path to file for writing logs. If omitted, writes to stdout")
    parser.add_argument("--log-level", default="ERROR", choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"))
    parser.add_argument("--evaluate-only", action="store_true", help="Only evaluate model")
    parser.add_argument("--seed", type=int, default=None,
                        help="Fix random generator seed. Setting this forces a deterministic run")

    # HOG parameters
    parser.add_argument("--hog-bins", default=9, type=int, help="Number or bins for orientation binning in HOG")
    parser.add_argument("--ppc", default=(8, 8), nargs="+", type=int, help="Pixels per cell in HOG")
    parser.add_argument("--cpb", default=(3, 3), nargs="+", type=int, help="Cells per block in HOG")
    parser.add_argument("--block-norm", default="L2-Hys", type=str, help="Block norm in HOG")
    parser.add_argument("--gamma-corr", default=True, type=bool, help="Use gamma correction in HOG")
    parser.add_argument("--hog-dim", default=(64, 64), nargs="+", type=int, help="Input dimensions for HOG extractor")

    # Sliding window parameters
    parser.add_argument("--downscale-factor", default=1.25, type=float,
                        help="Downscale factor in each iteration of gaussian pyramid")
    return parser


def main(args):
    # TODO: Check arguments

    utils.initialise_logging(args)
    logging.info("Started")

    if args.seed is not None:
        logging.info(f"Seed set to {args.seed}")
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # Parse class definition file
    logging.info("Loading class definitions...")
    with open(args.class_file, "r") as f:
        classes = json.load(f)
    num_classes = len(classes)
    logging.info(f"Training with {num_classes} classes: " + ", ".join(['background'] + list(classes.keys())))

    # Load data
    logging.info("Loading dataset...")
    train_dataset, val_dataset = datasets.load_train_val(name=args.dataset, train_path=args.train_path, classes=classes,
                                                         val_split=args.val_split)

    # Extract features
    logging.info("Extracting features...")
    feature_extractor = models.HOG(orientations=args.hog_bins, pixels_per_cell=args.ppc, cells_per_block=args.cpb,
                                   block_norm=args.block_norm, gamma_corr=args.gamma_corr, resize_to=args.hog_dim)
    descriptors, labels = feature_extractor.extract_all(train_dataset)

    # Train SVM
    logging.info("Training SVM on feature descriptors")
    svm = trainingtools.train_svm(descriptors, labels, num_classes)

    # TODO: Implement hard negative mining
    # hn_samples = trainingtools.mine_hard_negative(train_dataset)

    # Evaluate
    logging.info("Evaluating SVM on test dataset")
    trainingtools.evaluate_svm(svm, feature_extractor=feature_extractor, dataset=val_dataset,
                                    iou_thresh=args.iou_thresh, output_dir=args.output_dir, plot_pc=True,
                                    downscale=args.downscale_factor)


if __name__ == '__main__':
    parsed_args = get_args_parser().parse_args()
    main(parsed_args)
