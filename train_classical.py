import argparse
import cProfile
import json
import os
import random
import numpy as np
import torch
import re
import cv2
from torch.utils.data import DataLoader
import pickle

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
    parser.add_argument("--epochs", default=1, type=int, help="number of epochs to perform hard negative mining")
    parser.add_argument("-j", "--workers", default=0, type=int, metavar="N",
                        help="number of data loading workers (default: 4)")
    parser.add_argument("--data-augmentation", default="hflip", type=str,
                        help="data augmentation policy (default: hflip)")
    parser.add_argument("--iou-thresh", default=0.5, type=float, help="IoU threshold for evaluation")
    parser.add_argument("--log-file", "--lf", default=None, type=str,
                        help="path to file for writing logs. If omitted, writes to stdout")
    parser.add_argument("--log-level", default="ERROR", choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"))
    parser.add_argument("--seed", type=int, default=None,
                        help="Fix random generator seed. Setting this forces a deterministic run")
    parser.add_argument("--profile", action="store_true", help="profile the code and write to file 'restats'")

    # HOG parameters
    parser.add_argument("--hog-bins", default=9, type=int, help="number or bins for orientation binning in HOG")
    parser.add_argument("--ppc", default=(8, 8), nargs="+", type=int, help="pixels per cell in HOG")
    parser.add_argument("--cpb", default=(3, 3), nargs="+", type=int, help="cells per block in HOG")
    parser.add_argument("--block-norm", default="L2-Hys", type=str, help="block norm in HOG")
    parser.add_argument("--gamma-corr", action="store_true", help="use gamma correction in HOG")
    parser.add_argument("--hog-dim", default=(64, 64), nargs="+", type=int, help="input dimensions for HOG extractor")

    # Sliding window parameters
    parser.add_argument("--downscale-factor", default=1.25, type=float,
                        help="downscale factor in each iteration of gaussian pyramid")
    parser.add_argument("--min-window", default=(50, 50), type=int, nargs="+",
                        help="minimum pixel size of sliding window")
    parser.add_argument("--step-size", default=(20, 20), type=int, nargs="+",
                        help="step size in pixels of sliding window")

    # Evaluation parameters
    parser.add_argument("--log-every", "--pe", default=10, type=int, help="log every ith batch")
    parser.add_argument("--evaluate-only", action="store_true", help="only evaluate model")
    parser.add_argument("--output-dir", default=".", type=str, help="path to save outputs")
    parser.add_argument("--plot-pc", action="store_true", help="plot precision recall curves")

    # SVM parameters
    parser.add_argument("--C", default=1.0, type=float, help="C parameter for SVM")
    parser.add_argument("--loss", default="hinge", type=str, help="loss type for SVM")
    parser.add_argument("--max-iter", default=1000, type=int, help="maximum iterations in training SVM")

    # Hard negative mining parameters
    parser.add_argument("--neg-per-img", default=50, type=int, help="how many hard negatives to mine from each image")

    # Multicore parameters
    parser.add_argument("--cpus", default=1, type=int, help="number of cores to use for parallel operations")
    return parser


def main(args):
    # TODO: Check arguments

    utils.initialise_logging(args)
    logging.info("Started")

    if args.seed is not None:
        utils.make_deterministic(args.seed)
        logging.info(f"Seed set to {args.seed}")

    # Parse class definition file
    logging.info("Loading class definitions...")
    with open(args.class_file, "r") as f:
        classes = json.load(f)
    num_classes = len(classes) + 1
    logging.info(f"Training with {num_classes} classes: " + ", ".join(['background'] + list(classes.keys())))

    # Load datasets
    logging.info("Loading dataset...")
    train_dataset, val_dataset = datasets.load_train_val(name=args.dataset, train_path=args.train_path, classes=classes,
                                                         val_split=args.val_split)

    # Create feature extractor
    logging.info("Creating feature extractor...")
    feature_extractor = models.HOG(orientations=args.hog_bins, pixels_per_cell=args.ppc, cells_per_block=args.cpb,
                                   block_norm=args.block_norm, gamma_corr=args.gamma_corr, resize_to=args.hog_dim)

    # Extract features
    logging.info("Extracting features...")
    descriptors, labels = feature_extractor.extract_all(train_dataset, cpus=args.cpus)

    # Add one background GT for SVM
    logging.info(f"N={len(descriptors)},D={len(descriptors[0])}")
    logging.info(
        f"Min={min([descriptor.min() for descriptor in descriptors])}, Max={max([descriptor.max() for descriptor in descriptors])}")
    descriptors.append(descriptors[0])
    labels.append(0)

    # Train SVM
    logging.info("Training classifier on feature descriptors")
    clf = trainingtools.train_classifier(classifier_name, descriptors, labels, num_classes,
                                         pca_components=args.pca_components,
                                         feature_map_gamma=args.feature_map_gamma,
                                         feature_map_components=args.feature_map_components,
                                         sgd_alpha=args.sgd_alpha, class_weight=args.class_weight,
                                         fit_intercept=args.fit_intercept, max_iter=args.max_iter)

    cv2.setUseOptimized(True)
    cv2.setNumThreads(1)

    # Evaluate
    logging.info("Evaluating classifier on test dataset")
    trainingtools.evaluate_classifier(clf, feature_extractor=feature_extractor, dataset=val_dataset,
                                      iou_thresh=args.iou_thresh, log_every=args.log_every, output_dir=args.output_dir,
                                      plot_pc=args.plot_pc, cpus=args.cpus)

    # Apply training procedure
    for epoch in range(args.epochs):
        # Apply hard negative mining
        logging.info("Performing hard negative mining")
        negative_samples = trainingtools.mine_hard_negatives(clf, feature_extractor, train_dataset,
                                                             iou_thresh=args.iou_thresh, max_per_img=args.neg_per_img,
                                                             cpus=args.cpus)

        # Add hard negatives to training samples
        descriptors.extend(negative_samples)
        labels.extend([0] * len(negative_samples))
        logging.info(
            f"Added {len(negative_samples)} negative samples to the previous {len(descriptors) - len(negative_samples)} total")

        # Save descriptors
        with open(os.path.join(args.output_dir, f"saved_descriptors_SGDNystroem_epoch_{epoch}.pickle"), "wb") as f:
            pickle.dump({'descriptors': descriptors, 'labels': labels}, f)

        # Train SVM
        logging.info("Training classifier on feature descriptors")
        clf = trainingtools.train_svm(descriptors, labels, num_classes, pca_components=args.pca_components,
                                      feature_map_gamma=args.feature_map_gamma,
                                      feature_map_components=args.feature_map_components,
                                      sgd_alpha=args.sgd_alpha, class_weight=args.class_weight,
                                      fit_intercept=args.fit_intercept, max_iter=args.max_iter)

        # Evaluate
        logging.info("Evaluating classifier on test dataset")
        trainingtools.evaluate_classifier(clf, feature_extractor=feature_extractor, dataset=val_dataset,
                                          iou_thresh=args.iou_thresh, log_every=args.log_every,
                                          output_dir=args.output_dir, plot_pc=args.plot_pc, cpus=args.cpus)


if __name__ == '__main__':
    parsed_args = get_args_parser().parse_args()
    if parsed_args.profile:
        cProfile.run('main(parsed_args)', 'restats')
    else:
        main(parsed_args)
