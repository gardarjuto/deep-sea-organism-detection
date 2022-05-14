import argparse
import cProfile
import json
import os
from time import time
import numpy as np
import joblib
import sklearn
import cv2
from sklearn.decomposition import PCA
from sklearn.kernel_approximation import Nystroem
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
import pickle

from detection import models, utils, datasets, evaluation
from detection import trainingtools
import logging


def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description="Marine Organism Object Detection Training using HOG+SVM",
                                     add_help=add_help)

    parser.add_argument("--train-path", default="data/train", type=str,
                        help="path to training dataset")
    parser.add_argument("--test-path", default="data/test", type=str,
                        help="path to test dataset")
    parser.add_argument("--class-file", default="classes.json", type=str,
                        help="path to class definitions")
    parser.add_argument("--load-model", default=None, type=str,
                        help="path if loading pretrained model")
    parser.add_argument("--val-split", default=0.2, type=float,
                        help="proportion of training dataset to use for validation")
    parser.add_argument("--iou-thresh", default=0.5, type=float,
                        help="IoU threshold for evaluation")
    parser.add_argument("--log-file", "--lf", default=None, type=str,
                        help="path to file for writing logs. If omitted, writes to stdout")
    parser.add_argument("--log-level", default="ERROR", choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
                        help="level of logging messages")
    parser.add_argument("--seed", type=int, default=None,
                        help="Fix random generator seed. Setting this forces a deterministic run")
    parser.add_argument("--profile", action="store_true",
                        help="profile the code and write to file 'restats'")
    parser.add_argument("--ss-height", type=int,
                        help="resize images to this height before applying Selective Search")

    # HOG parameters
    parser.add_argument("--hog-bins", default=9, type=int,
                        help="number or bins for orientation binning in HOG")
    parser.add_argument("--ppc", default=(8, 8), nargs="+", type=int,
                        help="pixels per cell in HOG")
    parser.add_argument("--cpb", default=(2, 2), nargs="+", type=int,
                        help="cells per block in HOG")
    parser.add_argument("--block-norm", default="L2-Hys", type=str,
                        help="block norm in HOG")
    parser.add_argument("--gamma-corr", action="store_true",
                        help="use gamma correction in HOG")
    parser.add_argument("--no-gamma-corr", action="store_false", dest="gamma-corr",
                        help="don't use gamma correction in HOG")
    parser.set_defaults(gamma_corr=True)
    parser.add_argument("--hog-dim", default=(128, 128), nargs="+", type=int,
                        help="input dimensions for HOG extractor")

    # Sliding window parameters
    parser.add_argument("--downscale-factor", default=1.25, type=float,
                        help="downscale factor in each iteration of gaussian pyramid")
    parser.add_argument("--min-window", default=(50, 50), type=int, nargs="+",
                        help="minimum pixel size of sliding window")
    parser.add_argument("--step-size", default=(20, 20), type=int, nargs="+",
                        help="step size in pixels of sliding window")

    # Classifier parameters
    parser.add_argument("--pca-components", default=600, type=int)
    parser.add_argument("--rbf-components", default=2000, type=int)
    parser.add_argument("--rbf-gamma", default=1e-4, type=float)
    parser.add_argument("--C", default=100, type=int)
    parser.add_argument("--dual", action="store_true")
    parser.add_argument("--max-iter", default=10000, type=int)


    # Evaluation parameters
    parser.add_argument("--evaluate-only", action="store_true",
                        help="only evaluate model")
    parser.add_argument("--output-dir", default=".", type=str,
                        help="path to save outputs")
    parser.add_argument("--plot-pc", action="store_true",
                        help="plot precision recall curves")

    # Hard negative mining parameters
    parser.add_argument("--neg-per-img", default=50, type=int,
                        help="how many hard negatives to mine from each image")

    # Multicore parameters
    parser.add_argument("--n-cpus", default=1, type=int,
                        help="number of cores to use for parallel operations")
    return parser


def main(args):
    utils.initialise_logging(args)
    logging.info("Started")

    if args.seed is not None:
        utils.make_deterministic(args.seed)
        logging.info(f"Seed set to {args.seed}")

    cv2.setUseOptimized(True)
    cv2.setNumThreads(1)

    if not os.path.exists(args.output_dir):
        logging.info("Creating output directory...")
        os.makedirs(args.output_dir)

    # Parse class definition file
    logging.info("Loading class definitions...")
    with open(args.class_file, "r") as f:
        classes = json.load(f)
    num_classes = len(classes) + 1
    logging.info(f"Training with {num_classes} classes: " + ", ".join(['background'] + list(classes.keys())))

    # Load datasets
    logging.info("Loading dataset...")
    if not args.evaluate_only:
        train_dataset, val_dataset = datasets.load_train_val(train_path=args.train_path, classes=classes,
                                                             val_split=args.val_split, train_transforms=None,
                                                             val_transforms=None)
    if args.test_path:
        test_dataset = datasets.load_test(test_path=args.test_path, classes=classes, transforms=None)

    # Create feature extractor
    logging.info("Creating feature extractor...")
    feature_extractor = models.HOG(orientations=args.hog_bins, pixels_per_cell=args.ppc, cells_per_block=args.cpb,
                                   block_norm=args.block_norm, gamma_corr=args.gamma_corr, resize_to=args.hog_dim)

    if not args.evaluate_only:
        cache_file = os.path.join(args.output_dir, 'saved_descriptors.cached.pickle')
        if os.path.isfile(cache_file):
            with open(cache_file, 'rb') as f:
                descriptors, labels = pickle.load(f)
            logging.info("Loading existing cached training data...")
        else:
            # Extract features
            logging.info("Extracting features...")
            descriptors, labels = feature_extractor.extract_all(train_dataset, cpus=args.n_cpus,
                                                                horizontal_flip=True,
                                                                rotations=[30, -30])
            clf = DummyClassifier(strategy='constant', constant=1)
            clf.fit(descriptors[:2], [0, 1])

            # Apply negative mining
            logging.info("Performing hard negative mining...")
            negative_samples = trainingtools.mine_hard_negatives(clf, feature_extractor, train_dataset,
                                                                 iou_thresh=args.iou_thresh, max_per_img=args.neg_per_img,
                                                                 cpus=args.n_cpus)
            descriptors = np.concatenate((descriptors, negative_samples))
            labels = np.concatenate((labels, np.zeros(negative_samples.shape[0], dtype=np.int64)))
            logging.info(
                f"Added {len(negative_samples)} negatives to the {len(descriptors) - len(negative_samples)} positives")
            with open(cache_file, 'wb') as f:
                pickle.dump((descriptors, labels), f)

        logging.info(f"Training on {len(descriptors)} samples with {len(descriptors[0])} dimensions...")

        clf = Pipeline(steps=[('scaler', StandardScaler()),
                              ('pca', PCA(n_components=args.pca_components)),
                              ('feature_map', Nystroem(n_components=args.rbf_components, gamma=args.rbf_gamma)),
                              ('model', LinearSVC(C=args.C, dual=args.dual, max_iter=args.max_iter,
                                                  class_weight='balanced'))])
        start = time()
        clf.fit(descriptors, labels)
        fit_time = time() - start
        logging.info(f"Fit took {fit_time:.1f} seconds")
        logging.info(f"Saving trained model to output directory {args.output_dir}")
        joblib.dump(clf, os.path.join(args.output_dir, "saved_classifier.pickle"))
        logging.info("Saved")
    else:
        clf = joblib.load(args.load_model)

    # Evaluate
    use_test = args.test_path is not None
    logging.info(f"Evaluating classifier on {'test' if use_test else 'validation'} dataset")
    trainingtools.evaluate_classifier(clf, feature_extractor=feature_extractor,
                                      dataset=test_dataset if use_test else val_dataset,
                                      iou_thresh=args.iou_thresh,
                                      ss_height=args.ss_height,
                                      cpus=args.n_cpus,
                                      save_to_file=os.path.join(args.output_dir, "evaluator.pickle"))


if __name__ == '__main__':
    parsed_args = get_args_parser().parse_args()
    if parsed_args.profile:
        cProfile.run('main(parsed_args)', 'restats')
    else:
        main(parsed_args)
