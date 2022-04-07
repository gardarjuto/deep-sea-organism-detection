import argparse
import cProfile
import json
import cv2

import sklearn.dummy

from detection import models, utils, datasets
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

    # Evaluation parameters
    parser.add_argument("--output-dir", default=".", type=str, help="path to save outputs")
    parser.add_argument("--plot-pc", action="store_true", help="plot precision recall curves")

    # Classifier parameters
    parser.add_argument("--C", default=1.0, type=float, help="C parameter for SVM")
    parser.add_argument("--loss", default="hinge", type=str, help="loss type for SVM")
    parser.add_argument("--classifier-name", type=str, help="Name of classifier to use")
    parser.add_argument("--pca-components", default=100, type=str, help="Number of PCA components")
    parser.add_argument("--nystroem-components", default=500, type=int, help="Number of Nystroem components")
    parser.add_argument("--nystroem-gamma", type=float, help="Nystroem gamma value")
    parser.add_argument("--sgd-alpha", type=float, help="SGD alpha value")
    parser.add_argument("--class-weight", type=str, help="Class weight to use ('balanced' or None)")
    parser.add_argument("--fit-intercept", action="store_true", help="Fit intercept")
    parser.add_argument("--max-iter", type=int, help="Maximum iterations in SGD")

    # Hard negative mining parameters
    parser.add_argument("--neg-per-img", default=None, type=int, help="how many hard negatives to mine from each image")

    # Multicore parameters
    parser.add_argument("--cpus", default=1, type=int, help="number of cores to use for parallel operations")

    # Selective search
    parser.add_argument("--ss-height", default=250, type=int,
                        help="Resize height of image before applying selective search")
    return parser


def main(args):
    utils.initialise_logging(args)
    print(args)
    logging.info("Started")

    if args.seed is not None:
        utils.make_deterministic(args.seed)
        logging.info(f"Seed set to {args.seed}")

    # Parse class definition file
    logging.info("Loading class definitions...")
    with open(args.class_file, "r") as f:
        classes = json.load(f)
    num_classes = len(classes)
    logging.info(f"Training with {num_classes} classes: " + ", ".join(list(classes.keys())))

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
    logging.info(f"N={len(descriptors)},D={len(descriptors[0])}")

    # Create first stage as dummy at first, classifying everything as object
    bg_clf = sklearn.dummy.DummyClassifier(strategy='constant', constant=1)
    bg_clf.fit(descriptors[:2], [0, 1])

    # Train second stage
    logging.info("Training object classifier on feature descriptors")
    obj_clf = trainingtools.train_classifier(args.classifier_name, descriptors, labels, num_classes,
                                             kwargs={
                                                 'pca_components': args.pca_components,
                                                 'nystroem_gamma': args.nystroem_gamma,
                                                 'nystroem_components': args.nystroem_components,
                                                 'sgd_alpha': args.sgd_alpha,
                                                 'class_weight': args.class_weight,
                                                 'fit_intercept': args.fit_intercept,
                                                 'max_iter': args.max_iter}
                                             )

    cv2.setUseOptimized(True)
    cv2.setNumThreads(1)

    # Evaluate
    logging.info("Evaluating on test dataset with dummy background classifier")
    trainingtools.evaluate_two_stage(bg_clf, obj_clf, feature_extractor=feature_extractor, dataset=val_dataset,
                                     iou_thresh=args.iou_thresh, ss_height=args.ss_height, output_dir=args.output_dir,
                                     plot_pc=args.plot_pc, cpus=args.cpus)

    # Apply hard negative mining
    logging.info("Performing hard negative mining")
    negative_samples = trainingtools.mine_hard_negatives(bg_clf, feature_extractor, train_dataset,
                                                         iou_thresh=args.iou_thresh, max_per_img=args.neg_per_img,
                                                         cpus=args.cpus)

    # Create background training data
    background_descriptors = descriptors + negative_samples
    background_labels = [1 for _ in range(len(descriptors))] + [0 for _ in range(len(negative_samples))]

    logging.info(f"Added {len(negative_samples)} negative samples to the {len(descriptors)} positive ones")

    # Train background classifier
    logging.info("Training background classifier")
    bg_clf = trainingtools.train_classifier(args.classifier_name, background_descriptors, background_labels, 2,
                                            kwargs={
                                                'pca_components': args.pca_components,
                                                'nystroem_gamma': args.nystroem_gamma,
                                                'nystroem_components': args.nystroem_components,
                                                'sgd_alpha': args.sgd_alpha,
                                                'class_weight': args.class_weight,
                                                'fit_intercept': args.fit_intercept,
                                                'max_iter': args.max_iter}
                                            )

    # Evaluate
    logging.info("Evaluating completed two-stage classifier on test dataset")
    trainingtools.evaluate_two_stage(bg_clf, obj_clf, feature_extractor=feature_extractor, dataset=val_dataset,
                                     iou_thresh=args.iou_thresh, ss_height=args.ss_height, output_dir=args.output_dir,
                                     plot_pc=args.plot_pc, cpus=args.cpus)


if __name__ == '__main__':
    parsed_args = get_args_parser().parse_args()
    if parsed_args.profile:
        cProfile.run('main(parsed_args)', 'restats')
    else:
        main(parsed_args)
