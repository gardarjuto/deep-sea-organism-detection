import argparse
import json
import logging
import os

import cv2
import numpy as np
import plotly
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from detection import models, utils, datasets, evaluation
from detection import trainingtools
import optuna
from optuna.visualization import plot_optimization_history, plot_contour, plot_parallel_coordinate


def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description="Hyperparameter Optimisation for Classical Pipeline", add_help=add_help)
    parser.add_argument("--train-path", type=str, help="path to training dataset")
    parser.add_argument("--test-path", type=str, help="path to test dataset")
    parser.add_argument("--class-file", default="classes.json", type=str, help="path to class definitions")
    parser.add_argument("--val-split", "--tr", default=0.2, type=float,
                        help="proportion of training dataset to use for validation")
    parser.add_argument("--iou-thresh", default=0.5, type=float, help="IoU threshold for evaluation")
    parser.add_argument("--log-file", "--lf", default=None, type=str,
                        help="path to file for writing logs. If omitted, writes to stdout")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"))
    parser.add_argument("--output-dir", default=".", type=str, help="path to save outputs")

    # Hard negative mining parameters
    parser.add_argument("--neg-per-img", default=50, type=int, help="how many hard negatives to mine from each image")

    parser.add_argument("--n-jobs", default=1, type=int, help="number of optimization jobs to run in parallel")
    parser.add_argument("--n-trials", default=5, type=int, help="number of optimization iterations")
    return parser


class Objective(object):
    def __init__(self, train_dataset, val_dataset, neg_per_img, n_jobs):
        # Hold this implementation specific arguments as the fields of the class.
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.neg_per_img = neg_per_img
        self.n_jobs = n_jobs

    def __call__(self, trial):
        # Create feature extractor
        orientations = 9
        ppc = trial.suggest_int('pixels per cell', 8, 16)
        cpb = trial.suggest_int('cells per block', 2, 3)
        block_norm = trial.suggest_categorical('block norm', ['L1', 'L1-sqrt', 'L2', 'L2-Hys'])
        gamma_corr = trial.suggest_categorical('gamma correction', [True, False])
        hog_inp_dim = 64  # trial.suggest_int('HOG input size', 64, 128, log=True)
        feature_extractor = models.HOG(orientations=orientations, pixels_per_cell=(ppc, ppc), cells_per_block=(cpb, cpb),
                                       block_norm=block_norm, gamma_corr=gamma_corr, resize_to=(hog_inp_dim, hog_inp_dim))

        # Extract features
        descriptors, labels = feature_extractor.extract_all(train_dataset, cpus=self.n_jobs)

        # Add one background GT for SVM
        descriptors.append(np.zeros_like(descriptors[0]))
        labels.append(0)
        print("Descriptors:", len(descriptors), descriptors[0].shape)
        print(orientations, ppc, cpb, hog_inp_dim)

        # Train SVM
        classifier = trial.suggest_categorical('classifier', ['SGD+Nystroem'])
        pca_components = min(trial.suggest_int('PCA components', 10, 30, log=True),
                             descriptors[0].shape[0],
                             len(descriptors))
        class_weight = trial.suggest_categorical('class weight', [None, 'balanced'])
        if classifier == 'SGD+Nystroem':
            gamma = trial.suggest_float('RBF gamma', 1e-10, 1)
            n_components = trial.suggest_int('Nystroem components', 10, 80, log=True)
            alpha = trial.suggest_loguniform('SGD alpha', 1e-10, 1)
            max_iter = trial.suggest_int('max iter', 100, 100000, log=True)
            clf = Pipeline(steps=[('scaler', StandardScaler()),
                                  ('pca', PCA(n_components=pca_components)),
                                  ('feature_map', Nystroem(gamma=gamma, n_components=n_components)),
                                  ('model', SGDClassifier(alpha=alpha, class_weight=class_weight,
                                                          fit_intercept=False, max_iter=max_iter))])
        elif classifier == 'RandomForest':
            n_estimators = trial.suggest_int('RF estimators', 10, 500, log=True)
            criterion = trial.suggest_categorical('RF criterion', ['gini', 'entropy'])
            max_features = trial.suggest_categorical('RF max features', [None, 'sqrt', 'log2'])
            clf = Pipeline(steps=[('scaler', StandardScaler()),
                                  ('pca', PCA(n_components=pca_components)),
                                  ('model', RandomForestClassifier(n_estimators=n_estimators, class_weight=class_weight,
                                                                   criterion=criterion, max_features=max_features))])

        clf.fit(descriptors, labels)

        perform_hard_negative_mining = trial.suggest_categorical('hard negative mining', [True, False])

        if perform_hard_negative_mining:
            negative_samples = trainingtools.mine_hard_negatives(clf, feature_extractor, train_dataset,
                                                                 max_per_img=self.neg_per_img, cpus=self.n_jobs)

            # Add hard negatives to training samples
            descriptors.extend(negative_samples)
            labels.extend([0] * len(negative_samples))
            logging.info(
                f"Added {len(negative_samples)} negative samples to the previous {len(descriptors) - len(negative_samples)} total")

            # Train SVM
            logging.info("Training classifier on feature descriptors")
            clf.fit(descriptors, labels)

        # Evaluate
        result = trainingtools.evaluate_classifier(clf, feature_extractor=feature_extractor, dataset=val_dataset,
                                                   plot_pc=False, cpus=self.n_jobs)

        return -result['mAP']


if __name__ == '__main__':
    # Parse arguments
    args = get_args_parser().parse_args()
    utils.initialise_logging(args)
    utils.make_deterministic(42)

    with open('classes', "r") as f:
        classes = json.load(f)
    num_classes = len(classes) + 1

    # Load datasets
    logging.info("Loading dataset...")
    train_dataset, val_dataset = datasets.load_train_val(name='FathomNet', train_path=args.train_path, classes=classes,
                                                         val_split=args.val_split)

    cv2.setUseOptimized(True)
    cv2.setNumThreads(1)

    objective = Objective(train_dataset, val_dataset, args.neg_per_img, args.n_jobs)
    study = optuna.create_study()
    study.optimize(objective, n_trials=args.n_trials)
    best_params = study.best_params
    print(best_params)
    fig = plot_optimization_history(study)
    plotly.offline.plot(fig, filename=os.path.join(args.output_dir, 'optimization_history.html'))
    fig = plot_contour(study)
    plotly.offline.plot(fig, filename=os.path.join(args.output_dir, 'contour.html'))
    fig = plot_parallel_coordinate(study)
    plotly.offline.plot(fig, filename=os.path.join(args.output_dir, 'parallel_coordinate.html'))
