import argparse
import cProfile
import json
import logging
from time import time

import cv2
import numpy as np
from optuna._callbacks import RetryFailedTrialCallback
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from detection import models, utils, datasets, evaluation
from detection import trainingtools
import optuna

MAX_RETRY = 2

def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description="Hyperparameter Optimisation for Classical Pipeline", add_help=add_help)
    parser.add_argument("--train-path", type=str, help="path to training dataset")
    parser.add_argument("--test-path", type=str, help="path to test dataset")
    parser.add_argument("--class-file", default="classes.json", type=str, help="path to class definitions")
    parser.add_argument("--db-login", type=str, help="path to database login file")
    parser.add_argument("--val-split", "--tr", default=0.2, type=float,
                        help="proportion of training dataset to use for validation")
    parser.add_argument("--iou-thresh", default=0.5, type=float, help="IoU threshold for evaluation")
    parser.add_argument("--log-file", "--lf", default=None, type=str,
                        help="path to file for writing logs. If omitted, writes to stdout")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"))

    # Hard negative mining parameters
    parser.add_argument("--neg-per-img", default=50, type=int, help="how many hard negatives to mine from each image")

    parser.add_argument("--profile", action="store_true", help="run with profile statistics")

    parser.add_argument("--n-cpus", type=int, help="number of cpus to use per trial")
    parser.add_argument("--n-trials", type=int, help="number of optimization trials")
    parser.add_argument("--study-name", type=str, help="name of optimization study")
    return parser


class Objective(object):
    def __init__(self, train_dataset, val_dataset, neg_per_img, iou_thresh, n_cpus):
        # Hold this implementation specific arguments as the fields of the class.
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.neg_per_img = neg_per_img
        self.iou_thresh = iou_thresh
        self.n_cpus = n_cpus

    def __call__(self, trial):
        # Create feature extractor
        block_norm = trial.suggest_categorical('block norm', ['L1', 'L1-sqrt', 'L2', 'L2-Hys'])
        hog_inp_dim = trial.suggest_int('HOG input size', 32, 128, log=True)
        feature_extractor = models.HOG(orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3),
                                       block_norm=block_norm, gamma_corr=True, resize_to=(hog_inp_dim, hog_inp_dim))

        # Extract features
        descriptors, labels = feature_extractor.extract_all(self.train_dataset, cpus=self.n_cpus)

        # Add one background GT for SVM
        descriptors.append(np.zeros_like(descriptors[0]))
        labels.append(0)
        print("Descriptors:", len(descriptors), descriptors[0].shape)

        # Train SVM
        pca_components = min(trial.suggest_int('PCA components', 10, 500, log=True),
                             descriptors[0].shape[0],
                             len(descriptors))
        class_weight = trial.suggest_categorical('class_weight', [None, 'balanced'])
        gamma = trial.suggest_float('RBF gamma', 1e-10, 1)
        n_components = trial.suggest_int('Nystroem components', 10, 800, log=True)
        alpha = trial.suggest_loguniform('SGD alpha', 1e-10, 1)
        clf = Pipeline(steps=[('scaler', StandardScaler()),
                              ('pca', PCA(n_components=pca_components)),
                              ('feature_map', Nystroem(gamma=gamma, n_components=n_components)),
                              ('model', SGDClassifier(alpha=alpha, class_weight=class_weight,
                                                      fit_intercept=False, max_iter=100000))])
        start = time()
        clf.fit(descriptors, labels)
        fit_time = time() - start
        logging.info(f"Fit took {fit_time:.1f} seconds")
        trial.set_user_attr("fit_time", fit_time)

        if self.neg_per_img > 0:
            negative_samples = trainingtools.mine_hard_negatives(clf, feature_extractor, self.train_dataset,
                                                                 max_per_img=self.neg_per_img, cpus=self.n_cpus)

            # Add hard negatives to training samples
            descriptors.extend(negative_samples)
            labels.extend([0] * len(negative_samples))
            logging.info(
                f"Added {len(negative_samples)} negative samples to the previous {len(descriptors) - len(negative_samples)} total")

            # Train SVM
            logging.info("Training classifier on feature descriptors")
            clf.fit(descriptors, labels)

        # Evaluate
        result = trainingtools.evaluate_classifier(clf, feature_extractor=feature_extractor, dataset=self.val_dataset,
                                                   plot_pc=False, cpus=self.n_cpus)
        logging.info("Evaluation done")

        return -result['mAP']


def main(args):
    utils.initialise_logging(args)
    utils.make_deterministic(42)

    cv2.setUseOptimized(True)
    cv2.setNumThreads(1)

    with open(args.class_file, "r") as f:
        classes = json.load(f)

    # Load datasets
    logging.info("Loading dataset...")
    train_dataset, val_dataset = datasets.load_train_val(name='FathomNet', train_path=args.train_path, classes=classes,
                                                         val_split=args.val_split)

    objective = Objective(train_dataset, val_dataset, args.neg_per_img, args.iou_thresh, args.n_cpus)

    with open(args.db_login, "r") as f:
        login = json.load(f)

    storage = optuna.storages.RDBStorage(
        "postgresql://" + login["username"] + ":" + login["password"] + "@" + login["host"] + "/postgres",
        heartbeat_interval=1,
        failed_trial_callback=RetryFailedTrialCallback(max_retry=MAX_RETRY),
    )
    study = optuna.create_study(
        storage=storage, study_name=args.study_name, direction="maximize", load_if_exists=True,
        sampler=optuna.samplers.TPESampler(multivariate=True, group=True, constant_liar=True),
    )
    study.optimize(objective, n_trials=args.n_trials)


if __name__ == '__main__':
    # Parse arguments
    parsed_args = get_args_parser().parse_args()
    if parsed_args.profile:
        cProfile.run('main(parsed_args)', 'restats')
    else:
        main(parsed_args)
