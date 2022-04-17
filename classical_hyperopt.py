import argparse
import cProfile
import json
import logging
import os
import pickle
from time import time

import cv2
import numpy as np
import sklearn
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
    parser.add_argument("--neg-per-img", default=None, type=int, help="how many hard negatives to mine from each image")

    parser.add_argument("--profile", action="store_true", help="run with profile statistics")

    parser.add_argument("--n-cpus", type=int, help="number of cpus to use per trial")
    parser.add_argument("--n-trials", type=int, help="number of optimization trials")
    parser.add_argument("--study-name", type=str, help="name of optimization study")

    parser.add_argument("--cached-path", type=str, help="path to cached descriptors and labels")
    return parser


class Objective(object):
    def __init__(self, val_dataset, n_cpus, feature_extractor, descriptors, labels):
        # Hold this implementation specific arguments as the fields of the class.
        self.val_dataset = val_dataset
        self.n_cpus = n_cpus
        self.feature_extractor = feature_extractor
        self.descriptors = descriptors
        self.labels = labels

    def __call__(self, trial):
        # Train SVM
        pca_components = 200
        nystroem_components = 800
        class_weight = trial.suggest_categorical('class_weight', ['None', 'balanced'])
        gamma = trial.suggest_float('rbf_gamma', 1e-8, 1e-3, log=True)
        alpha = trial.suggest_float('sgd_alpha', 1e-12, 1e-2, log=True)
        clf = Pipeline(steps=[('scaler', StandardScaler()),
                              ('pca', PCA(n_components=pca_components)),
                              ('feature_map', Nystroem(gamma=gamma, n_components=nystroem_components)),
                              ('model', SGDClassifier(alpha=alpha, max_iter=100000, early_stopping=True))])
        start = time()
        clf.fit(self.descriptors, self.labels)
        fit_time = time() - start
        logging.info(f"Fit took {fit_time:.1f} seconds")
        trial.set_user_attr("fit_time", fit_time)

        # Evaluate
        result = trainingtools.evaluate_classifier(clf, feature_extractor=self.feature_extractor,
                                                   dataset=self.val_dataset, plot_pc=False, cpus=self.n_cpus)

        return result['mAP'] if not np.isnan(result['mAP']) else 0.0


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

    # Create feature extractor
    feature_extractor = models.HOG(orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                                   block_norm='L2-Hys', gamma_corr=True, resize_to=(128, 128))

    cache_file = os.path.join(args.cached_path, 'classical_hyperopt.cached.pickle')
    if os.path.isfile(cache_file):
        with open(cache_file, 'rb') as f:
            descriptors, labels = pickle.load(f)
        logging.info("Loading existing cached training data")
    else:
        # Extract features
        descriptors, labels = feature_extractor.extract_all(train_dataset, cpus=args.n_cpus)

        clf = sklearn.dummy.DummyClassifier(strategy='constant', constant=1)
        clf.fit(descriptors[:2], [0, 1])

        # Apply negative mining
        logging.info("Performing hard negative mining")
        negative_samples = trainingtools.mine_hard_negatives(clf, feature_extractor, train_dataset,
                                                             iou_thresh=args.iou_thresh, max_per_img=args.neg_per_img,
                                                             cpus=args.n_cpus)
        descriptors += negative_samples
        labels += [0 for _ in negative_samples]
        logging.info(
            f"Added {len(negative_samples)} negatives to the {len(descriptors) - len(negative_samples)} positives")
        with open(cache_file, 'wb') as f:
            pickle.dump((descriptors, labels), f)

    logging.info(f"Training on {len(descriptors)} samples with {len(descriptors[0])} dimensions")

    objective = Objective(val_dataset, args.n_cpus, feature_extractor, descriptors, labels)

    with open(args.db_login, "r") as f:
        login = json.load(f)

    storage = optuna.storages.RDBStorage(
        "postgresql://" + login["username"] + ":" + login["password"] + "@" + login["host"] + "/postgres",
        heartbeat_interval=1,
        failed_trial_callback=RetryFailedTrialCallback(max_retry=MAX_RETRY),
    )
    search_space = {
        "rbf_gamma": np.logspace(-8, -3, 6),
        "sgd_alpha": np.logspace(-12, -2, 11)
    }
    study = optuna.create_study(
        storage=storage, study_name=args.study_name, direction="maximize", load_if_exists=True,
        sampler=optuna.samplers.GridSampler(search_space)
    )
    study.optimize(objective)


if __name__ == '__main__':
    # Parse arguments
    parsed_args = get_args_parser().parse_args()
    if parsed_args.profile:
        cProfile.run('main(parsed_args)', 'restats')
    else:
        main(parsed_args)
