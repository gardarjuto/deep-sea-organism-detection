import argparse
import json
import numpy as np
import torch
import optuna
from torch.utils.data import DataLoader
from detection import models, utils, datasets
from detection import trainingtools
import logging

MILESTONES = [16, 22]
GAMMA = 0.1


def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description="Hyperparameter Optimisation for FasterRCNN Pipeline",
                                     add_help=add_help)
    parser.add_argument("--data-path", default="data", type=str, help="dataset path")
    parser.add_argument("--class-file", default="classes.json", type=str, help="path to class definitions")
    parser.add_argument("--db-login", type=str, help="path to database login file")
    parser.add_argument("--model", default="rcnn_resnet50_fpn", type=str, help="model name")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument(
        "--train-ratio", "--tr", default=0.8, type=float, help="proportion of dataset to use for training"
    )
    parser.add_argument("--epochs", default=5, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument("--output-dir", type=str, help="directory for storing program output")
    parser.add_argument(
        "-j", "--workers", default=4, type=int, metavar="N", help="number of data loading workers (default: 4)"
    )
    parser.add_argument("--iou-thresh", default=0.5, type=float, help="IoU threshold for evaluation")
    parser.add_argument("--log-file", "--lf", default=None, type=str, help="path to file for writing logs. If "
                                                                           "omitted, writes to stdout")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"))
    parser.add_argument("--log-every", default=10, type=int, help="log every ith batch")

    parser.add_argument(
        "--seed", type=int, default=None, help="fix random generator seed. Setting this forces a deterministic run"
    )
    parser.add_argument("--n-trials", type=int, help="number of optimization trials")

    return parser


class Objective(object):
    def __init__(self, train_dataset, val_dataset, classes, workers, device, model, epochs, iou_thresh):
        # Hold this implementation specific arguments as the fields of the class.
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.classes = classes
        self.workers = workers
        self.device = device
        self.model = model
        self.epochs = epochs
        self.iou_thresh = iou_thresh

    def __call__(self, trial):
        logging.info("Started")
        device = torch.device(self.device)
        logging.info(f"Using device: {device}")

        logging.info("Creating data loaders...")
        train_sampler = torch.utils.data.RandomSampler(self.train_dataset)
        val_sampler = torch.utils.data.SequentialSampler(self.val_dataset)

        batch_size = trial.suggest_int("batch size", 1, 8)
        train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, batch_size, drop_last=True)

        train_loader = DataLoader(self.train_dataset, batch_sampler=train_batch_sampler, num_workers=self.workers,
                                  collate_fn=utils.collate_fn)
        val_loader = DataLoader(self.val_dataset, batch_size=1, sampler=val_sampler, num_workers=self.workers,
                                collate_fn=utils.collate_fn)

        # Load model
        logging.info("Loading model...")
        model = models.load_model(self.model, num_classes=len(self.classes)+1, pretrained=True)
        model = model.to(device)

        # Observe that all parameters are being optimized
        params = [p for p in model.parameters() if p.requires_grad]
        lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
        optim = trial.suggest_categorical("optimizer", ["SGD", "Adam"])
        if optim == "SGD":
            momentum = trial.suggest_uniform("momentum", 0.0, 1.0)
            optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum)
        else:
            optimizer = torch.optim.Adam(params, lr)

        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONES, gamma=GAMMA)

        best_mAP = 0.0

        # Train the model
        logging.info("Beginning training:")
        for epoch in range(self.epochs):
            # Train one epoch
            status = trainingtools.train_one_epoch(model, train_loader, device=device, optimizer=optimizer,
                                                   epoch=epoch, n_epochs=self.epochs)
            if status == -1:
                raise optuna.exceptions.TrialPruned()
            # Update the learning rate
            lr_scheduler.step()

            # Evaluate on the test data
            res = trainingtools.evaluate(model, loader=val_loader, device=device, epoch=epoch,
                                         iou_thresh=self.iou_thresh, plot_pc=False)
            mAP = res["mAP"] if np.isfinite(res["mAP"]) else 0.0
            best_mAP = max(best_mAP, mAP)
            trial.report(best_mAP, epoch)

            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        return best_mAP


def main(args):
    utils.initialise_logging(args)
    utils.make_deterministic(args.seed)

    with open(args.class_file, "r") as f:
        classes = json.load(f)

    # Load data
    logging.info("Loading dataset...")
    train_dataset, val_dataset = datasets.load_train_val(name="FathomNet", train_path=args.data_path, classes=classes,
                                                         val_split=1-args.train_ratio)

    objective = Objective(train_dataset, val_dataset, classes, args.workers, args.device, args.model, args.epochs,
                          args.iou_thresh)

    with open(args.db_login, "r") as f:
        login = json.load(f)

    storage = optuna.storages.RDBStorage(
        "postgresql://" + login["username"] + ":" + login["password"] + "@" + login["host"] + "/postgres",
        heartbeat_interval=1
    )
    study = optuna.create_study(
        storage=storage, study_name=args.study_name, direction="maximize", load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=args.seed, multivariate=True, group=True, constant_liar=True),
        pruner=optuna.pruners.HyperbandPruner(min_resource=1, max_resource=args.epochs, reduction_factor=3)
    )
    study.optimize(objective, n_trials=args.n_trials)

    logging.info("Done")


if __name__ == '__main__':
    # Parse arguments
    parsed_args = get_args_parser().parse_args()
    main(parsed_args)
