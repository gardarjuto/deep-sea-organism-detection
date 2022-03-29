import argparse
import json
import os
import shutil

import plotly
import torch
import optuna
from optuna.storages import RetryFailedTrialCallback
from optuna.visualization import plot_intermediate_values, plot_contour, plot_parallel_coordinate, \
    plot_optimization_history
from torch import nn
from torch.utils.data import DataLoader
from detection import models, utils, datasets
from detection import trainingtools
import logging

MILESTONES = [16, 22]
GAMMA = 0.1

def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description="Hyperparameter Optimisation for FasterRCNN Pipeline", add_help=add_help)

    parser.add_argument("--data-path", default="data", type=str, help="dataset path")
    parser.add_argument("--class-file", default="classes.json", type=str, help="path to class definitions")
    parser.add_argument("--model", default="rcnn_resnet50_fpn", type=str, help="model name")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument(
        "--train-ratio", "--tr", default=0.8, type=float, help="proportion of dataset to use for training"
    )
    parser.add_argument("--epochs", default=5, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument("--checkpoint-dir", type=str, help="directory for storing checkpoint")
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
    def __init__(self, train_dataset, val_dataset, classes, workers, device, model, epochs, checkpoint_dir, iou_thresh):
        # Hold this implementation specific arguments as the fields of the class.
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.classes = classes
        self.workers = workers
        self.device = device
        self.model = model
        self.epochs = epochs
        self.checkpoint_dir = checkpoint_dir
        self.iou_thresh = iou_thresh

    def __call__(self, trial):
        logging.info("Started")

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
        device = torch.device(self.device)
        logging.info(f"Using device: {device}")
        logging.info("Loading model...")
        model = models.load_model(self.model, num_classes=num_classes, pretrained=True)
        model = model.to(device)

        # Observe that all parameters are being optimized
        params = [p for p in model.parameters() if p.requires_grad]
        lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
        momentum = trial.suggest_uniform("momentum", 0.0, 1.0)
        optim = trial.suggest_categorical("optimizer", ["SGD", "Adam"])
        if optim == "SGD":
            optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum)
        else:
            optimizer = torch.optim.Adam(params, lr)

        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONES, gamma=GAMMA)

        trial_number = RetryFailedTrialCallback.retried_trial_number(trial)
        trial_checkpoint_dir = os.path.join(self.checkpoint_dir, str(trial_number))
        checkpoint_path = os.path.join(trial_checkpoint_dir, "model.pt")
        checkpoint_exists = os.path.isfile(checkpoint_path)

        if trial_number is not None and checkpoint_exists:
            checkpoint = torch.load(checkpoint_path)
            epoch = checkpoint["epoch"]
            start_epoch = epoch + 1
            logging.info(f"Loading a checkpoint from trial {trial_number} in epoch {epoch}.")
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            mAP = checkpoint["mAP"]
        else:
            trial_checkpoint_dir = os.path.join(self.checkpoint_dir, str(trial.number))
            checkpoint_path = os.path.join(trial_checkpoint_dir, "model.pt")
            start_epoch = 0
            mAP = 0

        os.makedirs(trial_checkpoint_dir, exist_ok=True)
        # A checkpoint may be corrupted when the process is killed during `torch.save`.
        # Reduce the risk by first calling `torch.save` to a temporary file, then copy.
        tmp_checkpoint_path = os.path.join(trial_checkpoint_dir, "tmp_model.pt")

        logging.info(f"Checkpoint path for trial is '{checkpoint_path}'.")

        best_mAP = mAP

        # Train the model
        logging.info("Beginning training:")
        for epoch in range(start_epoch, self.epochs):

            # Train one epoch
            trainingtools.train_one_epoch(model, train_loader, device=device, optimizer=optimizer,
                                          epoch=epoch, n_epochs=self.epochs)

            # Update the learning rate
            lr_scheduler.step()

            # Evaluate on the test data
            res = trainingtools.evaluate(model, loader=val_loader, device=device, epoch=epoch,
                                         iou_thresh=self.iou_thresh, plot_pc=False)

            trial.report(res["mAP"], epoch)
            best_mAP = max(best_mAP, res["mAP"])

            logging.info(f"Saving a checkpoint in epoch {epoch}.")
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
                "mAP": res["mAP"]
            }
            torch.save(checkpoint, tmp_checkpoint_path)
            shutil.move(tmp_checkpoint_path, checkpoint_path)

            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        return best_mAP


if __name__ == '__main__':
    # Parse arguments
    args = get_args_parser().parse_args()
    utils.initialise_logging(args)
    utils.make_deterministic(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.class_file, "r") as f:
        classes = json.load(f)
    num_classes = len(classes) + 1

    # Load data
    logging.info("Loading dataset...")
    train_dataset, val_dataset = datasets.load_train_val(name="FathomNet", train_path=args.data_path, classes=classes,
                                                         val_split=1-args.train_ratio)

    objective = Objective(train_dataset, val_dataset, classes, args.workers, args.device, args.model, args.epochs,
                          args.checkpoint_dir, args.iou_thresh)
    storage = optuna.storages.RDBStorage(
        "sqlite:///" + os.path.join(args.checkpoint_dir, "neural_hyperopt.db"),
        heartbeat_interval=1,
        failed_trial_callback=RetryFailedTrialCallback(),
    )
    study = optuna.create_study(
        storage=storage, study_name="pytorch_checkpoint", direction="maximize", load_if_exists=True
    )
    study.optimize(objective, n_trials=args.n_trials, timeout=600)

    pruned_trials = study.get_trials(states=(optuna.trial.TrialState.PRUNED,))
    complete_trials = study.get_trials(states=(optuna.trial.TrialState.COMPLETE,))

    logging.info("Study statistics: ")
    logging.info(f"  Number of finished trials: {len(study.trials)}")
    logging.info(f"  Number of pruned trials: {len(pruned_trials)}")
    logging.info(f"  Number of complete trials: {len(complete_trials)}")

    logging.info("Best trial:")
    trial = study.best_trial

    logging.info(f"  Value: {trial.value}")

    logging.info("  Params: ")
    for key, value in trial.params.items():
        logging.info(f"    {key}: {value}")

    fig = plot_intermediate_values(study)
    plotly.offline.plot(fig, filename=os.path.join(args.output_dir, 'intermediate_values.html'))
    fig = plot_optimization_history(study)
    plotly.offline.plot(fig, filename=os.path.join(args.output_dir, 'optimization_history.html'))
    fig = plot_contour(study)
    plotly.offline.plot(fig, filename=os.path.join(args.output_dir, 'contour.html'))
    fig = plot_parallel_coordinate(study)
    plotly.offline.plot(fig, filename=os.path.join(args.output_dir, 'parallel_coordinate.html'))
