import argparse
import json
import os

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader

from detection import models, utils, datasets
from detection import trainingtools
from detection.datasets import FathomNetDataset
import logging


def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description="Marine Organism Object Detection Training", add_help=add_help)

    parser.add_argument("--data-path", default="data", type=str, help="dataset path")
    parser.add_argument("--class-file", default="classes.json", type=str, help="path to class definitions")
    parser.add_argument("--dataset", default="FathomNet", type=str, help="dataset name")
    parser.add_argument("--model", default="rcnn_resnet50_fpn", type=str, help="model name")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument(
        "--train-ratio", "--tr", default=0.7, type=float, help="proportion of dataset to use for training"
    )
    parser.add_argument(
        "-b", "--batch-size", default=2, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
    )
    parser.add_argument("--epochs", default=5, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument("--checkpoint", default=None, type=str, help="path of stored checkpoint")
    parser.add_argument("--start-epoch", default=0, type=int, help="start epoch")
    parser.add_argument(
        "-j", "--workers", default=4, type=int, metavar="N", help="number of data loading workers (default: 4)"
    )
    parser.add_argument(
        "--lr", default=0.02, type=float,
        help="initial learning rate, 0.02 is the default value for training on 8 gpus and 2 images_per_gpu",
    )
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument(
        "--wd", "--weight-decay", default=1e-4, type=float, metavar="W", help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument(
        "--lr-step-size", default=8, type=int, help="decrease lr every step-size epochs (multisteplr scheduler only)"
    )
    parser.add_argument(
        "--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma (multisteplr scheduler only)"
    )
    parser.add_argument("--output-dir", default=".", type=str, help="path to save outputs")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument("--aspect-ratio-group-factor", default=3, type=int)
    parser.add_argument("--rpn-score-thresh", default=None, type=float, help="rpn score threshold for faster-rcnn")
    parser.add_argument(
        "--data-augmentation", default="hflip", type=str, help="data augmentation policy (default: hflip)"
    )
    parser.add_argument("--sync-bn", dest="sync_bn", help="Use sync batch norm", action="store_true")
    parser.add_argument("--test-only", dest="test_only", help="Only test the model", action="store_true")
    parser.add_argument(
        "--pretrained", dest="pretrained", help="Use pre-trained models from the modelzoo", action="store_true"
    )
    parser.add_argument("--log-file", "--lf", default=None, type=str, help="path to file for writing logs. If "
                                                                           "omitted, writes to stdout")
    parser.add_argument("--log-level", default="ERROR", choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"))
    parser.add_argument("--log-every", "--pe", default=10, type=int, help="log every ith batch")
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")

    return parser


def main(args):
    # TODO: Check arguments

    utils.initialise_distributed(args)
    utils.initialise_logging(args)
    logging.info("Started")

    # Parse class definition file
    logging.info("Loading class definitions...")
    with open(args.class_file) as f:
        classes = json.load(f)
    num_classes = len(classes) + 1
    logging.info(f"Training with {num_classes} classes: " + ", ".join(['background'] + list(classes.keys())))

    # Load data
    logging.info("Loading dataset...")
    train_dataset, test_dataset = datasets.load_datasets(name=args.dataset, root=args.data_path, classes=classes,
                                                         train_ratio=args.train_ratio)

    print("Creating data loaders...")
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        test_sampler = torch.utils.data.SequentialSampler(test_dataset)

    train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, args.batch_size, drop_last=True)

    train_loader = DataLoader(train_dataset, batch_sampler=train_batch_sampler, num_workers=args.workers,
                              collate_fn=utils.collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=1, sampler=test_sampler, num_workers=args.workers,
                             collate_fn=utils.collate_fn)

    # Load model
    device = torch.device(args.device)
    logging.info(f"Using device: {device}")
    logging.info("Loading model...")
    model = models.load_model(args.model, num_classes=num_classes, pretrained=args.pretrained)
    model = model.to(device)
    if args.distributed:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], output_device=args.gpu)

    # Observe that all parameters are being optimized
    params = [p for p in model.parameters() if p.requires_grad]
    optimiser = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=args.lr_step_size, gamma=args.lr_gamma)

    if args.checkpoint:
        logging.info("Resuming from checkpoint...")
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        model.module.load_state_dict(checkpoint["model"])
        optimiser.load_state_dict(checkpoint["optimiser"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])

    # Train the model
    logging.info("Beginning training:")
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        # Train one epoch
        trainingtools.train_one_epoch(model, train_loader, device=device, optimiser=optimiser,
                                      epoch=epoch, n_epochs=args.epochs, log_every=args.log_every, scaler=scaler)

        # Update the learning rate
        lr_scheduler.step()

        if args.output_dir:
            checkpoint = {
                "model": model.module.state_dict(),
                "optimizer": optimiser.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "args": args,
                "epoch": epoch,
            }
            if args.amp:
                checkpoint["scaler"] = scaler.state_dict()
            utils.save_state(checkpoint, args.output_dir, epoch, args.distributed)

        # Evaluate on the test data
        # trainingtools.evaluate(model, test_loader, device=device)


if __name__ == '__main__':
    parsed_args = get_args_parser().parse_args()
    main(parsed_args)
