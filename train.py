import argparse
import json
import os

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from utils import models
from utils import trainingtools
from utils.datasets import FathomNetDataset


def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description="Marine Organism Object Detection Training", add_help=add_help)

    parser.add_argument("--data-path", default="data", type=str, help="dataset path")
    parser.add_argument("--class-path", default="classes.json", type=str, help="path to class definitions")
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
        "--lr-scheduler", default="multisteplr", type=str, help="name of lr scheduler (default: multisteplr)"
    )
    parser.add_argument(
        "--lr-step-size", default=8, type=int, help="decrease lr every step-size epochs (multisteplr scheduler only)"
    )
    parser.add_argument(
        "--lr-steps", default=[16, 22], nargs="+", type=int,
        help="decrease lr every step-size epochs (multisteplr scheduler only)",
    )
    parser.add_argument(
        "--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma (multisteplr scheduler only)"
    )
    parser.add_argument("--print-progress", dest="print_progress", help="print training progress", action="store_true")
    parser.add_argument("--output-dir", default=".", type=str, help="path to save outputs")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument("--aspect-ratio-group-factor", default=3, type=int)
    parser.add_argument("--rpn-score-thresh", default=None, type=float, help="rpn score threshold for faster-rcnn")
    parser.add_argument(
        "--trainable-backbone-layers", default=None, type=int, help="number of trainable layers of backbone"
    )
    parser.add_argument(
        "--data-augmentation", default="hflip", type=str, help="data augmentation policy (default: hflip)"
    )
    parser.add_argument("--sync-bn", dest="sync_bn", help="Use sync batch norm", action="store_true")
    parser.add_argument("--test-only", dest="test_only", help="Only test the model", action="store_true")
    parser.add_argument(
        "--pretrained", dest="pretrained", help="Use pre-trained models from the modelzoo", action="store_true"
    )
    parser.add_argument("--print-every", "--pe", default=10, type=int, help="print every ith batch")

    return parser


def main(args):
    # TODO: Check arguments

    # Parse class definition file
    print("Loading class definitions...")
    with open(args.class_path) as f:
        classes = json.load(f)
    num_classes = len(classes) + 1
    print(f"Training with {num_classes} classes:")
    print(", ".join(['background'] + list(classes.keys())))

    # Load data
    print("Loading dataset...")
    dataset = FathomNetDataset(root=args.data_path, classes=classes,
                               transforms=trainingtools.get_transforms(train=True))

    train_size = int(len(dataset) * args.train_ratio)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=True,
                              collate_fn=trainingtools.collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=False,
                             collate_fn=trainingtools.collate_fn)

    # Load model
    print("Loading model...")
    device = torch.device(args.device)
    model = models.load_model(args.model, num_classes=num_classes, pretrained=args.pretrained, device=device)

    # Observe that all parameters are being optimized
    params = [p for p in model.parameters() if p.requires_grad]
    optimiser = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=args.lr_step_size, gamma=args.lr_gamma)

    # Train the model
    print("Beginning training:")
    sample_image = '010b0dfe-8ae8-4f62-8516-1d96085c33e6.png'

    for epoch in range(1, args.epochs + 1):
        # Train one epoch
        trainingtools.train_one_epoch(model, train_loader, device=device, optimiser=optimiser,
                                      epoch=epoch, n_epochs=args.epochs, print_every=args.print_every)

        # Evaluate on the test data
        # trainingtools.evaluate(model, test_loader, device=device)

        # Update the learning rate
        lr_scheduler.step()

    sample_pred = trainingtools.visualise_prediction(model, device, sample_image, dataset)
    plt.imshow(sample_pred)
    plt.savefig(os.path.join(args.output_dir, 'sample_pred.png'))


if __name__ == '__main__':
    parsed_args = get_args_parser().parse_args()
    main(parsed_args)
