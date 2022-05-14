import argparse
import json
import os
import joblib
import torch
from detection import models, utils, datasets
from detection import trainingtools
import logging
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms.transforms import ToTensor


def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description="Script for applying trained object detection models on data",
                                     add_help=add_help)

    # General
    parser.add_argument("--image-path", type=str, help="path to directory of images to run model on")
    parser.add_argument("--class-file", default="classes.json", type=str, help="path to class definitions")
    parser.add_argument("--model", default="rcnn_resnet50_fpn", type=str, help="model name")
    parser.add_argument("--model-path", type=str, help="path to stored model")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")

    parser.add_argument("--output-dir", default=".", type=str, help="path to save outputs")
    parser.add_argument("--pretrained", action="store_true", help="Use a pretrained model")
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.set_defaults(pretrained=True)
    parser.add_argument("--iou-thresh", default=0.5, type=float, help="IoU threshold for evaluation")

    # Logging
    parser.add_argument("--log-file", "--lf", default=None, type=str,
                        help="path to file for writing logs. If omitted, writes to stdout")
    parser.add_argument("--log-level", default="ERROR", choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"))

    # HOG params
    parser.add_argument("--hog-bins", default=9, type=int, help="number or bins for orientation binning in HOG")
    parser.add_argument("--ppc", default=(8, 8), nargs="+", type=int, help="pixels per cell in HOG")
    parser.add_argument("--cpb", default=(2, 2), nargs="+", type=int, help="cells per block in HOG")
    parser.add_argument("--block-norm", default="L2-Hys", type=str, help="block norm in HOG")
    parser.add_argument("--gamma-corr", action="store_true", help="use gamma correction in HOG")
    parser.add_argument("--no-gamma-corr", action="store_false", dest="gamma-corr",
                        help="don't use gamma correction in HOG")
    parser.set_defaults(gamma_corr=True)
    parser.add_argument("--hog-dim", default=(128, 128), nargs="+", type=int, help="input dimensions for HOG extractor")

    return parser


def main(args):
    utils.initialise_logging(args)
    logging.info("Started")

    if not os.path.exists(args.output_dir):
        logging.info("Creating output directory...")
        os.makedirs(args.output_dir)

    # Parse class definition file
    logging.info("Loading class definitions...")
    with open(args.class_file, "r") as f:
        classes = json.load(f)
    num_classes = len(classes) + 1
    logging.info(f"Using model with {num_classes} classes: " + ", ".join(['background'] + list(classes.keys())))


    # Load model
    device = torch.device(args.device)
    logging.info(f"Using device: {device}")
    logging.info("Loading model...")
    if args.model == 'HOGSVM':
        feature_extractor = models.HOG(orientations=args.hog_bins, pixels_per_cell=args.ppc, cells_per_block=args.cpb,
                                       block_norm=args.block_norm, gamma_corr=args.gamma_corr, resize_to=args.hog_dim)
        clf = joblib.load(args.model_path)
        for image_name in os.listdir(args.image_path):
            image = Image.open(os.path.join(args.image_path, image_name)).convert('RGB')
            predictions = trainingtools.get_predictions(clf, feature_extractor, image)
            utils.visualise_image(image, predictions=predictions, name_mapping=lambda label: sorted(classes)[label-1])
            plt.show()
    else:
        model = models.load_model(args.model, num_classes=num_classes, pretrained=args.pretrained)
        model = model.to(device)

        checkpoint = torch.load(args.model_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        model.eval()

        trans = ToTensor()
        for image_name in os.listdir(args.image_path):
            image = Image.open(os.path.join(args.image_path, image_name)).convert('RGB')
            image = trans(image)
            predictions = model([image])
            predictions = [{k: v.detach().numpy() for k, v in t.items()} for t in predictions]
            utils.visualise_image(image, predictions=predictions[0], name_mapping=lambda label: sorted(classes)[label-1])
            plt.axis('off')
            plt.savefig(os.path.join(args.output_dir, os.path.splitext(image_name)[0] + '.pdf'), bbox_inches='tight',
                        format='pdf')


if __name__ == '__main__':
    parsed_args = get_args_parser().parse_args()
    main(parsed_args)
