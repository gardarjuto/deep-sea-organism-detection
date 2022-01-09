import math
import os
import sys
import torch
from PIL import Image
from torchvision.transforms import transforms, functional
from torchvision.utils import draw_bounding_boxes
import logging


def train_one_epoch(model, loader, device, optimiser, epoch, n_epochs, log_every=None, scaler=None):
    model.train()

    total_loss_classifier = 0.0
    total_loss_box_reg = 0.0
    total_loss_objectness = 0.0
    total_loss_rpn_box_reg = 0.0

    for i, (images, targets) in enumerate(loader, start=1):
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        # if not any([len(target['labels']) > 0 for target in targets]):
        #     print("Continuing")
        #     continue
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        loss_value = losses.item()

        if not math.isfinite(loss_value):
            logging.error(f"Loss is {loss_value}, stopping training")
            logging.error(loss_dict)
            sys.exit(1)

        total_loss_classifier += loss_dict['loss_classifier']
        total_loss_box_reg += loss_dict['loss_box_reg']
        total_loss_objectness += loss_dict['loss_objectness']
        total_loss_rpn_box_reg += loss_dict['loss_rpn_box_reg']

        optimiser.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimiser)
            scaler.update()
        else:
            losses.backward()
            optimiser.step()

        if log_every and i % log_every == 0:
            logging.info(f"Epoch [{epoch}/{n_epochs}]  [{i}/{len(loader)}]  " +
                         ", ".join([f"{loss_type}: {loss.item():.3f}" for loss_type, loss in loss_dict.items()]))

    logging.info(f'Summary:')
    logging.info(f'\tloss_classifier (mean): {total_loss_classifier.item() / len(loader):.3f}, '
                 f'loss_box_reg: {total_loss_box_reg.item() / len(loader):.3f}, '
                 f'loss_objectness: {total_loss_objectness.item() / len(loader):.3f}, '
                 f'loss_rpn_box_reg (mean): {total_loss_rpn_box_reg.item() / len(loader):.3f}')


def visualise_prediction(model, device, img_name, dataset, show_ground_truth=True):
    model.eval()
    idx = dataset.index_of(img_name)
    img, targets = dataset[idx]
    img = [img.to(device)]
    targets = {k: v.to(device) for k, v in targets.items()}
    prediction = model(img)[0]
    boxes = prediction['boxes']
    labels = ['pred_(' + dataset.from_label(lab) + ')' for lab in prediction['labels'].tolist()]
    colours = ['blue'] * len(prediction['boxes'])
    if show_ground_truth:
        boxes = torch.cat((targets['boxes'], boxes), dim=0)
        labels = ['true_(' + dataset.from_label(lab) + ')' for lab in targets['labels'].tolist()] + labels
        colours = ['red'] * len(targets['boxes']) + colours
    image255 = Image.open(os.path.join(dataset.root, 'images', img_name)).convert('RGB')
    image255 = functional.pil_to_tensor(image255)
    res = draw_bounding_boxes(image255, boxes, labels, colours, width=3)
    return functional.to_pil_image(res)


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()

    for images, targets in loader:
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        predictions = model(images)
        predictions = [{k: v.to(device) for k, v in t.items()} for t in predictions]

        # TODO: Actually evaluate results