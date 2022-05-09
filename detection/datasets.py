import logging

import torch
import os
from PIL import Image
from xml.etree import ElementTree

from sklearn.model_selection import train_test_split
from torch import nn
from torchvision.transforms import transforms as T, functional as F

from detection import utils
from detection.fathomnethelper.json_loader import Taxonomicon


class FathomNetDataset(torch.utils.data.Dataset):
    def __init__(self, root, classes=None, transforms=None, subset=None, remove_empty=False):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, 'images'))))
        self.anns = list(sorted(os.listdir(os.path.join(root, 'annotations'))))
        self.label_mapping = {cls: i+1 for (i, cls) in enumerate(sorted(classes))}
        if subset:
            self.imgs = [self.imgs[i] for i in subset]
            self.anns = [self.anns[i] for i in subset]

        tax = Taxonomicon()

        self.class_mapping = {}
        for cls in classes:
            if type(classes) == list:
                nodes = set(tax.get_subtree_nodes(cls))
            elif type(classes) == dict:
                nodes = set.union(*[set(tax.get_subtree_nodes(cls2)) for cls2 in classes[cls]])
            else:
                raise TypeError('Class definition needs to be of type list or dict.')
            for node in nodes:
                self.class_mapping[node] = cls

        if remove_empty:
            valid_indices = []
            for idx in range(len(self.imgs)):
                ann_path = os.path.join(self.root, 'annotations', self.anns[idx])
                tree_root = ElementTree.parse(ann_path).getroot()
                for box in tree_root.iter('object'):
                    name = box.find('name').text
                    if name in self.class_mapping:
                        valid_indices.append(idx)
                        break
            self.imgs = [self.imgs[i] for i in valid_indices]
            self.anns = [self.anns[i] for i in valid_indices]

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, 'images', self.imgs[idx])
        ann_path = os.path.join(self.root, 'annotations', self.anns[idx])

        img = Image.open(img_path).convert('RGB')

        boxes = []
        labels = []
        tree_root = ElementTree.parse(ann_path).getroot()
        for box in tree_root.iter('object'):
            name = box.find('name').text
            if name in self.class_mapping:
                cls = self.class_mapping[name]
                xmin = int(box.find('bndbox/xmin').text)
                ymin = int(box.find('bndbox/ymin').text)
                xmax = int(box.find('bndbox/xmax').text)
                ymax = int(box.find('bndbox/ymax').text)
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(self.label_mapping[cls])

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        img_id_enc = utils.tensor_encode_id(os.path.splitext(self.imgs[idx])[0])

        targets = {'boxes': boxes, 'labels': labels, 'img_id': img_id_enc}

        if self.transforms is not None:
            img, targets = self.transforms(img, targets)

        return img, targets

    def __len__(self):
        return len(self.imgs)

    def index_of(self, img_id):
        return self.imgs.index(img_id)

    def get_class_name(self, label: int):
        return [k for k, v in self.label_mapping.items() if v == label][0]

    def set_transforms(self, transforms):
        self.transforms = transforms


class FathomNetCroppedDataset(torch.utils.data.Dataset):
    def __init__(self, root, classes=None, transforms=None, subset=None):
        self.root = root
        self.transforms = transforms
        imgs = list(sorted(os.listdir(os.path.join(root, 'images'))))
        tax = Taxonomicon()
        self.label_mapping = {cls: i+1 for (i, cls) in enumerate(sorted(classes))}
        if subset:
            imgs = [imgs[i] for i in subset]


        class_mapping = {}
        for cls in classes:
            if type(classes) == list:
                nodes = set(tax.get_subtree_nodes(cls))
            elif type(classes) == dict:
                nodes = set.union(*[set(tax.get_subtree_nodes(cls2)) for cls2 in classes[cls]])
            else:
                raise TypeError('Class definition must be of type list or dict.')
            for node in nodes:
                class_mapping[node] = cls

        imgs_and_boxes = {cls: [] for cls in classes}

        for img in imgs:
            ann_tree = ElementTree.parse(os.path.join(root, 'annotations', os.path.splitext(img)[0] + '.xml'))
            tree_root = ann_tree.getroot()
            for box in tree_root.iter('object'):
                name = box.find('name').text
                if name in class_mapping:
                    cls = class_mapping[name]
                    xmin = int(box.find('bndbox/xmin').text)
                    ymin = int(box.find('bndbox/ymin').text)
                    xmax = int(box.find('bndbox/xmax').text)
                    ymax = int(box.find('bndbox/ymax').text)
                    imgs_and_boxes[cls].append((img, (xmin, ymin, xmax, ymax)))

        self.boxes = []
        self.labels = []
        self.classes = []
        for name, boxes in imgs_and_boxes.items():
            self.classes.append(name)
            for box in boxes:
                self.boxes.append(box)
                self.labels.append(self.label_mapping[name])  # Map class names to integers

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, 'images', self.boxes[idx][0])
        box = self.boxes[idx][1]
        label = self.labels[idx]
        img = Image.open(img_path).convert('RGB').crop(box)

        if self.transforms is not None:
            img = self.transforms(img)

        return img, label


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor(nn.Module):
    def forward(self, image, target):
        image = F.pil_to_tensor(image)
        image = F.convert_image_dtype(image)
        return image, target


class RandomHorizontalFlip(T.RandomHorizontalFlip):
    def forward(self, image, target):
        if torch.rand(1) < self.p:
            image = F.hflip(image)
            width, _ = F.get_image_size(image)
            target["boxes"][:, [0, 2]] = width - target["boxes"][:, [2, 0]]
        return image, target


class ColorJitter(T.ColorJitter):
    def forward(self, img, target):
        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = self.get_params(self.brightness,
                                                                                                    self.contrast,
                                                                                                    self.saturation,
                                                                                                    self.hue)
        for fn_id in fn_idx:
            if fn_id == 0 and brightness_factor is not None:
                img = F.adjust_brightness(img, brightness_factor)
            elif fn_id == 1 and contrast_factor is not None:
                img = F.adjust_contrast(img, contrast_factor)
            elif fn_id == 2 and saturation_factor is not None:
                img = F.adjust_saturation(img, saturation_factor)
            elif fn_id == 3 and hue_factor is not None:
                img = F.adjust_hue(img, hue_factor)

        return img, target


def load_datasets(name, root, classes, train_ratio):
    """DEPRECATED"""
    if name == 'FathomNet':
        # train_transforms = Compose([ToTensor(), RandomHorizontalFlip()])
        trans = Compose([ToTensor()])
        dataset = FathomNetDataset(root=root, classes=classes, transforms=trans)

        train_size = int(len(dataset) * train_ratio)
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        train_dataset = remove_images_without_annotations(train_dataset)
    elif name == 'FathomNetCropped':
        trans = T.Compose([T.ToTensor()])
        dataset = FathomNetCroppedDataset(root=root, classes=classes, transforms=trans)

        train_size = int(len(dataset) * train_ratio)
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    else:
        raise NotImplementedError('Currently supported datasets: FathomNet, FathomNetCropped')
    return train_dataset, test_dataset


def load_train_val(train_path, classes, val_split, train_transforms=-1, val_transforms=-1):
    if train_transforms == -1:
        train_transforms = Compose([ToTensor(), RandomHorizontalFlip()]) #, ColorJitter(brightness=.5, contrast=.5, hue=.3)])
    if val_transforms == -1:
        val_transforms = Compose([ToTensor()])
    length = len(os.listdir(os.path.join(train_path, 'images')))
    train_idx, val_idx = train_test_split(range(length), test_size=val_split)
    train_dataset = FathomNetDataset(root=train_path, classes=classes, transforms=train_transforms, subset=train_idx, remove_empty=True)
    val_dataset = FathomNetDataset(root=train_path, classes=classes, transforms=val_transforms, subset=val_idx)
    return train_dataset, val_dataset


def load_test(test_path, classes, transforms=-1):
    if transforms == -1:
        transforms = Compose([ToTensor()])
    test_dataset = FathomNetDataset(root=test_path, classes=classes, transforms=transforms)
    return test_dataset


def remove_images_without_annotations(subset: torch.utils.data.dataset.Subset):
    logging.info("Removing images without annotations from training data")
    valid_indices = []
    dataset = subset
    if isinstance(subset, torch.utils.data.dataset.Subset):
        dataset = subset.dataset

    for i, idx in enumerate(subset.indices, start=1):
        ann_path = os.path.join(dataset.root, 'annotations', dataset.anns[idx])
        tree_root = ElementTree.parse(ann_path).getroot()
        for box in tree_root.iter('object'):
            name = box.find('name').text
            if name in dataset.class_mapping:
                valid_indices.append(idx)
                break
    new_subset = torch.utils.data.dataset.Subset(dataset, valid_indices)
    logging.info(f"Reduced training set size from {len(subset.indices)} to {len(valid_indices)}")
    return new_subset
