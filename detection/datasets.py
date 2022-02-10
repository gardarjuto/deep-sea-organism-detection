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
    def __init__(self, root, classes=None, transforms=None, subset=None):
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


def load_train_val(name, train_path, classes, val_split):
    if name == 'FathomNet':
        train_transforms = Compose([ToTensor(), RandomHorizontalFlip()])
        val_transforms = Compose([ToTensor()])
        length = len(os.listdir(os.path.join(train_path, 'images')))
        train_idx, val_idx = train_test_split(range(length), test_size=val_split)
        train_dataset = FathomNetDataset(root=train_path, classes=classes, transforms=train_transforms, subset=train_idx)
        val_dataset = FathomNetDataset(root=train_path, classes=classes, transforms=val_transforms, subset=val_idx)
    elif name == 'FathomNetCropped':
        train_transforms = T.Compose([T.ToTensor(), T.RandomHorizontalFlip()])
        val_transforms = T.Compose([T.ToTensor()])
        length = len(os.listdir(os.path.join(train_path, 'images')))
        train_idx, val_idx = train_test_split(range(length), test_size=val_split)
        train_dataset = FathomNetCroppedDataset(root=train_path, classes=classes, transforms=train_transforms, subset=train_idx)
        val_dataset = FathomNetCroppedDataset(root=train_path, classes=classes, transforms=val_transforms, subset=val_idx)
    elif name == 'mixed':
        train_transforms = T.Compose([T.ToTensor(), T.RandomHorizontalFlip()])
        val_transforms = Compose([ToTensor()])
        length = len(os.listdir(os.path.join(train_path, 'images')))
        train_idx, val_idx = train_test_split(range(length), test_size=val_split)
        train_dataset = FathomNetCroppedDataset(root=train_path, classes=classes, transforms=train_transforms, subset=train_idx)
        val_dataset = FathomNetDataset(root=train_path, classes=classes, transforms=val_transforms, subset=val_idx)
    else:
        raise NotImplementedError('Currently supported datasets: FathomNet, FathomNetCropped')
    return train_dataset, val_dataset


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
