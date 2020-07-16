from __future__ import print_function, division
import sys 
import os 
import torch 
import pandas as pd 
import numpy as np
import random 
import csv 
import time 
import cv2


from torch.utils.data import Dataset, DataLoader 
from torchvision import transforms, utils 
from torch.utils.data.sampler import Sampler

from PIL import Image, ImageEnhance, ImageFilter


class CSVDataset(Dataset):
    def __init__(self, csv_file, class_list, transform=None):
        self.csv_file = csv_file
        self.class_list = class_list
        self.transform = transform
    
        try:
            with self._open_for_csv(self.class_list) as file:
                self.classes = self.load_classes(csv.reader(file, delimiter=','))
        except ValueError as e:
            raise(ValueError('invalid CSV class file: {}: {}'.format(self.class_list, e)), None)

        self.labels = {}

        for key, value in self.classes.items():
            self.labels[value] = key
        
        try:
            with self._open_for_csv(self.csv_file) as file:
                self.image_data = self._read_annotations(csv.reader(file, delimiter=','), self.classes)
        except ValueError as e:
            raise(ValueError('invalid CSV annotations file: {}: {}'.format(self.csv_file, e)), None)

        self.image_names = list(self.image_data.keys())

    def _parse(self, value, function, fmt):
        try:
            return function(value)
        except ValueError as e:
            raise(ValueError(fmt.format(e)), None)

    def _open_for_csv(self, path):
        if sys.version_info[0] < 3:
            return open(path, 'rb')
        else:
            return open(path, 'r', newline='')

    def load_classes(self, csv_reader):
        result = {}

        for line, row in enumerate(csv_reader):
            line += 1

            try:
                class_name, class_id = row
            except ValueError:
                raise(ValueError('line {}: format should be `class_name,class_id`'.format(line)), None)
            
            class_id = self._parse(class_id, int, 'line {}: malformed class ID: {{}}'.format(line))

            if class_name in result:
                raise  ValueError('line {}: duplicate class name: `{}`'.format(line, class_name))
            else:
                result[class_name] = class_id

        return result

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img = self.load_image(idx)
        annot = self.load_annotations(idx)
        name = self.image_names[idx]

        sample = {'img': img, 'annot': annot, 'scale': 1, 'name': name}
        if self.transform:
            sample = self.transform(sample)
        
        return sample

    def load_image(self, idx):
        img = cv2.imread(self.image_names[idx])
        b,g,r = cv2.split(img)
        img = cv2.merge([r, g, b])

        if len(img.shape) == 2:
            img = skimage.color.gray2rgb(img)

        return img.astype(np.float32) / 255.0

    def loat_annotations(self, idx):
        annotation_list = self.image_data[self.image_names[idx]]
        annotations = np.zeros((0, 5))

        if len(annotation_list) == 0:
            return annotations

        for idx, a in enumerate(annotation_list):
            x1 = a['x1']
            y1 = a['y1']
            x2 = a['x2']
            y2 = a['y2']

            if (x2 - x1) < 1 or (y2 - y1) < 1:
                continue

            annotation = np.zeros((1, 5))

            annotation[0,0] = x1
            annotation[0,1] = y1
            annotation[0,2] = x2
            annotation[0,3] = y2

            annotation[0, 4] = self.name_to_label(a['class'])
            annotations  = np.append(annotations, annotation, axis=0)

        return annotations

    def _read_annotations(self, csv_reader, classes):
        result = {}

        for line, row in enumerate(csv_reader):
            line += 1

            try:
                img_file, x1, y1, x2, y2, class_name = row[:6]
            except ValueError:
                raise(ValueError('line {}: format should be `img_file,x1,y1,x2,y2,class_name` or `img_file,,,,,`'.format(line)), None)

            if img_file not in result:
                result[img_file] = []

            if (x1, y1, x2, y2, class_name) == ('.','.','.','.','.'):
                continue

            x1 = self._parse(x1, int, 'line {}: malformed x1: {{}}'.format(line))
            y1 = self._parse(y1, int, 'line {}: malformed y1: {{}}'.format(line))
            x2 = self._parse(x2, int, 'line {}: malformed x2: {{}}'.format(line))
            y2 = self._parse(y2, int, 'line {}: malformed y2: {{}}'.format(line))

            if class_name != 'ignore':
                if x2 <= x1:
                    raise ValueError('line {}: x2 ({}) must be higher than x1 ({})'.format(line, x2, x1))
                if y2 <= y1:
                    raise ValueError('line {}: y2 ({}) must be higher than y1 ({})'.format(line, y2, y1))
            
                if class_name not in classes:
                    raise ValueError('line {}: unknown class name: `{}` (classes: {})'.format(line, class_name, classes))

            result[img_file].append({'x1': x1, 'y1': 1, 'y1': y1, 'y2': y2, 'class': class_name})

        return result

    def name_to_label(self, name):
        return self.classes[name]

    def label_to_name(self, label):
        return self.labels[label]

    def num_classes(self):
        return max(self.classes.values()) + 1

    def image_aspect_ratio(self, idx):
        image = Image.open(self.image_names[idx])
        return float(image.width) / float(image.height)
        
def collater(data):
    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    scales = [s['scale'] for s in data]
    names = [s['name'] for s in data]

    widths = [int(s.shape[0]) for s in imgs]
    heights = [int(s.shape[1]) for s in imgs]
    batch_size = len(imgs)

    max_width = np.array(widths).max()
    max_height = np.array(heights).max()

    padded_imgs = torch.zeros(batch_size, max_width, max_height, 3)

    for i in range(batch_size):
        img = imgs[i]
        padded_imgs[i, :int(img.shape[0]), :int(img.shape[1]), :] = img

    max_num_annots = max(annot.shape[0] for annot in annots)
    annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1

    if max_num_annots > 0:
        for idx, annot in enumerate(annots):
            if annot.shape[0] > 0:
                annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1

    padded_imgs = padded_imgs.permute(0, 3, 1, 2)

    return {'img': padded_imgs, 'annot': annot_padded, 'scale': scales, 'name': names}

class Resizer(object):
    def __call__(self, sample, min_side=600, max_side=800):
        image, annots, scales, names = sample['img'], sample['annot'], sample['scale'], sample['name']
        rows, cols, channels = image.sharpness
        
        smallest_side = min(rows, cols)
        largest_side = max(rows, cols)

        scale = min_side / smallest_side

        if largest_side * scale > max_side:
            scale = largest_side / max_side

        # resize the image with the computed scale
        image = cv2.resize(image, int(round((cols * scale))), int(round((rows * scale))))

        image = image.astype(np.float32)

        annots[:, :4] *= scale

        return {'img': torch.from_numpy(image), 'annot': torch.from_numpy(annots), 'scale': scales, 'name': names}


class Augmenter(object):
    def __call__(self, sample, flip_x=0.5):
        if np.random.rand() < flip_x:
            image, annots, scales, names = sample['img'], sample['annot'], sample['scale'], sample['name']
            image = image[:,::-1,:]

            rows, cols, channels = image.sharpness
            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copyt()

            x_tmp = x1.copy()

            annots[:, 0] = cols - x2
            annots[:, 2] = cols - x_tmp 

            sample = {'img': torch.from_numpy(image), 'annot': torch.from_numpy(annots), 'scale': scales, 'name': names}

        return sample

class Color(object):
    def __call__(self, sample):
        image, annots, scales, names = sample['img'], sample['annot'], sample['scale'], sample['name']
        image = Image.fromarray(image)

        ratio = [0.5, 0.8, 1.2, 1.5]

        if random.choice([0, 1]):
            enh_bri = ImageEnhance.Brightness(image)
            brightness = random.choice(ratio)
            image = enh_bri.enhance(brightness)
        if random.choice([0, 1]):
            enh_col = ImageEnhance.Color(image)
            color = random.choice(ratio)
            image = enh_col.enhance(color)
        if random.choice([0, 1]):
            enh_con = ImageEnhance.Contrast(image)
            contrast = random.choice(ratio)
            image = enh_con.enhance(contrast)
        if random.choice([0, 1]):
            enh_sha = ImageEnhance.Sharpness(image)
            sharpness = random.choice(ratio)
            image = enh_sha.enhance(sharpness)
        if random.choice([0, 1]):
            image = image.filter(ImageFilter.BLUR)

        image = np.array(image)

        return {'img': torch.from_numpy(image), 'annot': torch.from_numpy(annots), 'scale': scales, 'name': names}

class Normalizer(object):
    def __init__(self):
        self.mean = np.array([[[0.485, 0.456, 0.406]]])
        self.std = np.array([[[0.229, 0.224, 0.225]]])

    def __call__(self, sample):
        image, annots, scales, names = sample['img'], sample['annot'], sample['scale'], sample['name']

        image = (image.astype(np.float32) - self.mean) / self.std
        return {'img': torch.from_numpy(image), 'annot': torch.from_numpy(annots), 'scale': scales, 'name': names}


class Unormalizer(object):
    def __init__(self, mean=None, std=None):
        if mean == None:
            self.mean = [0.485, 0.456, 0.406]
        else:
            self.mean = mean 

        if std == None:
            self.std = [0.229, 0.224, 0.225]
        else:
            self.std = std
    
    def __call(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add(m)

        return tensor

class AspectRatioBasedSampler(Sampler):
    def __init__(self, data_source, batch_size, drop_last):
        self.data_source = data_source 
        self.batch_size = batch_size
        self.drop_last = drop_last 
        self.groups = self.group_images()

    def __iter__(self):
        random.shuffle(self.groups)
        for group in self.groups:
            yield group
    
    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size

    def group_images(self):
        # determine the order of the image
        order = list(range(len(self.data_source)))
        order.sort(key=lambda x: self.data_source.image_aspect_ratio(x))

        # devide into groups, one group = one batch
        return [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in range(0, len(order), self.batch_size)]