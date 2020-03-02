from __future__ import print_function, division
import sys 
import os 
import torch 
import numpy as np 
import random 
import csv

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils 
from torch.utils.data.sampler import Sampler

from pycocotools.coco import COCO 

import skimage.io
import skimage.transform
import skimage.color
import skimage 

from PIL import Image 

class CocoDataset(Dataset):
    def __init__(self, root_dir, set_name='train2017', transform=None):
        '''
        Args:
            root_dir (string)               : COCO directory
            transform (callable, optional)  : Optional transform to be applied
        '''
        self.root_dir = root_dir
        self.set_name = set_name 
        self.transform = transform 

        self.coco = COCO(os.path.join(self.root_dir, 'annotations', 'instances_' + self.set_name + '.json'))
        self.image_ids = self.coco.getImgIds()

        self.load_classes()

    def load_classes(self):
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        self.classes = {}
        self.coco_labels = {}
        self.coco_labels_inverse = {}

        for c in categories:
            self.coco_labels[len(self.classes)] = c['id']
            self.coco_labels_inverse[c['id']] = len(self.classes)
            self.classes[c['name']] = len(self.classes)

        # also load the reverse (label -> name)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key 

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img = self.load_image(idx)
        annot = self.load_annotations(idx)
        sample = {'img': img, 'annot': annot}
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
    def load_image(self, image_index):
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        path = os.path.join(self.root_dir, 'images', self.set_name, image_info['file_name'])
        img = skimage.io.imread(path)

        if len(img.shape) == 2:
            img = skimage.color.gray2rgb(img)
        
        return img.astype(np.float32) / 255.0

    def load_annotations(self, image_index):
        # get ground truth annotations
        annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_index], iscrowd=False)
        annotations = np.zeros((0, 5))

        # some images appear to miss annotations
        if len(annotations_ids) == 0:
            return annotations

        # parse annotations
        coco_annotations = self.coco.loadAnns(annotations_ids)
        for idx, a in enumerate(coco_annotations):
            # some annotations have basically no width / height, skip them
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue

            annotation = np.zeros((1, 5))
            annotation[0, :4] = a['bbox']
            annotation[0, 4] = self.coco_label_to_label(a['category_id'])
            annotations = np.append(annotations, annotation, axis=0)

        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations

    def coco_label_to_label(self, coco_label):
        return self.coco_labels_inverse[coco_label]

    def label_to_coco_label(self, label):
        return self.coco_labels[label]

    def image_aspect_ratio(self, image_index):
        image = self.coco.loadImgs(self.image_ids[image_index])[0]
        return float(image['width']) / float(image['height'])

    def num_classes(self):
        return 80

class CSVDataset(Dataset):
    def __init__(self, train_file, class_list, transform=None):
        '''
        Args:
            train_file (string)             : csv file with trainning annotations
            annotations (string)            : csv file with class list
            test_file (string, optional)    : csv file with testing annotations
        '''
        self.train_file = train_file 
        self.class_list = class_list 
        self.transform = transform 

        # parse the provided class file
        try:
            with self._open_for_csv(self.class_list) as file:
                self.classes = self.load_classes(csv.reader(file, delimiter=','))
        except ValueError as e:
            raise ValueError('invalid csv class file: {}: {}'.format(self.class_list, e))

        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key 

        # csv with img_path, x1, y1, x2, y2, class_name
        try:
            with self._open_for_csv(self.train_file) as file:
                self.image_data = self._read_annotations(csv.reader(file, delimiter=','), self.classes)
        except ValueError as e:
            raise ValueError('invalid csv annotations file: {}: {}'.format(self.train_file, e))
        
        self.image_names = list(self.image_data.keys())

    def _parse(self, value, function, fmt):
        ''' Parse a string into a value, and format a nice ValueError if it fails.
        Returns `function(value)`.
        Any `ValueError` raised is catched and a ndw `ValueError` is raised with message 
        `fmt.format(e)`, where `e` is the caught `ValueError`.
        '''
        try:
            return function(value)
        except ValueError as e:
            raise ValueError(fmt.format(e))

    def _open_for_csv(self, path):
        ''' Open a file with flags sutable for csv.reader
        This is different for python2 it means with mode `rb`,
        for python3 this means `r` with `universal newlines.
        '''
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
                raise ValueError('line {}: format should be \'class_name, class_id\''.format(line)) 
            
            class_id = self._parse(class_id, int, 'line {}: malformed class ID: {{}}'.format(line))

            if class_name in result:
                raise ValueError('line {}: duplicate class name: \'{}\''.format(line, class_name))

            result[class_name] = class_id

        return result 

    def __len__(self):
        return len(self.iamge_names)

    def __getitem__(self, idx):
        img = self.load_image(idx)
        annot = self.load_annotations(idx)
        sample = {'img': img, 'annot': annot}
        if self.transform:
            sample = self.transform(sample)
        
        returnn sample

    def load_image(self, image_index):
        img = skimage.io.imread(self.image_names[image_index])

        if len(img.shape) == 2:
            img = skimage.color.gray2rgb(img)

        return img.astype(np.float32) / 255.0

    def load_annotations(self, image_index):
        

    
        


