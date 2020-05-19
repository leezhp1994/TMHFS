"""
This code based on codes from https://github.com/tristandeleu/ntm-one-shot \
                              and https://github.com/kjunelee/MetaOptNet
"""
import numpy as np
import random, os
import pickle as pkl
from PIL import Image
import pandas as pd
import utils.generator.additional_transforms as add_transforms
import torchvision.transforms as transforms
from .auto_augment import AutoAugment, Cutout

class TransformLoader:
    def __init__(self, image_size,
                 normalize_param=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                 jitter_param=dict(Brightness=0.4, Contrast=0.4, Color=0.4)):
        self.image_size = image_size
        self.normalize_param = normalize_param
        self.jitter_param = jitter_param

    def parse_transform(self, transform_type):
        if transform_type == 'ImageJitter':
            method = add_transforms.ImageJitter(self.jitter_param)
            return method
        if transform_type == 'AutoAugment':
            method = AutoAugment()
            return method
        method = getattr(transforms, transform_type)
        if transform_type == 'RandomResizedCrop':
            return method(self.image_size)
        elif transform_type == 'CenterCrop':
            return method(self.image_size)
        elif transform_type == 'Resize':
            return method([int(self.image_size * 1), int(self.image_size * 1)])
        elif transform_type == 'Normalize':
            return method(**self.normalize_param)
        elif transform_type == 'RandomRotation':
            return method(45)
        else:
            return method()

    def get_composed_transform(self):

        transform_lists = [['RandomResizedCrop','RandomResizedCrop', 'ImageJitter', 'ColorJitter', 'RandomRotation',                            'RandomHorizontalFlip'],
                              ['Resize', 'ImageJitter', 'RandomHorizontalFlip'],
                              ['RandomResizedCrop'],
                              ['RandomResizedCrop', 'ImageJitter'],
                              ['RandomResizedCrop', 'RandomHorizontalFlip'],
                              ['RandomResizedCrop', 'ImageJitter', 'RandomHorizontalFlip'], 
                              ['RandomResizedCrop'],
                              ['RandomResizedCrop','ImageJitter'],
                              ['RandomResizedCrop', 'ImageJitter', 'RandomHorizontalFlip'],
                              ['RandomResizedCrop', 'RandomHorizontalFlip']
                             ]
        transforms_all = []
        for transform_list in transform_lists:
            transform_funcs = [self.parse_transform(x) for x in transform_list]
            transforms_all.append(transforms.Compose(transform_funcs))
        return transforms_all
    
class miniImageNetGenerator(object):

    def __init__(self, data_file, nb_classes=5, nb_samples_per_class=15,
                  max_iter=None, xp=np, aug_num=10):
        super(miniImageNetGenerator, self).__init__()
        self.csv_path = data_file + "/Data_Entry_2017.csv"
        self.image_path = data_file + "/images/"
        self.data_file = data_file
        self.nb_classes = nb_classes
        self.nb_samples_per_class = nb_samples_per_class
        self.max_iter = max_iter
        self.xp = xp
        self.num_iter = 0
        self.aug_num = aug_num
        self.trans_loader = TransformLoader(84)

        self.used_labels = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia",
                            "Pneumothorax"]

        self.labels_maps = {"Atelectasis": 0, "Cardiomegaly": 1, "Effusion": 2, "Infiltration": 3, "Mass": 4,
                            "Nodule": 5, "Pneumothorax": 6}

        labels_set = []

        # Read the csv file
        self.data_info = pd.read_csv(self.csv_path, skiprows=[0], header=None)

        # First column contains the image paths
        self.image_name_all = np.asarray(self.data_info.iloc[:, 0])
        self.labels_all = np.asarray(self.data_info.iloc[:, 1])

        self.image_name = []
        self.labels = []
  

        self.data = self._load_data()

    def _load_data(self):
        data = {}
        for key, value in self.labels_maps.items():
            data[value] = []
        for name, label in zip(self.image_name_all, self.labels_all):
            label = label.split("|")
            if len(label) == 1 and label[0] != "No Finding" and label[0] != "Pneumonia" and label[
                0] in self.used_labels:
                _label = self.labels_maps[label[0]]
                data[_label].append(name)

        return data

    def load_data(self, data_file):
        try:
            with open(data_file, 'rb') as fo:
                data = pkl.load(fo)
            return data
        except:
            with open(data_file, 'rb') as f:
                u = pkl._Unpickler(f)
                u.encoding = 'latin1'
                data = u.load()
            return data

    def buildLabelIndex(self, labels):
        label2inds = {}
        for idx, label in enumerate(labels):
            if label not in label2inds:
                label2inds[label] = []
            label2inds[label].append(idx)

        return label2inds


    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if (self.max_iter is None) or (self.num_iter < self.max_iter):
            self.num_iter += 1
            images, labels = self.sample(self.nb_classes, self.nb_samples_per_class)

            return (self.num_iter - 1), (images, labels)
        else:
            raise StopIteration()

    def augment(self, path, aug_ind=0):
        path = os.path.join(self.image_path, path)
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                img = img.convert('RGB')
        if aug_ind==0:
            img = img.resize((84, 84), Image.BILINEAR)
            imgs = np.array(img)
        elif aug_ind==-1:
            imgs = np.array(img)
        else:
            transform_all = self.trans_loader.get_composed_transform()
            imgs=np.array(transform_all[aug_ind](img))
        return imgs

    def sample(self, nb_classes, nb_samples_per_class):
        sampled_characters = random.sample(self.data.keys(), nb_classes)
        labels = []
        images = []
        _ind_all = []
        _imgss=[]
        for ii in range(self.aug_num):
            
            labels_and_images = []
            for (k, char) in enumerate(sampled_characters):
                _imgs = self.data[char]
                if ii ==0:
                    _ind_all.append(random.sample(range(len(_imgs)), nb_samples_per_class))
                    
                _ind = _ind_all[k]
                if k==0:
                    _imgss.append(_imgs[_ind[1]])
    
                labels_and_images.extend([(k, self.xp.array(self.augment(_imgs[i],ii)/np.float32(255).flatten())) for i in _ind])
            arg_labels_and_images = []
            for i in range(self.nb_samples_per_class):
                for j in range(self.nb_classes):
                    arg_labels_and_images.extend([labels_and_images[i+j*self.nb_samples_per_class]])
            labels_temp, images_temp = zip(*arg_labels_and_images)
            labels.append(labels_temp)
            images.append(images_temp)
        return images, labels_temp
