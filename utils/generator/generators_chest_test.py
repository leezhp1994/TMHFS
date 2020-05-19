"""
This code based on codes from https://github.com/tristandeleu/ntm-one-shot \
                              and https://github.com/kjunelee/MetaOptNet
"""
import numpy as np
import random, os
import pickle as pkl
from PIL import Image
import pandas as pd

img_size=84

class miniImageNetGenerator(object):

    def __init__(self, data_file, nb_classes=5, nb_samples_per_class=15,
                  max_iter=None, xp=np):
        super(miniImageNetGenerator, self).__init__()
        self.csv_path = data_file + "/Data_Entry_2017.csv"
        self.image_path = data_file + "/images/"
        self.data_file = data_file
        self.nb_classes = nb_classes
        self.nb_samples_per_class = nb_samples_per_class
        self.max_iter = max_iter
        self.xp = xp
        self.num_iter = 0

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
        # self.data_len = len(self.image_name)
        #
        # self.image_name = np.asarray(self.image_name)
        # self.labels = np.asarray(self.labels)

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

    def augment(self, path):
        path = os.path.join(self.image_path, path)
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                img = img.convert('RGB')
        # trans = transforms.ToTensor()
        # img = Image.fromarray(img.astype('uint8')).convert('RGB')
        # w, h = img.size
        # l = min(w, h)
        # new_w = int(w * (84 / l))
        # new_h = int(h * (84 / l))
        img = img.resize((img_size, img_size), Image.BILINEAR)
        # x, y, w, h = 16, 16, 84, 84
        # img = img.crop((x, y, x + w, y + h))
        img = np.array(img)
        return img

    def sample(self, nb_classes, nb_samples_per_class):
        sampled_characters = random.sample(self.data.keys(), nb_classes)
        labels_and_images = []
        for (k, char) in enumerate(sampled_characters):
            _imgs = self.data[char]
            _ind = random.sample(range(len(_imgs)), nb_samples_per_class)
            labels_and_images.extend([(k, self.xp.array(self.augment(_imgs[i])/np.float32(255).flatten())) for i in _ind])
        arg_labels_and_images = []
        for i in range(self.nb_samples_per_class):
            for j in range(self.nb_classes):
                arg_labels_and_images.extend([labels_and_images[i+j*self.nb_samples_per_class]])

        labels, images = zip(*arg_labels_and_images)
        return images, labels

