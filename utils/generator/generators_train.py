"""
This code based on codes from https://github.com/tristandeleu/ntm-one-shot \
                              and https://github.com/kjunelee/MetaOptNet
"""
import numpy as np
import random, os,sys
import pickle as pkl
from PIL import Image
sys.path.append('/nfs/project/jianan/workspace/project/MCT_DFMN/mini_ImageNet/utils/generator')

img_size = 84

class miniImageNetGenerator(object):

    def __init__(self, data_file, nb_classes=5, nb_samples_per_class=10,
                  max_iter=None, xp=np):
        super(miniImageNetGenerator, self).__init__()
        self.data_file = data_file
        self.nb_classes = nb_classes
        self.nb_samples_per_class = nb_samples_per_class
        self.max_iter = max_iter
        self.xp = xp
        self.num_iter = 0
        self.data = self._load_data(self.data_file)
        self.image_size = 224
        # self.transform_train = T.Compose([
        #     T.Resize((self.image_size, self.image_size), interpolation=3),
        #     T.RandomCrop(self.image_size, padding=8),
        #     T.RandomHorizontalFlip(),
        #     T.ToTensor(),
        #     T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        #     # T.RandomErasing(0.5)
        # ])

    def _load_data(self, data_file):

        cat_container = sorted(os.listdir(data_file))
        cats2label = {cat: label for label, cat in enumerate(cat_container)}

        data = {}
        for key, value in cats2label.items():
            data[value] = []
        for cat in cat_container:
            for img_path in sorted(os.listdir(os.path.join(data_file, cat))):
                # if '.jpg' not in img_path:
                #     continue
                label = cats2label[cat]
                path = os.path.join(data_file, cat, img_path)
                with open(path, 'rb') as f:
                    with Image.open(f) as img:
                        img = img.convert('RGB')
                img = img.resize((img_size, img_size), Image.BILINEAR)
                data[label].append(np.array(img))

        return data

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

    def augment(self, img):
        npad = ((8, 8), (8, 8), (0, 0))
        img = np.pad(img, npad, 'constant', constant_values=(0.0))
        x = random.randint(0, 16)
        y = random.randint(0, 16)
        img = img[y:y + img_size, x:x + img_size]

        # random flipping
        flip_sign = random.randint(1,2)
        if flip_sign == 1:
            img = np.flip(img, 1)

        return img

    def sample(self, nb_classes, nb_samples_per_class):

        picture_list = sorted(set(self.data.keys()))
        pic_to_idx = {pic: i for i, pic in enumerate(picture_list)}
        sampled_characters = random.sample(self.data.keys(), nb_classes)

        labels_and_images = []
        for (k, char) in enumerate(sampled_characters):
            label = pic_to_idx[char]
            _imgs = self.data[char]
            _ind = random.sample(range(len(_imgs)), nb_samples_per_class)
            labels_and_images.extend([(label, self.xp.array(self.augment(_imgs[i]/np.float32(255)))) for i in _ind])
        arg_labels_and_images = []
        for i in range(self.nb_samples_per_class):
            for j in range(self.nb_classes):
                arg_labels_and_images.extend([labels_and_images[i+j*self.nb_samples_per_class]])

        labels, images = zip(*arg_labels_and_images)
        return images, labels
