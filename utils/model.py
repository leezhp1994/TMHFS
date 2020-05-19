import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 

from utils.backbone.resnet12 import ResNet12
from utils.backbone.conv256 import ConvNet as ConvNet_256
from utils.backbone.conv128 import ConvNet as ConvNet_128
from utils.backbone.conv64 import ConvNet as ConvNet_64

from utils.backbone.alex import AlexNet
from utils.backbone.wrn import wideResnet28_10
from utils.backbone.resnet import resnet18, resnet34, resnet12

import torch.nn.functional as F
import torchvision.transforms as transforms

out_dim = 512
img_size = 84


class Runner(object):
    def __init__(self, nb_class_train, nb_class_test,  input_size, n_shot, n_query,
                 backbone='ResNet-12', transductive_train=True, flip=True, drop=True):

        self.nb_class_train = nb_class_train
        self.nb_class_test = nb_class_test
        self.input_size = input_size
        self.n_shot = n_shot
        self.n_query = n_query
        self.is_transductive = transductive_train
        self.flip = flip if transductive_train else False
        self.drop = drop if transductive_train else False
        self.pool = nn.AvgPool2d(7)
        self.cls_list = []
        self.transductive_list = []
        self.classifier = nn.Linear(out_dim, 64)
        self.classifier.bias.data.fill_(0)
        # self.trans_loader = TransformLoader(84)
        self.classifier.cuda()

        # self.relu = nn.ReLU(inplace=inp)

        self.model = ResNet12(with_drop=drop)
        self.model.cuda()
        self.loss = nn.CrossEntropyLoss().cuda()


    def set_optimizer(self, learning_rate, weight_decay_rate):

        self.optimizer = optim.SGD([{'params': self.model.parameters(), 'weight_decay': weight_decay_rate},
                                    {'params': self.classifier.parameters(), 'weight_decay': weight_decay_rate}],
                                   lr = learning_rate, momentum=0.9, nesterov=True)

    def compute_accuracy(self, t_data, prob):
        t_est = torch.argmax(prob, dim=1)

        return (t_est == t_data)

    def make_protomap(self, support_set, nb_class):
        B, C, W, H = support_set.shape
        protomap = support_set.reshape(self.n_shot, nb_class, C, W, H)
        protomap = protomap.mean(dim=0)

        return protomap

    def make_input(self, images):

        images = np.stack(images)
        images = torch.Tensor(images).cuda()
        images = images.view(images.size(0), img_size, img_size, 3)
        images = images.permute(0, 3, 1, 2)

        return images

    def element_wise_scale(self, set):
        # print(set.size())
        x = self.model.conv1_ls(set)
        x = self.model.bn1_ls(x)
        x = self.model.relu(x)
        # print(x.size())
        x = x.reshape(x.size(0), -1)
        x = self.model.fc1_ls(x)
        x = F.softplus(x)

        return x

    def add_query(self, support_set, query_set, prob, nb_class):

        B, C, W, H = support_set.shape
        per_class = support_set.reshape(self.n_shot, nb_class, C, W, H)

        for i in range(nb_class):
            ith_prob = prob[:,i].reshape(prob.size(0), 1, 1, 1)
            ith_map = torch.cat((per_class[:,i], query_set*ith_prob), dim=0)
            ith_map = torch.sum(ith_map, dim=0, keepdim=True)/(ith_prob.sum()+self.n_shot)
            if i == 0: protomap = ith_map
            else: protomap = torch.cat((protomap, ith_map), dim=0)

        return protomap

    def norm_flatten(self, set):
        set = torch.flatten(set, start_dim=1)
        set = F.normalize(set, dim=1)

        return set

    def flip_key(self, images):
        self.model.eval()
        with torch.no_grad():
            flipped_key = self.model(torch.flip(images, dims=[3]))
            return flipped_key

    def train_transduction(self, original_key, flipped_key, nb_class, iters=1):

        if not self.is_transductive: iters = 0
        nb_key = 2 if self.flip else 1
        prob_list = []
        for iter in range(iters):
            prob_sum = 0
            for i in range(nb_key):
                if i != nb_key - 1: key_list = flipped_key
                else: key_list = original_key
                for idx, key in enumerate(key_list):
                    support_set = key[:nb_class * self.n_shot]
                    query_set = key[nb_class * self.n_shot:]
                    # Make Protomap
                    if iter == 0: protomap = self.make_protomap(support_set, nb_class)
                    else: protomap = self.add_query(support_set, query_set, prob_list[iter-1], nb_class)
                    # Element-wise length scaling
                    if idx == 0:
                        s_q = self.element_wise_scale(query_set)
                        s_p = self.element_wise_scale(protomap)
                    query_NF = self.norm_flatten(query_set) / s_q
                    proto_NF = self.norm_flatten(protomap) / s_p
                    # Calculate distance
                    distance = query_NF.unsqueeze(1) - proto_NF
                    distance = distance.pow(2).sum(dim=2)
                    prob = F.softmax(-distance, dim=1)
                    prob_sum += prob / (nb_key * len(key_list))
            prob_list.append(prob_sum)

        key = original_key[0]
        support_set = key[:nb_class * self.n_shot]
        query_set = key[nb_class * self.n_shot:]

        protomap = None
        if self.is_transductive:
            protomap = self.add_query(support_set, query_set, prob_list[-1], nb_class)
        elif not self.is_transductive:
            protomap = self.make_protomap(support_set, nb_class)

        s_p = self.element_wise_scale(protomap)
        scaled_proto = self.norm_flatten(protomap) / s_p

        return scaled_proto

    def train(self, images, labels):
        self.classifier.train()
        nb_class = self.nb_class_train
        images = self.make_input(images)
        labels_DC = torch.tensor(labels, dtype=torch.long).cuda()

        flipped_key = self.flip_key(images) if self.flip else None

        self.model.train()
        original_key = self.model(images)
        key = original_key[0]
        # print(key.size())
        # pixel-wise classification
        key_DC = key[nb_class * self.n_shot:]
        key_DC = key_DC.reshape(key_DC.size(0), key_DC.size(1), -1)
        key_DC = key_DC.permute(0, 2, 1)
        prototype = self.model.weight.weight

        loss_dense = 0
        distance = key_DC.unsqueeze(2) - prototype
        distance = distance.pow(2).sum(dim=3)
        for i in range(distance.size(1)):
            loss_dense += self.loss(-distance[:,i], labels_DC[nb_class * self.n_shot:])/distance.size(1)

        # instance-wise classification
        labels_IC = tuple([i for i in range(nb_class)]) * (self.n_query)
        labels_IC = torch.tensor(labels_IC, dtype=torch.long).cuda()
        #make prototype
        scaled_proto = self.train_transduction(original_key, flipped_key, nb_class, iters=1)
        query_set = key[nb_class * self.n_shot:]
        s_q = self.element_wise_scale(query_set)
        scaled_query = self.norm_flatten(query_set) / s_q

        distance = scaled_query.unsqueeze(1) - scaled_proto
        distance = distance.pow(2).sum(dim=2)
        loss_instance = self.loss(-distance, labels_IC)

        loss = 0
        loss += 1 * loss_dense
        loss += 1/5 * loss_instance
        # loss += loss_instance

        x = self.pool(original_key[0])
        x = x.view(x.size(0), -1)

        pred = self.classifier(x)
        loss_cls = self.loss(pred, labels_DC)
        loss = 0.5*loss + 0.5*loss_cls
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.data

    def evaluate(self, images, labels):

        nb_class = self.nb_class_test
        for ii in range(len(images)):
            images[ii] = self.make_input(images[ii])
        labels = torch.tensor(labels, dtype=torch.long).cuda()
        scores_cls, prob_cls= self.train_cls(images, labels)
        self.model.eval()
        with torch.no_grad():
            flipped_key = [self.model(torch.flip(image_temp, dims=[3])) for image_temp in images]
            original_key = [self.model(image_temp) for image_temp in images]
            iteration= 11 if self.is_transductive else 1
            nb_key = 2 if self.flip else 1
            prob_list = []
            for iter in range(iteration):
                prob_sum = 0
                for i in range(nb_key):
                    if i != nb_key - 1:  key_list = flipped_key
                    else: key_list = original_key
                    for ii in range(len(key_list)):
                        for idx in range(len(key_list[0])):
                            support_set = key_list[ii][idx][:nb_class * self.n_shot]
                            query_set = key_list[ii][idx][nb_class * self.n_shot:]
                            # Make Protomap
                            if iter == 0: protomap = self.make_protomap(support_set, nb_class)
                            else: protomap = self.add_query(support_set, query_set, prob_list[iter-1], nb_class)
                            if idx == 0:
                                s_q = self.element_wise_scale(query_set)
                                s_p = self.element_wise_scale(protomap)
                            # Element-wise Scaling
                            query_NF = self.norm_flatten(query_set) / s_q
                            proto_NF = self.norm_flatten(protomap) / s_p
                            # Calculate Distance
                            distance = query_NF.unsqueeze(1) - proto_NF
                            distance = distance.pow(2).sum(dim=2)
                            prob = F.softmax(-distance, dim=1)
                            prob_sum += prob
                prob_list.append(prob_sum / (len(key_list[0]) * nb_key * len(key_list)))
            prob = prob_list[-1]

            acc = self.compute_accuracy(labels[nb_class * self.n_shot:], prob)

            prob = prob.data.cpu().numpy()
            return acc, prob, labels[nb_class*self.n_shot:], scores_cls, prob_cls
            # return acc, prob, labels[nb_class * self.n_shot:]

    def train_cls(self, images_all, labels_all):
        self.model.train()
        
        test_imgs = []
        for ii in range(len(images_all)):
            images_temp = images_all[ii][:self.nb_class_test * self.n_shot]
            labels_temp = labels_all[:self.nb_class_test * self.n_shot]

            test_imgs_temp = images_all[ii][self.nb_class_test * self.n_shot:]
            test_labels_temp = labels_all[self.nb_class_test * self.n_shot:]
            if ii == 0:
                images = images_temp
                labels = labels_temp
            else:
                images = torch.cat([images,images_temp],0)
                labels = torch.cat([labels,labels_temp],0)
            test_imgs.append(test_imgs_temp)
            test_labels=test_labels_temp

        classifier = nn.Linear(out_dim, self.nb_class_test)
        classifier.cuda()
        classifier.train()
        classifier_opt = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9, dampening=0.9,
                                         weight_decay=0.001)
        delta_opt = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=0.01)

        batch_size = 4
        total_epoch = 100
        for epoch in range(total_epoch):
            rand_id = np.random.permutation(images.size()[0])
            for j in range(0, images.size(0), batch_size):
                # print(images.size(),labels)
                classifier_opt.zero_grad()
                delta_opt.zero_grad()
                selected_id = torch.from_numpy(rand_id[j: min(j + batch_size, images.size()[0])]).cuda()
                _images = images[selected_id]
                _labels = labels[selected_id]
                x = self.model(_images.cuda())
                x = self.pool(x[0])
                x = x.view(x.size(0), -1)
                pred = classifier(x)
                loss_cls = self.loss(pred, _labels)
                loss_cls.backward()
                classifier_opt.step()
                delta_opt.step()
           
        self.model.eval()
        classifier.eval()
        
        scores=[]
        for ii in range(len(test_imgs)):
            output = self.model(test_imgs[ii])
            output = self.pool(output[0])
            output = output.view(output.size(0), -1)
            score = classifier(output)
            scores.append(F.softmax(score, dim=1))
        scores = sum(scores)/len(scores)
        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind[:, 0] == test_labels.cpu().numpy())
        correct_this, count_this = float(top1_correct), test_labels.size()[0]
        return correct_this / count_this * 100, [topk_scores.data.cpu().numpy(),topk_labels.data.cpu().numpy()]

    