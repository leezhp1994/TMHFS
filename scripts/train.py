import os
import sys
sys.path.append('../')
import argparse
import numpy as np

import torch

from utils.generator.generators_train import miniImageNetGenerator as train_loader
from utils.model import Runner
from utils import configs

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--is_train', type=str2bool, default=True,
                        help='Choice train or test.')
    parser.add_argument('--n_folder', type=int, default=0,
                        help='Number of folder.')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device number.')
    parser.add_argument('--backbone', type=str, default='ResNet-12',
                        help='Choice backbone such as ConvNet-64, ConvNet-128, ConvNet-256 and ResNet-12.')
    parser.add_argument('--initial_lr', type=float, default=1e-1,
                        help='Initial learning rate.')
    parser.add_argument('--first_decay', type=int, default=25000,
                        help='First decay step.')
    parser.add_argument('--second_decay', type=int, default=35000,
                        help='Second decay step.')

    parser.add_argument('--transductive', type=str2bool, default=True,
                        help='Whether to use transductive training or not.')
    parser.add_argument('--flip', type=str2bool, default=True,
                        help='Whether to inject data uncertainty.')
    parser.add_argument('--drop', type=str2bool, default=True,
                        help='Whether to inject model uncertainty.')

    parser.add_argument('--n_shot', type=int, default=5,
                        help='Number of support set per class in train.')
    parser.add_argument('--n_query', type=int, default=8,
                        help='Number of queries per class in train.')
    parser.add_argument('--n_test_query', type=int, default=15,
                        help='Number of queries per class in test.')
    parser.add_argument('--n_train_class', type=int, default=15,
                        help='Number of way for training episode.')
    parser.add_argument('--n_test_class', type=int, default=5,
                        help='Number of way for test episode.')
    parser.add_argument('--save', type=str, default='default',
                        help='Choice backbone such as ConvNet-64, ConvNet-128, ConvNet-256 and ResNet-12.')
    parser.add_argument('--test_data', type=str, default='ISIC',
                        help='Name of test dataset.')
    parser.add_argument('--test_aug', type=int, default=1,
                        help='Number of data augmentation methods.')

    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    #######################
    folder_num = args.n_folder

    # optimizer setting
    max_iter = 50000
    lrstep2 = args.second_decay
    lrstep1 = args.first_decay
    initial_lr = args.initial_lr

    # train episode setting
    n_shot=args.n_shot
    n_query=args.n_query
    nb_class_train = args.n_train_class

    # test episode setting
    n_query_test = args.n_test_query
    nb_class_test=args.n_test_class

    train_path = configs.imagenet_path

    #save_path
    save_path = 'save/baseline_' + str(args.save) + str(folder_num).zfill(3)
    filename_5shot=save_path + '/miniImageNet_ResNet12'
    filename_5shot_last= save_path + '/miniImageNet_ResNet12_last'

    # set up training
    # ------------------
    model = Runner(nb_class_train=nb_class_train, nb_class_test=nb_class_test, input_size=3*84*84,
                   n_shot=n_shot, n_query=n_query, backbone=args.backbone,
                   transductive_train=args.transductive, flip=args.flip, drop=args.drop)
    model.set_optimizer(learning_rate=initial_lr, weight_decay_rate=5e-4)


    if not os.path.exists(save_path):
        os.makedirs(save_path)

    loss_h=[]
    accuracy_h_val=[]
    accuracy_h_test=[]

    acc_best=0
    epoch_best=0
    # start training
    # ----------------
    if args.is_train:
        train_generator = train_loader(data_file=train_path, nb_classes=nb_class_train,
                                       nb_samples_per_class=n_shot + n_query, max_iter=max_iter)
        for t, (images, labels) in train_generator:
            # train
            loss = model.train(images, labels)
            # logging
            loss_h.extend([loss.tolist()])
            if (t % 100 == 0):
                print("Episode: %d, Train Loss: %f "%(t, loss))
                torch.save(model.model.state_dict(), filename_5shot_last)

            if (t != 0) & (t % lrstep1 == 0):
                for param_group in model.optimizer.param_groups:
                    param_group['lr'] *= 0.06
                    print('-------Decay Learning Rate to ', param_group['lr'], '------')
            if (t != 0) & (t % lrstep2 == 0):
                for param_group in model.optimizer.param_groups:
                    param_group['lr'] *= 0.2
                    print('-------Decay Learning Rate to ', param_group['lr'], '------')

    accuracy_h5=[]
    total_acc = []
    checkpoint = torch.load(filename_5shot)
    state_keys = list(checkpoint.keys())
    for _, key in enumerate(state_keys):
        if "classifier." in key:
            checkpoint.pop(key)

    # print(checkpoint.keys())

    print('Evaluating the best {}-shot model for {}...'.format(n_shot,args.test_data))
    test_data = args.test_data
    aug_num = max(args.test_aug,1)
    print('aug_num:',args.test_data)
    if 'cropdiseases' in test_data.lower():
        save_name = 'CropDiseases'
        test_path = configs.CropDisease_path
        from utils.generator.generators_test import miniImageNetGenerator as test_loader    
    elif 'isic' in test_data.lower():
        save_name = 'ISIC'
        test_path = configs.ISIC_path
        from utils.generator.generators_isic_test import miniImageNetGenerator as test_loader
    elif 'eurosat' in test_data.lower():
        save_name = 'EuroSAT'
        test_path = configs.EuroSAT_path
        from utils.generator.generators_eurosat_cropdiseases_test import miniImageNetGenerator as test_loader
    elif 'chestx' in test_data.lower():
        save_name = 'chestX'
        test_path = configs.ChestX_path
        from utils.generator.generators_chestX_test import miniImageNetGenerator as test_loader
    else:
        raise ValueError('Unknown test data')
        
    for i in range(1):
        test_generator = test_loader(data_file=test_path, nb_classes=nb_class_test,
                                     nb_samples_per_class=n_shot+n_query_test, max_iter=600,aug_num=aug_num)
        scores=[]
        acc_all = []
        prob_cls_list = []
        prob_traductive_list = []
        for j, (images, labels) in test_generator:
            model.model.load_state_dict(checkpoint)
            acc, prob, label, scores_cls, prob_cls = model.evaluate(images, labels)
            # acc, prob, label = model.evaluate(images, labels)
            score = acc.data.cpu().numpy()
            scores.append(score)
            print('Episode %3d : accuracy %4f'%(j, np.mean(score) * 100))
            total_acc.append(np.mean(score) * 100)
            # acc_all.append(scores_cls)
        accuracy_t=100*np.mean(np.array(scores))
        accuracy_h5.extend([accuracy_t.tolist()])
#        print(('600 episodes with 15-query accuracy: {}-shot ={:.2f}%').format(n_shot, accuracy_t))
        del(test_generator)
        del(acc)
        del(accuracy_t)
#
#        acc_all = np.asarray(acc_all)
#        acc_mean = np.mean(acc_all)
#        acc_std = np.std(acc_all)
#        print('Test Acc = %4.2f%% +- %4.2f%%' % ( acc_mean, 1.96 * acc_std / np.sqrt(600)))

    stds = np.std(total_acc, axis=0)
    ci95 = 1.96 * stds / np.sqrt(len(total_acc))

    print(('Accuracy_test {}-shot ={:.2f}({:.2f})').format(n_shot, np.mean(accuracy_h5), ci95))

