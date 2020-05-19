# A Transductive Multi-Head Model for Cross-Domain Few-Shot Learning

## Abstract
In this paper, we present a new method, Transductive Multi-Head Few-Shot learning (TMHFS), to address the Cross-Domain Few-Shot Learning (CD-FSL) challenge. The TMHFS method extends the Meta-Confidence Transduction (MCT) and Dense Feature Matching Networks (DFMN) method [2] by introducing a new prediction head, i.e, an instance-wise global classification network based on semantic information, after the common feature embedding network. We train the embedding network with the multiple-heads, i.e,, the MCT loss, the DFMN loss and the semantic classifier loss, simultaneously in the source domain. For the few-shot learning in the target domain, we first perform fine-tuning on the embedding network with only the semantic global classifier and the support instances, and then use the MCT part to predict labels of the query set with the fine-tuned embedding network. Moreover, we further exploit data augmentation techniques during the fine-tuning and test stages to improve the prediction performance. The experimental results demonstrate that the proposed methods reatly outperform the strong baseline, fine-tuning, on four different target domains.

## Datasets  
The following datasets are used for evaluation in this challenge:

### Source domain:  
- **miniImageNet**  
   https://drive.google.com/file/d/1uxpnJ3Pmmwl-6779qiVJ5JpWwOGl48xt/view

### Target domain:  
   - **EuroSAT**:  
   Home: http://madm.dfki.de/downloads  
   Direct: http://madm.dfki.de/files/sentinel/EuroSAT.zip

   - **ISIC2018**:  
   Home: http://challenge2018.isic-archive.com  
   Direct (must login): https://challenge.kitware.com/#phase/5abcbc6f56357d0139260e66

   - **Plant Disease**:  
   Home: https://www.kaggle.com/saroz014/plant-disease/  
   Direct: command line `kaggle datasets download -d plant-disease/data`

   - **ChestX-Ray8**:  
   Home: https://www.kaggle.com/nih-chest-xrays/data  
   Direct: command line `kaggle datasets download -d nih-chest-xrays/data`

## Requiresments
   - Python 3.6
   - Pytorch 1.0.0
   
## Steps
1. Download the source and target datasets  (miniImageNet, EuroSAT, ISIC2018, Plant Disease, ChestX-Ray8) using the above links.
2. Change configuration file ./configs.py to reflect the correct paths to each dataset. Please see the existing example paths for information on which subfolders these paths should point to.
3. Train base models on miniImageNet
   ```shell
   python train.py --is_train True --transductive True --flip True --drop True --n_query 15 -n_shot 5 --n_train_class 15 --gpu 0 
   ```
4. Finetune & Test  
   - *Finetune & Test without Data Augmentation*
   ```shell
   python train.py --is_train False --transductive True --flip True --drop True --n_test_query 15 --n_shot 5 --n_test_class 5 --gpu 0 --test_data ISIC
   ```
   - *Finetune & Test with Data Augmentation*  
   ```shell
   python train.py --is_train False --transductive True --flip True --drop True --n_test_query 15 --n_shot 5 --n_test_class 5 --gpu 0 --test_data ISIC --test_aug 10
   ```
5. If you want to train and test your own methods, you should knoe the means of the following arguments:  
   - *test_data: name of the corresponding dataset (EuroSAT, ISIC, CropDiseases, ChestX)*  
   - *n_train_class: number of way for training episode*  
   - *n_test_class: number of way for test episode*  
   - *n_query: number of queries per class in train*  
   - *n_test_query: number of queries per class in test*  
   - *n_shot: number of support set per class*
   - *test_aug: number of data augmentation methods, default 1 means no data augmentation*
   
## Acknowledgments
This code is based on the implementation of [MCT_DFMN](https://github.com/seongmin-kye/MCT_DFMN "MCT_DFMN").

## References
[1] Y. Guo, N. C. F. Codella, L. Karlinsky, J. R. Smith, T. Rosing, and R. Feris. A new benchmark for evaluation of cross-domain few-shot learning. 2019.  
[2] S. M. Kye, H. B. Lee, H. Kim, and S. J. Hwang. Transductive few-shot learning with meta-learned confidence. arXiv preprint arXiv:2002.12017, 2020.
