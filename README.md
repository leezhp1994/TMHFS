# A Transductive Multi-Head Model for Cross-Domain Few-Shot Learning

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
   Direct: command line kaggle datasets download -d plant-disease/data

   - **ChestX-Ray8**:
   Home: https://www.kaggle.com/nih-chest-xrays/data
   Direct: command line kaggle datasets download -d nih-chest-xrays/data

## Requiresments
   - Python 3.6
   - Pytorch 1.0.0
   
## Steps
   1. Download the source and target datasets  (miniImageNet, EuroSAT, ISIC2018, Plant Disease, ChestX-Ray8) using the above links.
   2. Change configuration file ./configs.py to reflect the correct paths to each dataset. Please see the existing example paths for information on which subfolders these paths should point to.
   3. Train base models on miniImageNet
```shell
python train.py --is_train True --transductive True --flip True --drop True --n_shot 5 --n_train_class 15 --gpu 0 
```
   4. Finetune & Test
   Finetune & Test without Data Augmentation
```shell
python train.py --is_train False --transductive True --flip True --drop True --n_shot 5 --n_train_class 15 --gpu 0 --test_data ISIC
```
   Finetune & Test with Data Augmentation
```shell
python train.py --is_train False --transductive True --flip True --drop True --n_shot 5 --n_train_class 15 --gpu 0 --test_data ISIC --test_aug 10
```
