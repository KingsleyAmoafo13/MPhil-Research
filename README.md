# EFFICIENT BACKBONE OPTIMIZATION FOR FEW-SHOT SEGMENTATION VIA MAGNITUDE-BASED WEIGHT PRUNING


Introduction
------------
This is the source code for our submitted paper [EFFICIENT BACKBONE OPTIMIZATION FOR FEW-SHOT SEGMENTATION VIA MAGNITUDE-BASED WEIGHT PRUNING], which is under review at IET Generation, Transmission & Distribution 2024.

Architecture
------------
The architecture.png denotes the architecture of the original SCCNet, the red regctangle illustrates the backnone replacement process
The architecture 2 denotes our pruning and finetuning process

### Installation

* Clone this repo

```
git clone https://github.com/KingsleyAmoafo13/MPhil-Research.git
```
* Install all dependenies

### Data Preparation

Download remote_sensing.tar.gz from (https://drive.google.com/drive/folders/1-URr9fX0v6_-Yo3B7St8UFNHiPWpXxnC?usp=sharing), 
Download VOC2012 from http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar


### Prune
```

Prune.py --backbone [specify backbone type] either vgg16, resnet50 or resnet101 

```
Example
The code below prunes resnet50

```
prune.py --backbone resnet50
```

### Train

```
python train.py  --max_steps 200000 --freeze True --datapath './remote_sensing/iSAID_patches' --img_size 256 --backbone resnet50 --fold 0 --benchmark isaid --lr 9e-4 --bsz 32 --logpath exp_name
```

The log and checkpoints are stored under directory 'logs'.ss

### Test

```
python test.py --datapath './remote_sensing/iSAID_patches' --img_size 256 --backbone resnet50 --fold 0 --benchmark isaid --bsz 64 --nshot 1 --load './logs/exp_name/best_model.pt'
```


### Acknowledgements

We borrow code from public projects [SDM](https://github.com/caoql98/SDM), [HSNet](https://github.com/juhongm999/hsnet), [dss](https://github.com/lukemelas/deep-spectral-segmentation) and [SCCNet] (https://github.com/linhanwang/SCCNet.git)

