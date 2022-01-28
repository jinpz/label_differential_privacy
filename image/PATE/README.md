This directory contains the following that is used to train PATE-FM on MNIST and CIFAR10 as seen in Section 3.1:

Most code is adapted from https://github.com/facebookresearch/label_dp_antipodes
1. ```train_teacher.py```
1. ```aggregate_votes.py```
1. ```train_student.py```
1. ```lib/```

#############################################

## Training

### PATE
PATE model is trained in 3 stages. Below are example commands with hyperparameters we used.

#### Stage 1: Train teacher ensemble

##### CIFAR10
```commandline
python train_teacher.py --dataset cifar10 --n_teachers 200 --teacher-id 0 --epochs 40
```

##### MNIST
```commandline
python train_teacher.py --dataset mnist --n_teachers 200 --teacher-id 0 --epochs 40
```

#### Stage 2: Aggregate votes
Once all teachers are trained, we need to aggregate all votes into a single file

```commandline
python aggregate_votes.py --n_teachers 200
```

#### Stage 3: Train student

##### CIFAR10

```commandline
python train_student.py --dataset cifar10 --n_samples 500 --selection_noise 160 --result_noise 20 --noise.threshold 100 --epochs 200 --n_teachers 200
```

##### MNIST
```commandline
python train_student.py --dataset mnist --n_samples 750 --selection_noise 160 --result_noise 20 --noise.threshold 100 --epochs 200 --n_teachers 200
```