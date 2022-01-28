This directory contains the following that is used to train PATE on Criteo dataset as seen in Section 5.2:

Most code is adapted from https://github.com/facebookresearch/label_dp_antipodes
1. ```train_teacher_ctr.py```
1. ```train_student_ctr.py```
1. ```lib/```

#############################################

## Training

### PATE
PATE model is trained in 2 stages. Below are example commands with hyperparameters we used.

#### Stage 1: Train teacher ensemble

```commandline
python train_teacher_ctr.py --n_teachers 100 --teacher_id 0
```

#### Stage 2: Aggregate votes and train student model

```commandline
python train_student_ctr.py --n_teachers 100 --n_samples 200 --tally_method sample --selection_method sample --result_noise 20 --mechanism lnmax
```
