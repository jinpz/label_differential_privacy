# Code for LP-MST on MNIST and CIFAR10

## 1. Dependencies and Data Preparation
### 1.1 Dependencies
```torch, panda, tqdm, scipy, sklearn, catboost```
### 1.2 Data Preparation
The processed prior file is located as
```
./pretrained_priors/criteo_domain_prior_probs.pl
```

## 2. LP-MST Training
We provide the scripts for multiple variants. `eps` is selected as `8.0, 4.0, 2.0, 1.0, 0.1`.

### LP-1ST
```
python train_criteoctr.py --data_path ../data/ --eps {eps} --num_stage 1
```

### LP-1ST (domain prior)
```
python train_criteoctr.py --data_path ../data/ --eps {eps} --num_stage 1 --prior domain
```

### LP-1ST (noise correction)
```
python train_criteoctr.py --data_path ../data/ --eps {eps} --num_stage 1 --noise_correction
```

### LP-2ST
```
python train_criteoctr.py --data_path ../data/ --eps {eps} --num_stage 2
```