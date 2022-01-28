# Code for LP-MST on MNIST and CIFAR10

## 1. Dependencies and Data Preparation
### 1.1 Dependencies
```torch, csv, tqdm, scipy, sklearn```
### 1.2 Data Preparation
The processed priors are
```
./pretrained_priors/{dataset}_{prior_type}_prior_probs.pt
./pretrained_priors/{dataset}_{prior_type}_prior_probs.pt
```
where `dataset` is `cifar10` or `mnist` and `prior_type` is `indomain` or `outdomain`.

We also provide the code to generate them from scratch: 
```
python get_priors.py --dataset {dataset} --prior_type {prior_type}
```

## 2. LP-MST Training
Run the vanila LP-1ST by
```
python train_{dataset}.py --eps {eps} --resume
```
and the LP-1ST with indomain/ourdomain prior by
```
python train_{dataset}.py --eps {eps} --prior {prior_type} --resume
```
where `dataset` is `cifar10` or `mnist`, `eps` is selected as `1.0, 0.1` for Table 1 in our paper and `prior_type` is `indomain` or `outdomain`.


# Reference
The training code is built on the original code of [Mixup](https://openreview.net/forum?id=r1Ddp1-Rb) ([https://github.com/facebookresearch/mixup-cifar10](https://github.com/facebookresearch/mixup-cifar10))