#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree.
from __future__ import print_function

import argparse
import csv
import os
from copy import deepcopy
import pickle
os.environ['KMP_WARNINGS'] = '0'

import numpy as np
import pandas as pd
import torch
import scipy
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from catboost import CatBoostClassifier
import math
from tqdm import tqdm
from logger import set_logger

parser = argparse.ArgumentParser(description='Criteo Training')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--data_path', default="../data/", type=str,
                    help='path to load data')
parser.add_argument('--prior', default=None, type=str,
                    help='prior probabilities from pretrained features')
parser.add_argument('--eps', default=1., type=float,
                    help='paramter of label DP')
parser.add_argument('--num_stage', default=1, type=int,
                    help='total stages to run')
parser.add_argument('--noise_correction', '-r', action='store_true',
                    help='make a post noise correction during the evaluation')

args = parser.parse_args()
logger = set_logger("")

# Data
logger.info(args)
logger.info('==> Preparing data..')
continue_var = ['I' + str(i) for i in range(1, 14)]
cat_features = ['C' + str(i) for i in range(1,27)]
trainset_path = os.path.join(args.data_path, "raw_dataset_1M_train.csv")
testset_path = os.path.join(args.data_path, "raw_dataset_1M_test.csv")
train = pd.read_csv(trainset_path)
test = pd.read_csv(testset_path)

np.random.seed(args.seed)
y_train = train[['Label']]
x_train = train.drop(['Label'], axis=1)
y_test = test[['Label']]
x_test = test.drop(['Label'], axis=1)
y_train_origin = deepcopy(y_train).to_numpy().squeeze()
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.20, stratify=y_test, random_state=256)
global_trans_mat = np.asarray([[math.exp(args.eps), 1], [1, math.exp(args.eps)]]) / (math.exp(args.eps) + 1)

y_test = y_test.to_numpy().squeeze()

def RRWithPrior_batch(labels, prior_probs, eps=1.0):
    num_data, num_classes = prior_probs.size()
    
    sorted_prior_probs, sorted_ids = torch.sort(prior_probs, descending=True)
    weight = sorted_prior_probs.cumsum(dim=1)
    weight = weight * math.exp(eps) / (math.exp(eps) + torch.arange(num_classes).float())
    opt_k = torch.argmax(weight, dim=1)

    rank = torch.zeros(sorted_ids.size())
    rank[torch.arange(num_data).repeat(num_classes, 1).t().contiguous().view(-1).long(), sorted_ids.view(-1)] = torch.arange(num_classes).repeat(num_data).float()
    rank_labels = rank[torch.arange(num_data).long(), labels]
    if_top_opt_k = (rank_labels <= opt_k).float()

    priv_labels = torch.zeros(num_data)
    for i in tqdm(range(num_data)):
        probs_sample = torch.zeros(num_classes)
        if if_top_opt_k[i]:
            probs_sample[sorted_ids[i, :(opt_k[i]+1)]] =  1 / (math.exp(eps) + opt_k[i])
            probs_sample[labels[i]] =  math.exp(eps) / (math.exp(eps) + opt_k[i])
        else:
            probs_sample[sorted_ids[i, :(opt_k[i]+1)]] = 1.0 / (opt_k[i] + 1)
        priv_labels[i] = torch.multinomial(probs_sample, 1, replacement=True)
   
    return priv_labels.long().numpy()

torch.manual_seed(args.seed)
randperm = torch.randperm(x_train.shape[0])

logger.info("==> Training..")
for t in range(args.num_stage):
    save_pre_name = f"criteo_{args.eps}_{t}_{args.num_stage}_{args.noise_correction}"
    eps = args.eps
    if args.prior is not None:
        save_pre_name = save_pre_name + "_" + args.prior
    save_name = f"./checkpoint/{save_pre_name}_{args.seed}"
    checkpoint = {}
        
    x_train_t, y_train_t = deepcopy(x_train), deepcopy(y_train)
    indices_t = randperm[int(float(t)*x_train.shape[0]/args.num_stage):int(float(t+1)*x_train.shape[0]/args.num_stage)].numpy()
    x_train_t, y_train_t = x_train_t.iloc[indices_t], y_train_t.iloc[indices_t]
    y_train_t = y_train_t.to_numpy().squeeze()
    if t == 0:
        if args.prior is None:
            prior_probs = torch.ones([x_train_t.shape[0], 2]) / 2.0
        else:
            eps -= 0.05
            prior_probs = torch.load(f"pretrained_priors/criteo_{args.prior}_prior_probs.pl")[indices_t]
    else:
        last_t = t - 1
        last_save_pre_name = f"criteoctr_{args.eps}_{last_t}_{args.num_stage}"
        last_save_name = f"./checkpoint/{last_save_pre_name}_{args.seed}"
        with (open(last_save_name, "rb")) as openfile:
            checkpoint = pickle.load(openfile)
            prior_probs = torch.from_numpy(checkpoint["y_train_prob"][indices_t]).float()
    priv_labels = RRWithPrior_batch(y_train_t, prior_probs, eps=eps)
    y_train.iloc[indices_t] = np.expand_dims(priv_labels, axis=1)
    
    cum_x_train_t, cum_y_train_t = deepcopy(x_train), deepcopy(y_train)
    
    cum_indices_t = randperm[:int(float(t+1)*cum_x_train_t.shape[0]/args.num_stage)]
    cum_x_train_t, cum_y_train_t = cum_x_train_t.iloc[cum_indices_t], cum_y_train_t.iloc[cum_indices_t]
    
    cat_features = [col for col in train.columns if "C" in col]
    model = CatBoostClassifier(
        iterations=100,
        learning_rate=0.4,
        task_type='GPU',
        loss_function='Logloss',
         depth=8,
    )

    fit_model = model.fit(
        cum_x_train_t, cum_y_train_t,
        eval_set=(x_val, y_val),
        cat_features=cat_features,
        verbose=10
    )

    y_test_prob = model.predict(x_test, 
                       prediction_type='Probability',
                       ntree_end=model.get_best_iteration(), 
                       thread_count=-1,
                       verbose=None)

    y_val_prob = model.predict(x_val, 
                           prediction_type='Probability',
                           ntree_start=0,
                           ntree_end=model.get_best_iteration(), 
                           thread_count=-1,
                           verbose=None)

    y_train_prob = model.predict(x_train, 
                           prediction_type='Probability',
                           ntree_start=0,
                           ntree_end=model.get_best_iteration(), 
                           thread_count=-1,
                           verbose=None)
        
    logger.info("==> Evaluating..")

    def weighted_acc(y, y_pred, class_weights):
        sample_weight = np.ones(len(y))
        sample_weight[y == 0] = class_weights[0]
        sample_weight[y == 1] = class_weights[1]
        acc = (y == y_pred).astype(np.float)
        return (acc * sample_weight).mean()

    def optimal_pred(y_probs, class_weights):
        class_weights = np.asarray(class_weights).squeeze()
        return (y_probs * class_weights).argmax(1)
    

    class_weights = [y_train_origin.mean(), 1-y_train_origin.mean()]
    B = (y_train_origin.mean() * class_weights[1] + (1 - y_train_origin.mean()) * class_weights[0]).item()
    
    if not args.noise_correction:
        y_train_opt_pred = optimal_pred(y_train_prob, class_weights)
        checkpoint["EAU_train"] = (weighted_acc(y_train_origin,y_train_opt_pred, class_weights)) / B

        y_test_opt_pred = optimal_pred(y_test_prob, class_weights)
        checkpoint["EAU_test"] = (weighted_acc(y_test, y_test_opt_pred, class_weights)) / B
        
        checkpoint["log_loss"] = log_loss(y_test, y_test_prob[:, 1])
    else:
        y_train_prob_correct = np.clip(np.matmul(y_train_prob, np.linalg.inv(global_trans_mat)), 0, 1)
        y_train_opt_pred = optimal_pred(y_train_prob_correct, class_weights)
        checkpoint["EAU_train"] = (weighted_acc(y_train_origin,y_train_opt_pred, class_weights)) / B

        y_test_prob_correct = np.clip(np.matmul(y_test_prob, np.linalg.inv(global_trans_mat)), 0, 1)
        y_test_opt_pred = optimal_pred(y_test_prob_correct, class_weights)
        checkpoint["EAU_test"] = (weighted_acc(y_test, y_test_opt_pred, class_weights)) / B
    
        checkpoint["log_loss"] = log_loss(y_test, y_test_prob_correct[:, 1])

    checkpoint["y_train_prob"] = y_train_prob

    logger.info("==> Saving..")
    logger.info(f"EAU on train set is {checkpoint['EAU_train']}")
    logger.info(f"EAU on test set is {checkpoint['EAU_test']}")
    logger.info(f"Log loss on test set is {checkpoint['log_loss']}")

    if not os.path.exists("./checkpoint"):
        os.makedirs("./checkpoint")
    with open(save_name, 'wb') as handle:
        pickle.dump(checkpoint, handle, protocol=pickle.HIGHEST_PROTOCOL)

