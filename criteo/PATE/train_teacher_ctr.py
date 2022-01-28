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
from lib.utils import weighted_acc, optimal_pred

os.environ['KMP_WARNINGS'] = '0'

import numpy as np
import pandas as pd
import scipy
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from catboost import CatBoostClassifier
import xgboost as xgb
from torch.utils.data import Dataset, DataLoader
import math
import torch
from tqdm import tqdm
import random

parser = argparse.ArgumentParser(description='PyTorch mnist Training')
parser.add_argument('--algo', default="catboost", type=str,
                    help='algorithm')
parser.add_argument('--seed', default=47, type=int, help='random seed')
parser.add_argument('--dataset', default="raw_dataset_1M", type=str,
                    help='dataset name')
parser.add_argument('--data_path', default="../data/", type=str,
                    help='path to load data')
parser.add_argument('--save_path', default="results/PATE_ctr/", type=str,
                    help='base path to save checkpoint; algo_n_teachers_seed')
parser.add_argument('--n_teachers', default=100, type=int,
                    help='# of teachers')
parser.add_argument('--teacher_id', type=int,
                    help='0 to n - 1 where n is total # of teachers')
args = parser.parse_args()


def partition_dataset_indices(dataset_len, n_teachers, teacher_id, seed=None):
    random.seed(seed)

    teacher_data_size = dataset_len // n_teachers
    indices = list(range(dataset_len))
    random.shuffle(indices)

    result = indices[
             teacher_id * teacher_data_size: (teacher_id + 1) * teacher_data_size
             ]

    logging.info(
        f"Teacher {teacher_id} processing {len(result)} samples. "
        f"First index: {indices[0]}, last index: {indices[-1]}. "
        f"Range: [{teacher_id * teacher_data_size}:{(teacher_id + 1) * teacher_data_size}]"
    )

    return result


checkpoint_dir = args.save_path + 'teacher_{0}/'.format(args.teacher_id)

# Data
print('==> Preparing data..')
if args.algo.startswith("logistic_regression"):
    trainset_path = os.path.join(args.data_path, args.dataset + "_10000_train.npz")
    testset_path = os.path.join(args.data_path, args.dataset + "_10000_test.npz")

    train = scipy.sparse.load_npz(trainset_path)
    test = scipy.sparse.load_npz(testset_path)
    x_train_original, y_train_original = train[:, 1:], train[:, 0]
    x_test, y_test = test[:, 1:], test[:, 0]
    y_train_original, y_test = np.asarray(y_train_original.todense()).squeeze(), np.asarray(y_test.todense()).squeeze()
else:
    continue_var = ['I' + str(i) for i in range(1, 14)]
    cat_features = ['C' + str(i) for i in range(1, 27)]
    trainset_path = os.path.join(args.data_path, args.dataset + "_train.csv")
    testset_path = os.path.join(args.data_path, args.dataset + "_test.csv")
    train = pd.read_csv(trainset_path)
    test = pd.read_csv(testset_path)

    y_train_original = train[['Label']]
    x_train_original = train.drop(['Label'], axis=1)
    y_test = test[['Label']]
    x_test = test.drop(['Label'], axis=1)

np.random.seed(args.seed)

labeled_indices = partition_dataset_indices(
    dataset_len=len(x_train_original),
    n_teachers=args.n_teachers,
    teacher_id=args.teacher_id,
    seed=args.seed,
)

x_train = x_train_original.iloc[labeled_indices]
y_train = y_train_original.iloc[labeled_indices]

x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.20, stratify=y_test, random_state=256)

print("==> Training..")
checkpoint = {}
if args.algo == "catboost":
    cat_features = [col for col in train.columns if "C" in col]
    model = CatBoostClassifier(
        iterations=100,
        learning_rate=0.4,
        task_type='GPU',
        loss_function='Logloss',
        depth=8,
    )

    fit_model = model.fit(
        x_train, y_train,
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
    y_train_original_prob = model.predict(x_train_original,
                                          prediction_type='Probability',
                                          ntree_start=0,
                                          ntree_end=model.get_best_iteration(),
                                          thread_count=-1,
                                          verbose=None)
elif args.algo == "xgboost":
    num_round = 100
    dtrain = xgb.DMatrix(x_train, label=y_train, enable_categorical=True)
    dval = xgb.DMatrix(x_val, label=y_val, enable_categorical=True)
    dtest = xgb.DMatrix(x_test, label=y_test, enable_categorical=True)
    dtrain_original = xgb.DMatrix(x_train_original, label=y_train_original, enable_categorical=True)
    evallist = [(dval, 'eval'), (dtrain, 'train')]
    bst = xgb.train({'eval_metric': 'logloss'}, dtrain, num_round, evallist)


    def prob_to_complete_prob(y_prob):
        y_prob = np.expand_dims(y_prob, axis=1)
        y_prob = np.concatenate([1 - y_prob, y_prob], axis=1)
        return y_prob


    y_train_prob = prob_to_complete_prob(bst.predict(dtrain, iteration_range=(0, bst.best_iteration)))
    y_val_prob = prob_to_complete_prob(bst.predict(dval, iteration_range=(0, bst.best_iteration)))
    y_test_prob = prob_to_complete_prob(bst.predict(dtest, iteration_range=(0, bst.best_iteration)))
    y_train_original_prob = prob_to_complete_prob(bst.predict(dtrain_original, iteration_range=(0, bst.best_iteration)))
elif args.algo.startswith("logistic_regression"):
    class CTR(Dataset):
        def __init__(self, sparse_x, y, transform=None):
            self.sparse_x = sparse_x
            self.y = torch.from_numpy(y).long()

        def __len__(self):
            return len(self.y)

        def __getitem__(self, idx):
            return torch.from_numpy(self.sparse_x[idx].toarray()).float(), self.y[idx]


    batch_size = 256
    trainset = CTR(x_train, y_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    testset = CTR(x_test, y_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)
    valset = CTR(x_val, y_val)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=4)
    trainset_original = CTR(x_train_original, y_train_original)
    train_original_loader = torch.utils.data.DataLoader(trainset_original, batch_size=batch_size, shuffle=False,
                                                        num_workers=4)


    def train(net, trainloader, optimizer, device, lam=0):
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in tqdm(enumerate(trainloader), total=len(trainloader)):
            inputs, targets = inputs.to(device).squeeze(), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets).sum()
            train_loss += loss.item()
            loss /= len(targets)
            if lam > 0:
                for param in net.parameters():
                    loss += lam * (param ** 2).sum()
            loss.backward()
            optimizer.step()
            total += targets.size(0)
        print('==>>> train loss: {:.6f}'.format(train_loss / total))


    def test(net, testloader, device):
        test_loss = 0
        correct = 0
        total = 0
        probs = torch.ones([len(testloader.dataset), 2])
        net.eval()
        with torch.no_grad():
            for batch_idx, (inputs, targets) in tqdm(enumerate(testloader), total=len(testloader)):
                inputs, targets = inputs.to(device).squeeze(), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets).sum()
                probs[batch_size * batch_idx: batch_size * (batch_idx + 1)] = torch.nn.Softmax()(outputs)
                test_loss += loss.item()
                total += targets.size(0)
            print('==>>> test loss: {:.6f}'.format(test_loss / total))
        return test_loss / total, probs


    model = args.algo.split("_")[-2]
    epochs = 50
    device = "cuda"
    num_classes = 2
    num_feat = x_train.shape[1]
    if model == "linear":
        net = torch.nn.Linear(num_feat, num_classes).to(device)
        lam = float(args.algo.split("_")[-1])
    elif model == "MLP":
        lam = 0
        net = MLP(input_size=num_feat, hidden_sizes=[100], num_class=num_classes).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    best_loss = 1e6
    best_net = None
    for epoch in range(epochs):
        print("epoch: {}".format(epoch))
        train(net, trainloader, optimizer, device, lam)
        loss, _ = test(net, valloader, device)
        if loss < best_loss:
            best_loss = loss
            best_net = deepcopy(net)
        if epoch == int(epochs * 2 / 3):
            for g in optimizer.param_groups:
                g['lr'] *= 0.1
    checkpoint = {f"net": net.state_dict(), f"loss": loss, "best_net": best_net.state_dict(), "best_loss": best_loss}
    _, y_train_prob = test(best_net, trainloader, device)
    _, y_val_prob = test(best_net, valloader, device)
    _, y_test_prob = test(best_net, testloader, device)
    _, y_train_original_prob = test(best_net, train_original_loader, device)
    y_train_prob = y_train_prob.numpy()
    y_val_prob = y_val_prob.numpy()
    y_test_prob = y_test_prob.numpy()
    y_train_original_prob = y_train_original_prob.numpy()
else:
    raise NotImplementedError

print("==> Evaluating..")

checkpoint["log_loss"] = log_loss(y_test, y_test_prob[:, 1])

class_weights = [y_train.mean(), 1 - y_train.mean()]
B = (y_train.mean() * class_weights[1] + (1 - y_train.mean()) * class_weights[0]).item()

if args.algo.startswith("logistic_regression"):
    y_train_opt_pred = optimal_pred(y_train_prob, class_weights)
    checkpoint["advantage_train"] = (weighted_acc(y_train, y_train_opt_pred, class_weights)) / B
    y_test_opt_pred = optimal_pred(y_test_prob, class_weights)
    checkpoint["advantage_test"] = (weighted_acc(y_test, y_test_opt_pred, class_weights)) / B
else:
    y_train_opt_pred = optimal_pred(y_train_prob, class_weights)
    checkpoint["advantage_train"] = (weighted_acc(y_train.to_numpy().squeeze(), y_train_opt_pred, class_weights)) / B
    y_test_opt_pred = optimal_pred(y_test_prob, class_weights)
    checkpoint["advantage_test"] = (weighted_acc(y_test.to_numpy().squeeze(), y_test_opt_pred, class_weights)) / B

checkpoint["B"] = B

checkpoint['train_original_prob'] = y_train_original_prob
checkpoint['test_prob'] = y_test_prob

print("==> Saving..")

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

if args.algo == 'catboost':
    fit_model.save_model(checkpoint_dir + 'model')
    checkpoint['best_iteration'] = fit_model.get_best_iteration()
else:
    raise NotImplementedError

# save_name = f"./checkpoint/criteoctr_{args.algo}_{args.seed}"
with open(checkpoint_dir + 'checkpoint_stats.pkl', 'wb') as handle:
    pickle.dump(checkpoint, handle, protocol=pickle.HIGHEST_PROTOCOL)
