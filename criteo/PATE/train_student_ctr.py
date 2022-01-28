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
import scipy
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from catboost import CatBoostClassifier
from torch.utils.data import Dataset, DataLoader
import math
import torch
from tqdm import tqdm
from lib.utils import sample_one_vectorized, weighted_acc, optimal_pred, get_eps, get_eps_data_independent

parser = argparse.ArgumentParser(description='PyTorch mnist Training')
parser.add_argument('--algo', default="catboost", type=str,
                    help='algorithm')
parser.add_argument('--seed', default=47, type=int, help='random seed')
parser.add_argument('--dataset', default="raw_dataset_1M", type=str,
                    help='dataset name')
parser.add_argument('--data_path', default="../data/", type=str,
                    help='path to load data')
parser.add_argument('--model_path', default="results/PATE_ctr/", type=str,
                    help='base path to load models')
parser.add_argument('--save_path', default="results/PATE_ctr/", type=str,
                    help='base path to save checkpoint; algo_n_teachers_seed')
parser.add_argument('--n_teachers', default=100, type=int,
                    help='# of teachers')
parser.add_argument('--mechanism', default="lnmax", type=str,
                    help='lnmax, gnmax, gnmax_conf')
parser.add_argument('--n_samples', default=100, type=int,
                    help='# of student samples')
parser.add_argument('--tally_method', default='sample', type=str,
                    help='argmax or sample')
parser.add_argument('--selection_method', default='sample', type=str,
                    help='argmax or sample')
parser.add_argument('--noise_threshold', default=0, type=float,
                    help='threshold tau')
parser.add_argument('--selection_noise', default=0, type=float,
                    help='sigma1')
parser.add_argument('--result_noise', default=20, type=float,
                    help='sigma2')
args = parser.parse_args()

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

x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.20, stratify=y_test, random_state=256)

student_indices = np.random.permutation(len(x_train_original))

# Aggregate votes
print('==> Aggregating votes..')


def get_one_teacher_vote(prob, mode):
    if mode == 'argmax':
        votes = np.zeros_like(prob)
        votes[np.arange(len(prob)), prob.argmax(1)] = 1
    elif mode == 'sample':
        samples = sample_one_vectorized(prob)
        votes = np.zeros_like(prob)
        votes[np.arange(len(prob)), samples] = 1
    else:
        raise NotImplementedError
    return votes


teacher_votes = []
teacher_votes_test = []
for i in range(args.n_teachers):
    with open(args.model_path + 'teacher_{0}/checkpoint_stats.pkl'.format(i), 'rb') as f:
        checkpoint = pickle.load(f)
    teacher_prob = checkpoint['train_original_prob']
    teacher_prob_test = checkpoint['test_prob']
    teacher_votes.append(get_one_teacher_vote(teacher_prob, args.tally_method))
    teacher_votes_test.append(get_one_teacher_vote(teacher_prob_test, args.tally_method))

agg_votes = sum(teacher_votes)
agg_votes_test = sum(teacher_votes_test)

# eps = get_eps(agg_votes, args.mechanism, args.n_samples, args.result_noise, args.selection_noise, args.noise_threshold)
# print('data dependent eps is {0}'.format(eps))

eps = get_eps_data_independent(args.n_samples, args.result_noise, 1e-4)
print('data independent eps is {0}'.format(eps))

class_weights = [y_train_original.mean(), 1 - y_train_original.mean()]
B = (y_train_original.mean() * class_weights[1] + (1 - y_train_original.mean()) * class_weights[0]).item()

print('B is {0}'.format(B))

y_train_opt_pred = optimal_pred(agg_votes, class_weights)
advantage_train = (weighted_acc(y_train_original.to_numpy().squeeze(), y_train_opt_pred, class_weights)) / B
y_test_opt_pred = optimal_pred(agg_votes_test, class_weights)
advantage_test = (weighted_acc(y_test.to_numpy().squeeze(), y_test_opt_pred, class_weights)) / B

# need to reorder to align the prediction and indices
agg_votes = agg_votes[student_indices]

print('class 1 count percentage {0}'.format(np.sum(agg_votes[:, 1]) / (np.sum(agg_votes))))
print('class 1 argmax percentage {0}'.format(np.mean(agg_votes[:, 0] < agg_votes[:, 1])))
print('advantage train with argmax on {1}-aggregated histogram is {0}'.format(advantage_train, args.tally_method))
print('advantage test with argmax on {1}-aggregated histogram is {0}'.format(advantage_test, args.tally_method))


def noisy_threshold_labels_custom(votes, mechanism, threshold, selection_noise_scale, result_noise_scale, mode, class_1_portion):
    def noise(scale, mechanism, shape):
        if scale == 0:
            return 0
        if mechanism.startswith('lnmax'):
            return np.random.laplace(0, scale, shape)
        elif mechanism.startswith('gnmax'):
            return np.random.normal(0, scale, shape)
        else:
            raise NotImplementedError

    if mechanism == 'gnmax_conf':
        noisy_votes = votes + noise(selection_noise_scale, mechanism, votes.shape)
        over_t_mask = noisy_votes.max(axis=1) > threshold
        over_t_counts = (votes[over_t_mask] + noise(result_noise_scale, mechanism, votes[over_t_mask].shape))
    else:
        noisy_votes = votes + noise(result_noise_scale, mechanism, votes.shape)
        over_t_mask = noisy_votes.max(axis=1) > float('-inf')
        over_t_counts = noisy_votes
    if mode == 'argmax':
        over_t_labels = over_t_counts.argmax(axis=1)
    elif mode == 'sample':
        counts_max_0 = np.maximum(over_t_counts, 0)
        zero_indices = np.sum(counts_max_0, axis=1) == 0
        counts_max_0[zero_indices, 0] = 1 - class_1_portion
        counts_max_0[zero_indices, 1] = class_1_portion
        p = counts_max_0 / np.sum(counts_max_0, axis=1)[:, None]
        over_t_labels = sample_one_vectorized(p)
    else:
        raise NotImplementedError

    return over_t_labels, over_t_mask


labels, threshold_mask = noisy_threshold_labels_custom(
    votes=agg_votes,
    mechanism=args.mechanism,
    threshold=args.noise_threshold,
    selection_noise_scale=args.selection_noise,
    result_noise_scale=args.result_noise, mode=args.selection_method, class_1_portion=y_train_original.mean().values[0]
)

threshold_indices = threshold_mask.nonzero()[0]
indices = student_indices[threshold_indices][:args.n_samples]
labels = labels[:args.n_samples]

x_student = x_train_original.iloc[indices]
y_student_actual = y_train_original.iloc[indices]
y_student = labels
print('portion of class 1 labels after noisy histogram {0}, original {1}'.format(labels.mean(), y_student_actual.mean().values[0]))

advantage_student = (weighted_acc(y_student_actual.to_numpy().squeeze(), y_student, class_weights)) / B
print('advantage student dataset after noisy histogram is {0}'.format(advantage_student))

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
        x_student, y_student,
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

    y_student_prob = model.predict(x_student,
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
    dstudent = xgb.DMatrix(x_student, label=y_student, enable_categorical=True)
    dval = xgb.DMatrix(x_val, label=y_val, enable_categorical=True)
    dtest = xgb.DMatrix(x_test, label=y_test, enable_categorical=True)
    dtrain_original = xgb.DMatrix(x_train_original, label=y_train_original, enable_categorical=True)
    evallist = [(dval, 'eval'), (dstudent, 'train')]
    bst = xgb.train({'eval_metric': 'logloss'}, dstudent, num_round, evallist)


    def prob_to_complete_prob(y_prob):
        y_prob = np.expand_dims(y_prob, axis=1)
        y_prob = np.concatenate([1 - y_prob, y_prob], axis=1)
        return y_prob


    y_student_prob = prob_to_complete_prob(bst.predict(dstudent, iteration_range=(0, bst.best_iteration)))
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
    trainset = CTR(x_student, y_student)
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
    num_feat = x_student.shape[1]
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
    _, y_student_prob = test(best_net, trainloader, device)
    _, y_val_prob = test(best_net, valloader, device)
    _, y_test_prob = test(best_net, testloader, device)
    _, y_train_original_prob = test(best_net, train_original_loader, device)
    y_student_prob = y_student_prob.numpy()
    y_val_prob = y_val_prob.numpy()
    y_test_prob = y_test_prob.numpy()
    y_train_original_prob = y_train_original_prob.numpy()
else:
    raise NotImplementedError

print("==> Evaluating..")

checkpoint["log_loss"] = log_loss(y_test, y_test_prob[:, 1])

print('test log loss is {0}'.format(checkpoint['log_loss']))

# TODO add if clause for logistic regression
y_student_opt_pred = optimal_pred(y_student_prob, class_weights)
checkpoint["advantage_student"] = (weighted_acc(y_student_actual.to_numpy().squeeze(), y_student_opt_pred, class_weights)) / B
print('advantage student dataset after training is {0}'.format(checkpoint["advantage_student"]))

y_train_original_opt_pred = optimal_pred(y_train_original_prob, class_weights)
checkpoint["advantage_train"] = (weighted_acc(y_train_original.to_numpy().squeeze(), y_train_original_opt_pred, class_weights)) / B
y_test_opt_pred = optimal_pred(y_test_prob, class_weights)
checkpoint["advantage_test"] = (weighted_acc(y_test.to_numpy().squeeze(), y_test_opt_pred, class_weights)) / B

print('advantage train dataset after training is {0}'.format(checkpoint["advantage_train"]))
print('advantage test dataset after training is {0}'.format(checkpoint["advantage_test"]))

train_predictions = np.argmax(y_train_original_prob, axis=1)
train_labels = y_train_original.to_numpy().squeeze()
print('train acc after training is {0}'.format((train_predictions == train_labels).mean()))
print('train prediction percent 0 is {0}'.format(1 - np.mean(train_predictions)))

test_predictions = np.argmax(y_test_prob, axis=1)
test_labels = y_test.to_numpy().squeeze()
print('test acc after training is {0}'.format((test_predictions == test_labels).mean()))
print('test prediction percent 0 is {0}'.format(1 - np.mean(test_predictions)))

checkpoint['eps'] = eps

print("==> Saving..")

if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

if args.algo == 'catboost':
    fit_model.save_model(args.save_path + 'model')
    checkpoint['best_iteration'] = fit_model.get_best_iteration()
else:
    raise NotImplementedError

# save_name = f"./checkpoint/criteoctr_{args.algo}_{args.seed}"
with open(args.save_path + 'checkpoint_stats.pkl', 'wb') as handle:
    pickle.dump(checkpoint, handle, protocol=pickle.HIGHEST_PROTOCOL)