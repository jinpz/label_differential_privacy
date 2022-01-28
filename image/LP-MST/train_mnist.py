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
os.environ['KMP_WARNINGS'] = '0'

import numpy as np
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tqdm import tqdm
import math

import mnist_models
# from utils import progress_bar

parser = argparse.ArgumentParser(description='PyTorch mnist Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--model', default="ResNet18", type=str,
                    help='model type (default: ResNet18)')
parser.add_argument('--prior', default=None, type=str,
                    help='prior probabilities from pretrained features')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--batch-size', default=256, type=int, help='batch size')
parser.add_argument('--epoch', default=200, type=int,
                    help='total epochs to run')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='use standard augmentation (default: True)')
parser.add_argument('--decay', default=1e-4, type=float, help='weight decay')
parser.add_argument('--alpha', default=0.4, type=float,
                    help='mixup interpolation coefficient (default: 1)')
parser.add_argument('--eps', default=1., type=float,
                    help='paramter of label DP')
parser.add_argument('--num_stage', default=1, type=int,
                    help='total stages to run')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()

torch.manual_seed(args.seed)

# Data
print('==> Preparing data..')
trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
trainset = datasets.MNIST(root="./data", train=True, transform=trans, download=True)
testset = datasets.MNIST(root="./data", train=False, transform=trans, download=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                         shuffle=False, num_workers=8)

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
    for i in range(num_data):
        probs_sample = torch.zeros(num_classes)
        if if_top_opt_k[i]:
            probs_sample[sorted_ids[i, :(opt_k[i]+1)]] =  1 / (math.exp(eps) + opt_k[i])
            probs_sample[labels[i]] =  math.exp(eps) / (math.exp(eps) + opt_k[i])
        else:
            probs_sample[sorted_ids[i, :(opt_k[i]+1)]] = 1.0 / (opt_k[i] + 1)
        priv_labels[i] = torch.multinomial(probs_sample, 1, replacement=True)
    return priv_labels.long()

randperm = torch.randperm(len(trainset))

for t in range(args.num_stage):
    save_name = f"mnist_{args.eps}_{t}_{args.num_stage}_{args.alpha}_{args.lr}"
    eps = args.eps
    if args.prior is not None:
        save_name = save_name + "_" + args.prior
        if args.prior.startswith("dino"):
            eps -= 0.025
        elif args.prior.startswith("byol"):
            eps -= 0.05
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    best_net = None
    
    trainset_t = datasets.MNIST(root="./data", train=True, transform=trans, download=True)
    indices_t = randperm[int(float(t)*len(trainset)/args.num_stage):int(float(t+1)*len(trainset)/args.num_stage)].numpy()
    #TODO: non-uniform split in the paper for two stages
    trainset_t.data, trainset_t.targets = trainset_t.data[indices_t], torch.tensor(trainset_t.targets)[indices_t]
    if t == 0:
        if args.prior is None:
            prior_probs = torch.ones([len(trainset_t), 10]) / 10.0
        else:
            prior_probs = torch.load(f"checkpoint/mnist_{args.prior}_prior_probs.pl")[indices_t]
        priv_labels = RRWithPrior_batch(trainset_t.targets, prior_probs, eps=eps)
    else:
        #TODO
        prior_probs = torch.ones([len(trainset_t), 10]) / 10.0
        priv_labels = RRWithPrior_batch(trainset_t.targets, prior_probs, eps=eps)
    trainset.targets[indices_t] = priv_labels
    
    cum_trainset_t = datasets.MNIST(root="./data", train=True, transform=trans, download=True)
    cum_indices_t = randperm[:int(float(t+1)*len(trainset)/args.num_stage)]
    cum_trainset_t.data, cum_trainset_t.targets = trainset.data[cum_indices_t], torch.tensor(trainset.targets)[cum_indices_t]
    trainloader = torch.utils.data.DataLoader(cum_trainset_t,
                                              batch_size=args.batch_size,
                                              shuffle=True, num_workers=8)
    # Model
    if args.resume:
        # Load checkpoint.
        if os.path.exists('./checkpoint/ckpt.t7' + save_name + '_'
                                    + str(args.seed)):
            print('==> Resuming from checkpoint..')
            checkpoint = torch.load('./checkpoint/ckpt.t7' + save_name + '_'
                                    + str(args.seed))
            net = checkpoint['net']
            best_net = checkpoint['best_net']
            acc = checkpoint['acc']
            best_acc = checkpoint['bset_acc']
            start_epoch = checkpoint['epoch'] + 1
            rng_state = checkpoint['rng_state']
            torch.set_rng_state(rng_state)
        else:
            print('==> Building model..')
            net = mnist_models.InceptionMnist(1, 10)
    else:
        print('==> Building model..')
        net = mnist_models.InceptionMnist(1, 10)

    if not os.path.isdir('results'):
        os.mkdir('results')
    logname = ('results/log_' + net.__class__.__name__ + '_' + save_name + '_'
               + str(args.seed) + '.csv')

    if use_cuda:
        net.cuda()
#         net = torch.nn.DataParallel(net)
        print(torch.cuda.device_count())
        cudnn.benchmark = True
        print('Using CUDA..')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9,
                          weight_decay=args.decay)


    def mixup_data(x, y, alpha=1.0, use_cuda=True):
        '''Returns mixed inputs, pairs of targets, and lambda'''
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size()[0]
        if use_cuda:
            index = torch.randperm(batch_size).cuda()
        else:
            index = torch.randperm(batch_size)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam


    def mixup_criterion(criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


    def train(epoch):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        reg_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
#                 print(inputs.device)

            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets,
                                                           args.alpha, use_cuda)
            inputs, targets_a, targets_b = map(Variable, (inputs,
                                                          targets_a, targets_b))
            outputs = net(inputs)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (lam * predicted.eq(targets_a.data).cpu().sum().float()
                        + (1 - lam) * predicted.eq(targets_b.data).cpu().sum().float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Loss: %.3f | Reg: %.5f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/len(trainloader), reg_loss/len(trainloader),
                        100.*correct/total, correct, total))
#             progress_bar(batch_idx, len(trainloader),
#                          'Loss: %.3f | Reg: %.5f | Acc: %.3f%% (%d/%d)'
#                          % (train_loss/(batch_idx+1), reg_loss/(batch_idx+1),
#                             100.*correct/total, correct, total))
        return (train_loss/batch_idx, reg_loss/batch_idx, 100.*correct/total)


    def test(epoch, dataloader, net=net, save=True):
        global best_acc, best_net
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs, volatile=True), Variable(targets)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

#             progress_bar(batch_idx, len(testloader),
#                          'Loss: %.3f | Acc: %.3f%% (%d/%d)'
#                          % (test_loss/(batch_idx+1), 100.*correct/total,
#                             correct, total))
        acc = 100.*correct/total
#         if epoch == start_epoch + args.epoch - 1 or acc > best_acc:
        if acc > best_acc:
            best_acc = acc
            best_net = deepcopy(net)
        if save:
            if (epoch + 1) % 20 == 0 or epoch == args.epoch - start_epoch - 1:
                checkpoint(acc, epoch)
        print('Loss: %.3f | Acc: %.3f%% (%d/%d) | Best Acc: %.3f%%'
                 % (test_loss/len(testloader), 
                    100.*correct/total, correct, total, best_acc))
        return (test_loss/batch_idx, 100.*correct/total)


    def checkpoint(acc, epoch):
        # Save checkpoint.
        print('Saving..')
        state = {
            'net': net,
            'acc': acc,
            'best_net': best_net,
            'bset_acc': best_acc,
            'epoch': epoch,
            'rng_state': torch.get_rng_state()
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7' + save_name + '_'
                   + str(args.seed))


    def adjust_learning_rate(optimizer, epoch):
        """linearly grow from 0 to 0.02 in the first 15% training iterations, and then linearly decay to 0 in the remaining iterations."""
        lr = args.lr
        if float(epoch)/args.epoch <= 0.15:
            lr *= (float(epoch) / args.epoch) / 0.15
        else:
            lr *= (1 - float(epoch) / args.epoch) / 0.85
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


    if not os.path.exists(logname):
        with open(logname, 'w') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow(['epoch', 'train loss', 'reg loss', 'train acc',
                                'test loss', 'test acc'])

    for epoch in tqdm(range(start_epoch, args.epoch)):
        adjust_learning_rate(optimizer, epoch)
        train_loss, reg_loss, train_acc = train(epoch)
        test_loss, test_acc = test(epoch, testloader)
        with open(logname, 'a') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow([epoch, train_loss, reg_loss, train_acc, test_loss,
                                test_acc])
    vanila_trainset = datasets.MNIST(root="./data", train=True, transform=trans, download=True)
    vanila_trainloader = torch.utils.data.DataLoader(vanila_trainset,
                                              batch_size=args.batch_size,
                                              shuffle=False, num_workers=8)
    test_loss, test_acc = test(args.epoch, vanila_trainloader, net=net, save=False)
    test_loss, test_acc = test(args.epoch, vanila_trainloader, net=best_net, save=False)
    test_loss, test_acc = test(args.epoch, testloader, net=best_net, save=False)
