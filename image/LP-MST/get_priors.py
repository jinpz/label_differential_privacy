#!/usr/bin/env python
# coding: utf-8

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models
from tqdm import tqdm
from sklearn.cluster import KMeans
from scipy.stats import dlaplace
from mnist_models import InceptionMnist
import scipy
import numpy as np

import os
import argparse

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--prior_type', default="indomain", type=str, help='the type of prior (indomain/outdomain)')
parser.add_argument('--dataset', default="mnist", type=str, help='the type of prior (indomain/outdomain)')
args = parser.parse_args()

prior_name = args.prior_type
dataset = args.dataset

#set-up
print("==> starting with set-up..")
if prior_name == "outdomain":
    checkpoint_name = "dino_vits8"
    model = torch.hub.load('facebookresearch/dino:main', checkpoint_name)
    num_clusters = 50
    eps_p = 0.025
    if dataset == "cifar10":
        transform_train = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
    elif dataset == "mnist":
        transform_train = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3,1,1)),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])        
elif prior_name == "indomain":
    num_clusters = 100
    eps_p = 0.05
    if dataset == "cifar10":
        model = models.resnet18(pretrained=False)
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        model.load_state_dict(torch.load("./BYOL/resnet18-CIFAR10-final.pt", map_location=device))
        model =  torch.nn.Sequential(*list(model.children())[:-1])
        transform_train = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
        ])
    elif dataset == "mnist":
        model = InceptionMnist(1, 10)
        checkpoint = torch.load("./pretrained_priors/mnist_byol_model.tar")
        model.load_state_dict(checkpoint)
        model.linear = torch.nn.Identity()
        MNIST_MEAN = (0.1307,)
        MNIST_STD = (0.3081,)
        normalize = [
            transforms.ToTensor(),
            transforms.Normalize(MNIST_MEAN, MNIST_STD),
        ]
        transform_train = transforms.Compose(normalize)

#load dataset
print("==> loading the dataset..")
batch_size = 256
if dataset == "cifar10":
    vanila_trainset = datasets.CIFAR10(root='~/data', train=True, download=True,
                                transform=transform_train)
    trainloader = torch.utils.data.DataLoader(vanila_trainset,
                                              batch_size=batch_size,
                                              shuffle=False, num_workers=8)
elif dataset == "mnist":
    vanila_trainset = datasets.MNIST(root="~/data", train=True, transform=transform_train, download=True)
    trainloader = torch.utils.data.DataLoader(vanila_trainset,
                                              batch_size=batch_size,
                                              shuffle=False, num_workers=8)

#get the embedding
print("==> getting the embedding..")
net = model.cuda()
embeddings = None
for batch_idx, (inputs, targets) in tqdm(enumerate(trainloader), total=len(trainloader)):
    net.eval()
    with torch.no_grad():
        inputs, targets = inputs.cuda(), targets.cuda()
        outputs = net(inputs)
        outputs = outputs.view(len(targets), -1)
        if embeddings is None:
            embeddings = torch.zeros([len(vanila_trainset), outputs.size()[1]])
        embeddings[batch_size * batch_idx:batch_size * (batch_idx + 1)] = outputs 
torch.save(embeddings, f"pretrained_priors/{dataset}_{prior_name}_embeddings.pl")

#kmeans clustering
print("==> clustering..")
embeddings_path = f"pretrained_priors/{dataset}_{prior_name}_embeddings.pl"
embeddings = torch.load(embeddings_path)
targets = torch.tensor(vanila_trainset.targets)
kmeans = KMeans(n_clusters=num_clusters, random_state=0, verbose=1).fit(embeddings)
checkpoint = {
    "cluster_labels": kmeans.labels_,
    "cluster_centers": kmeans.cluster_centers_
}
torch.save(checkpoint, f"pretrained_priors/{dataset}_{prior_name}_kmeans.pl")

#compute the prior probs
print("==> computing the prior probs..")
checkpoint = torch.load(f"pretrained_priors/{dataset}_{prior_name}_kmeans.pl")
cluster_labels = checkpoint["cluster_labels"]
num_classes = 10
prior_probs = torch.zeros([len(targets), num_classes])
targets = torch.tensor(vanila_trainset.targets)
correct = 0
torch.manual_seed(0)
for c in range(num_clusters):
    if_c = (cluster_labels == c)
    targets_c = targets[if_c]
    histo = torch.histc(targets_c.float(), bins=num_classes, min=-0.5, max=num_classes-0.5)
    correct += (targets_c == torch.argmax(histo)).sum()
    histo = histo + torch.tensor(dlaplace.rvs(eps_p/2, num_classes))
    histo = torch.clamp(histo, min=0)
    histo = histo / histo.sum()
    prior_probs[if_c] = histo

print("==> saving..")
torch.save(prior_probs, f"pretrained_priors/{dataset}_{prior_name}_prior_probs.pl")

