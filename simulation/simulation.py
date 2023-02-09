import numpy as np
from scipy.stats import multivariate_normal
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import os
import pickle
import argparse


def sample_one_vectorized(p):
    # p of shape N, num_classes
    # https://stackoverflow.com/questions/40474436/
    c = p.cumsum(axis=1)
    u = np.random.rand(len(c), 1)
    choices = (u < c).argmax(axis=1)
    return choices


parser = argparse.ArgumentParser(description='PyTorch mnist Training')
parser.add_argument('-num_classes', type=int, default=2, help='')
parser.add_argument('-d', type=int, default=100, help='data dimension')
parser.add_argument('-N', type=int, default=100, help='number of data points')
parser.add_argument('-sigma', type=float, default=1, help='standard deviation')
parser.add_argument('-eps', type=float, default=0.1, help='epsilon, privacy budget')
parser.add_argument('-repeat_outer', type=int, default=1, help='number of repeats sampling training data')
parser.add_argument('-repeat_inner', type=int, default=1000, help='number of repeats given the same training data')
parser.add_argument('-save_dir', type=str, default='results/', help='save location')
args = parser.parse_args()

print(vars(args))

num_classes = args.num_classes
d = args.d
N = args.N
sigma = args.sigma
var_eps = args.eps
repeat_outer = args.repeat_outer
repeat_inner = args.repeat_inner

# generate distributions

cov = np.eye(d) * sigma
means = []

for i in range(num_classes):
    current_mean = np.zeros((d,))
    current_mean[i] = 1
    means.append(current_mean)
distributions = []
for i in range(len(means)):
    distributions.append(multivariate_normal(means[i], cov))

L_EAUs = []
train_accs = []
train_dp_accs = []
for i in tqdm(range(repeat_outer)):
    rv_samples = []
    for j in range(num_classes):
        rv_samples.append(distributions[j].rvs(N // num_classes))
    train_samples = np.vstack(rv_samples)
    np.random.shuffle(train_samples)
    probabilities = np.zeros((N, num_classes))
    for j in range(num_classes):
        probabilities[:, j] = distributions[j].pdf(train_samples)
    probabilities = probabilities / np.sum(probabilities, axis=1)[:, None]
    L_EAU = np.mean(np.max(probabilities, axis=1))
    L_EAUs.append(L_EAU)
    current_train_accs = []
    current_dp_accs = []
    for j in range(repeat_inner):
        train_labels = sample_one_vectorized(probabilities)
        # 0: don't flip, 1: flip
        class_probabilities = np.ones((N, num_classes)) * 1 / (np.exp(var_eps) + num_classes - 1)
        class_probabilities[np.arange(N), train_labels] *= np.exp(var_eps)
        dp_labels = sample_one_vectorized(class_probabilities)
        if len(np.unique(dp_labels)) == 1:
            train_predictions = np.ones((train_samples.shape[0],)) * dp_labels[0]
        else:
            clf = LogisticRegression(random_state=0, solver='liblinear').fit(train_samples, dp_labels)
            train_predictions = clf.predict(train_samples)
        train_acc = np.mean(train_predictions == train_labels)
        dp_acc = np.mean(train_predictions == dp_labels)
        current_train_accs.append(train_acc)
        current_dp_accs.append(dp_acc)
    train_accs.append(current_train_accs)
    train_dp_accs.append(current_dp_accs)

print('mean of label-independent EAU {0}'.format(np.mean(L_EAUs)))

train_accs_mean = np.mean(np.array(train_accs), axis=1)
advantages = train_accs_mean - np.array(L_EAUs)

print('mean of advantage {0}'.format(np.mean(advantages)))

print('upper bound of advantage {0}'.format(1 - 2 / (1 + np.exp(var_eps))))

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

with open(args.save_dir + 'data.pkl', 'wb') as f:
    pickle.dump([train_accs, L_EAUs, train_dp_accs], f)

print('done')
