import numpy as np
from lib.pate.accountant import run_analysis


def sample_one_vectorized(p):
    # p of shape N, num_classes
    # https://stackoverflow.com/questions/40474436/
    c = p.cumsum(axis=1)
    u = np.random.rand(len(c), 1)
    choices = (u < c).argmax(axis=1)
    return choices


def weighted_acc(y, y_pred, class_weights):
    sample_weight = np.ones(len(y))
    sample_weight[y == 0] = class_weights[0]
    sample_weight[y == 1] = class_weights[1]
    acc = (y == y_pred).astype(np.float)
    return (acc * sample_weight).mean()


def optimal_pred(y_probs, class_weights):
    class_weights = np.asarray(class_weights).squeeze()
    return (y_probs * class_weights).argmax(1)


def get_eps(votes, mechanism, n_samples, result_noise, selection_noise, threshold):
    if result_noise == 0:
        return float('inf')
    if mechanism == 'gnmax_conf':
        if selection_noise == 0:
            return float('inf')

    eps_total, partition, answered, order_opt = run_analysis(votes, mechanism, result_noise,
                                                             {"sigma1": selection_noise, "t": threshold})
    for i, x in enumerate(answered):
        if int(x) >= n_samples:
            return eps_total[i]
    print(x)
    return -1


def get_eps_data_independent(n_samples, result_noise, delta):
    gamma = 1 / result_noise
    return 4 * n_samples * gamma ** 2 + 2 * gamma * np.sqrt(2 * n_samples * np.log(1 / delta))
