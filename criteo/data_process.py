#!/usr/bin/env python
# coding: utf-8

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")

import warnings
warnings.filterwarnings('ignore')

import gc, itertools, time, math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import category_encoders as ce
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, log_loss, roc_auc_score
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer, LabelEncoder, RobustScaler
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import plot_metric
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import scipy
import os


continue_var = ['I' + str(i) for i in range(1, 14)]
cat_features = ['C' + str(i) for i in range(1,27)]


col_names_train = ['Label'] + continue_var + cat_features
col_names_test = col_names_train[1:]

reader = pd.read_csv('./data/ctr_prediction/day_0', sep='\t', 
                     names=col_names_train, chunksize=100000, iterator=True)
raw_dataset = pd.DataFrame()
start = time.time()  
for i, chunk in enumerate(reader): 
    if raw_dataset.shape[0] >= 1000000:
        break
    raw_dataset = pd.concat([raw_dataset, chunk.sample(frac=.05, replace=False, random_state=911)], axis=0)  
    if i % 20 == 0:
        print('Processing Chunk No. {}, train size {}'.format(i, raw_dataset.shape[0])) 
print('Reading data costs %.2f seconds'%(time.time() - start))

raw_dataset.to_csv("./data/ctr_prediction/raw_dataset_1M.csv", index=False)

dataset = pd.read_csv(os.path.join("./data/ctr_prediction/", "raw_dataset_1M"+".csv"))
continue_var = ['I' + str(i) for i in range(1, 14)]
cat_features = ['C' + str(i) for i in range(1,27)]
trainset_path = os.path.join("./data/ctr_prediction/", "raw_dataset_1M"+"_train.csv")
testset_path = os.path.join("./data/ctr_prediction/", "raw_dataset_1M"+"_test.csv")

test_ratio = 0.2
np.random.seed(0)
n = len(dataset)
permutation = np.random.permutation(n)
train_indices = permutation[:int(n * (1 - test_ratio))]
test_indices = permutation[int(n * (1 - test_ratio)):]
train = dataset.iloc[train_indices]
test = dataset.iloc[test_indices]

fill_mean = lambda x: x.fillna(x.mean())
for col in continue_var:
    train[col] = train[col].groupby(train['C7']).apply(fill_mean)
    test[col] = test[col].groupby(test['C7']).apply(fill_mean)
    train[col] = train[col].fillna(test[col].mean())
    test[col] = test[col].fillna(test[col].mean())
    train[col] = train[col].astype('float64')
    test[col] = test[col].astype('float64')
train = train.fillna('unknown')
test = test.fillna('unknown')
for col in x_train.columns:
    if col.startswith("C"):
        x_train[col] = x_train[col].astype("category")
        x_val[col] = x_val[col].astype("category")
        x_test[col] = x_test[col].astype("category")
train.to_csv(trainset_path, index=False)
test.to_csv(testset_path, index=False)