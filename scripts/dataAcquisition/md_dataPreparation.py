from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import os
import sys
import json
import math
import pickle
import numpy as np
import pandas as pd

# from hyper_dti.settings import constants

seed = 42


def get_md_data(dataset, drug_encoder, target_encoder):
    # 加载嵌入数据文件
    md_datapath = "F:\\MDDTI_YYL\\Data\\Mental_disorder\\"
    with open(os.path.join(md_datapath, f'processed\\Molecule\\{drug_encoder}/{dataset.split(".")[0]}_{drug_encoder}_encoding.pickle'),
              'rb') as handle:
        drug_embedding = pickle.load(handle)
    with open(os.path.join(md_datapath, f'processed\\Target\\{target_encoder}/{dataset.split(".")[0]}_{target_encoder}_encoding.pickle'), 'rb') as handle:
        target_embedding = pickle.load(handle)

    # 针对未筛选的Lenselink数据集可按照如下操作分割数据集
    if dataset == 'Lenselink':
        data = pd.read_pickle(os.path.join(md_datapath, f"data.pickle"))
        data = data.astype({"MID": "int"}, copy=False)

        train_set = data[data["temporal"] == "train"]
        test_set = data[data["temporal"] == "test"]
        valid_set = data[data["temporal"] == "valid"]

        train_interactions = train_set[["MID", "PID"]]
        train_y = np.array(train_set[["Bioactivity"]])[:, 0]
        test_interactions = test_set[["MID", "PID"]]
        test_y = np.array(test_set[["Bioactivity"]])[:, 0]
        valid_interactions = valid_set[["MID", "PID"]]
        valid_y = np.array(valid_set[["Bioactivity"]])[:, 0]

        train_x = []
        for x in train_interactions.iterrows():
            train_x.append(np.concatenate((drug_embedding[x[1]['MID']], target_embedding[x[1]['PID']])))
        test_x = []
        for x in test_interactions.iterrows():
            test_x.append(np.concatenate((drug_embedding[x[1]['MID']], target_embedding[x[1]['PID']])))
        valid_x = []
        for x in valid_interactions.iterrows():
            valid_x.append(np.concatenate((drug_embedding[x[1]['MID']], target_embedding[x[1]['PID']])))
    else:
        data_cls = pd.read_csv(os.path.join(md_datapath, dataset), sep=',')
        data_cls = data_cls.rename(
            columns={'Drug_ID': 'MID', 'Drug': 'Drug', 'Target_ID': 'PID', 'Target': 'Target',
                     'Y': 'Bioactivity'})

        data_interactions = data_cls[['MID', 'PID']]
        data_y = np.array(data_cls[['Bioactivity']])[:, 0]
        data_interactions = data_interactions.dropna(axis=0, subset=['MID', 'PID'])

        data_x = []
        for x in data_interactions.iterrows():
            data_x.append(np.concatenate((drug_embedding[x[1]['MID']], target_embedding[x[1]['PID']])))

        train_x, tmp_x, train_y, tmp_y = train_test_split(data_x, data_y, test_size=0.2, random_state=seed)
        valid_x, test_x, valid_y, test_y = train_test_split(tmp_x, tmp_y, test_size=0.5, random_state=seed)

    return train_x, valid_x, test_x, train_y, valid_y, test_y


def get_md_regressor(dataset, drug_encoder, target_encoder):
    # 加载嵌入数据文件
    md_datapath = "F:\\MDDTI_YYL\\Data\\Mental_disorder\\"
    with open(os.path.join(md_datapath, f'processed\\Molecule\\{drug_encoder}/{dataset.split(".")[0]}_{drug_encoder}_encoding.pickle'), 'rb') as handle:
        drug_embedding = pickle.load(handle)
    with open(os.path.join(md_datapath, f'processed\\Target\\{target_encoder}/{dataset.split(".")[0]}_{target_encoder}_encoding.pickle'), 'rb') as handle:
        target_embedding = pickle.load(handle)
    # 针对未筛选的Lenselink数据集可按照如下操作分割数据集
    if dataset == 'Lenselink':
        data = pd.read_pickle(os.path.join(md_datapath, f"data.pickle"))
        data = data.astype({"MID": "int"}, copy=False)
        train_set = data[data["temporal"] == "train"]
        test_set = data[data["temporal"] == "test"]

        train_interactions = train_set[["MID", "PID"]]
        train_y = np.array(train_set[["Bioactivity"]])[:, 0]
        test_interactions = test_set[["MID", "PID"]]
        test_y = np.array(test_set[["Bioactivity"]])[:, 0]

        train_x = []
        for x in train_interactions.iterrows():
            train_x.append(np.concatenate((drug_embedding[x[1]['MID']], target_embedding[x[1]['PID']])))
        test_x = []
        for x in test_interactions.iterrows():
            test_x.append(np.concatenate((drug_embedding[x[1]['MID']], target_embedding[x[1]['PID']])))
    else:
        data_cls = pd.read_csv(os.path.join(md_datapath, dataset), sep=',')
        data_cls = data_cls.rename(
            columns={'Drug_ID': 'MID', 'Drug': 'Drug', 'Target_ID': 'PID', 'Target': 'Target',
                     'Y': 'Bioactivity'})

        data_interactions = data_cls[['MID', 'PID']]
        data_y = np.array(data_cls[['Bioactivity']])[:, 0]

        data_x = []
        for x in data_interactions.iterrows():
            data_x.append(np.concatenate((drug_embedding[x[1]['MID']], target_embedding[x[1]['PID']])))

        train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.2, random_state=seed)

    return train_x, test_x, train_y, test_y