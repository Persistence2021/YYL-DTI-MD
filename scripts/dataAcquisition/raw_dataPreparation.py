import os
import pickle
import numpy as np


# objectives = {
#     'RandomForest': {
#         'BCE': 'gini',
#         'MSE': 'squared_error',
#         'MAE': 'absolute_error',
#     },
#     'XGBoost': {
#         'BCE': 'binary:logistic',
#         'MSE': 'reg:squarederror',
#         'MAE': 'reg:absoluteerror'
#     }
# }
# drug_encoder='ECFP'
# target_encoder = ['One-Hot', 'FastText', 'Glove', 'Word2Vec', 'UniRep']
# dataset='Davis'
# split=''

seed = 42
def get_data(dataset, drug_encoder, target_encoder, split):
    data_path = '/home/cpua212/code/ylyao/MDDTI_YYL/Data/'

    with open(os.path.join(data_path, f'processed/Molecule/{drug_encoder}/{dataset}_{drug_encoder}_encoding.pickle'), 'rb') as handle:
        drug_embedding = pickle.load(handle)
    with open(os.path.join(data_path, f'processed/Target/{dataset}_{target_encoder}_encoding.pickle'), 'rb') as handle:
        target_embedding = pickle.load(handle)

    from tdc.multi_pred import DTI

    data_cls = DTI(name=dataset, path=os.path.join(data_path, f'raw'))
    data_cls.convert_to_log(form='binding')

    if split == 'random':
        data_cls = data_cls.get_split(seed=seed, frac=[0.8, 0.1, 0.1])
    elif split == 'cold-drug':
        data_cls = data_cls.get_split(method='cold_split', column_name='Drug', seed=seed)
    elif split == 'cold-target':
        data_cls = data_cls.get_split(method='cold_split', column_name='Target', seed=seed)
    elif split == 'cold':
        data_cls = data_cls.get_split(method='cold_split', column_name=['Drug', 'Target'], seed=seed)
    else:
        assert split in ['random', 'cold-drug', 'cold-target', 'cold'], \
                f'Splitting {split} not supported for TDC datasets, choose between ' \
                f'[random, cold-drug, cold-target, cold]'



    train_set = data_cls['train'].rename(
        columns={'Drug_ID': 'MID', 'Drug': 'Drug', 'Target_ID': 'PID', 'Target': 'Target',
                    'Y': 'Bioactivity'})
    train_interactions = train_set[['MID', 'PID']]
    train_y = np.array(train_set[['Bioactivity']])

    valid_set = data_cls['valid'].rename(
        columns={'Drug_ID': 'MID', 'Drug': 'Molecule', 'Target_ID': 'PID', 'Target': 'Protein',
            'Y': 'Bioactivity'})
    valid_interactions = valid_set[['MID', 'PID']]
    valid_y = np.array(valid_set[['Bioactivity']])[:, 0]

    test_set = data_cls['test'].rename(
        columns={'Drug_ID': 'MID', 'Drug': 'Molecule', 'Target_ID': 'PID', 'Target': 'Protein',
            'Y': 'Bioactivity'})
    test_interactions = test_set[['MID', 'PID']]
    test_y = np.array(test_set[['Bioactivity']])[:, 0]

    # 删除具有空值的记录
    train_interactions = train_interactions.dropna(axis=0, subset=['MID', 'PID'])
    test_interactions = test_interactions.dropna(axis=0, subset=['MID', 'PID'])
    valid_interactions = valid_interactions.dropna(axis=0, subset=['MID', 'PID'])

    train_x = []
    for x in train_interactions.iterrows():
        train_x.append(np.concatenate((drug_embedding[x[1]['MID']], target_embedding[x[1]['PID']])))

    valid_x = []
    for x in valid_interactions.iterrows():
        valid_x.append(np.concatenate((drug_embedding[x[1]['MID']], target_embedding[x[1]['PID']])))

    test_x = []
    for x in test_interactions.iterrows():
        test_x.append(np.concatenate((drug_embedding[x[1]['MID']], target_embedding[x[1]['PID']])))

    return train_x, valid_x, test_x, train_y,  valid_y,  test_y


def get_data_Regression(dataset, drug_encoder, target_encoder, split):
    data_path = '/home/cpua212/code/ylyao/MDDTI_YYL/Data/'

    with open(os.path.join(data_path, f'processed/Molecule/{drug_encoder}/{dataset}_{drug_encoder}_encoding.pickle'),
              'rb') as handle:
        drug_embedding = pickle.load(handle)
    with open(os.path.join(data_path, f'processed/Target/{dataset}_{target_encoder}_encoding.pickle'),
              'rb') as handle:
        target_embedding = pickle.load(handle)



    from tdc.multi_pred import DTI

    data_cls = DTI(name=dataset, path=os.path.join(data_path, f'raw'))
    # 转化y活性为P值
    # data_cls.convert_to_log(form='binding')


    if split == 'random':
        data_cls = data_cls.get_split(seed=seed)
    elif split == 'cold-drug':
        data_cls = data_cls.get_split(method='cold_split', column_name='Drug', seed=seed)
    elif split == 'cold-target':
        data_cls = data_cls.get_split(method='cold_split', column_name='Target', seed=seed)
    elif split == 'cold':
        data_cls = data_cls.get_split(method='cold_split', column_name=['Drug', 'Target'], seed=seed)
    else:
        assert split in ['random', 'cold-drug', 'cold-target', 'cold'], \
            f'Splitting {split} not supported for TDC datasets, choose between ' \
            f'[random, cold-drug, cold-target, cold]'


    train_set = data_cls['train'].rename(
        columns={'Drug_ID': 'MID', 'Drug': 'Drug', 'Target_ID': 'PID', 'Target': 'Target',
                 'Y': 'Bioactivity'})
    train_interactions = train_set[['MID', 'PID']]
    train_y = np.array(train_set[['Bioactivity']])

    test_set = data_cls['test'].rename(
        columns={'Drug_ID': 'MID', 'Drug': 'Molecule', 'Target_ID': 'PID', 'Target': 'Protein',
                 'Y': 'Bioactivity'})
    test_interactions = test_set[['MID', 'PID']]
    test_y = np.array(test_set[['Bioactivity']])[:, 0]


    # 删除具有空值的记录
    train_interactions = train_interactions.dropna(axis=0, subset=['MID', 'PID'])

    test_interactions = test_interactions.dropna(axis=0, subset=['MID', 'PID'])
    excepted_list = ['O94925', 'Q9NXA8']
    train_x = []
    for x in train_interactions.iterrows():
        # if x[1]['MID'] in excepted_list or x[1]['PID'] in excepted_list:
        #     pass
        # else:
        train_x.append(np.concatenate((drug_embedding[x[1]['MID']], target_embedding[x[1]['PID']])))

    test_x = []
    for x in test_interactions.iterrows():
        test_x.append(np.concatenate((drug_embedding[x[1]['MID']], target_embedding[x[1]['PID']])))

    return train_x, test_x, train_y, test_y