import os.path
# import os
import pickle
import gc
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
'''
需要针对不同的编码方式配置不同的conda环境，避免python包的版本冲突
'''

'''****************************初始基本设置****************************'''
# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置显示非科学计数法，最多6位小数展示
np.set_printoptions(precision=6, suppress=False)

def encode_drug(batch, encoder, name):
    """ Wraps encoding of drug compounds, i.e. drugs, from different encoders. """
    embeddings = encoder(batch)
    return embeddings[0]

# def precompute_drug_embeddings(drugs, encoder_name, split, batch_size):
def precompute_drug_embeddings(drugs, encoder_name, batch_size):
    gc.collect()
    if encoder_name == 'ECFP':         # Server conda env molbert
        from molbert.utils.featurizer.molfeaturizer import MorganFPFeaturizer
        ecfp_model = MorganFPFeaturizer(fp_size=2048, radius=2, use_counts=True, use_features=False)
        mol_encoder = ecfp_model.transform
    elif encoder_name == 'RDKit':        # Server conda env molbert? ERROR
        from molbert.utils.featurizer.molfeaturizer import PhysChemFeaturizer
        rdkit_norm_model = PhysChemFeaturizer(normalise=True)
        mol_encoder = rdkit_norm_model.transform
    elif encoder_name == 'MolBert':     # Server conda env molbert
        from molbert.utils.featurizer.molbert_featurizer import MolBertFeaturizer
        MOLBERT_MODEL_DIR = 'checkpoints/MolBert/molbert_100epochs/checkpoints/last.ckpt'
        assert os.path.exists(MOLBERT_MODEL_DIR), \
            'Error: checkpoint should be downloaded according to CDDD github.' \
            'Place the file last.ckpt under checkpoints/MolBert/molbert_100epochs/checkpoints/last.ckpt'
        molbert_model = MolBertFeaturizer(MOLBERT_MODEL_DIR, max_seq_len=500, embedding_type='average-1-cat-pooled')
        mol_encoder = molbert_model.transform

    else:
        print(f'Target encoder {encoder_name} currently not supported.')
        sys.exit()

    embeddings = np.array([])
    with ThreadPoolExecutor(max_workers=8) as executor:# 绘制进度条
        # desc = f'Pre-computing drug encodings with {encoder_name} for {split}: '
        desc = f'Pre-computing drug encodings with {encoder_name}'
        batches = (drugs[i:i + batch_size] for i in range(0, len(drugs), batch_size))
        threads = [executor.submit(encode_drug, batch, mol_encoder, encoder_name) for batch in batches]
        for t in tqdm(threads, desc=desc, colour='white'):
            emb = t.result()
            embeddings = emb if len(embeddings) == 0 else np.append(embeddings, emb, axis=0)
    return embeddings






# 可选数据集Davis，KIBA，BindingDB
md_datapath = "/home/cpua212/code/ylyao/MDDTI_YYL/Data/Mental_disorder/"
data_path = '/home/cpua212/code/ylyao/MDDTI_YYL/Data/'
datafiles = ['KIBA_md.csv', 'BindingDBIC50_md.csv', 'Lenselink_md.csv']
drug_type = ['ECFP', 'RDKit', 'MolBert']
dataset = datafiles[0]
drug_encoder = drug_type[1]
data = pd.read_csv(os.path.join(md_datapath, dataset))
print(data.head())

data = data.rename(columns={'Drug_ID': 'MID', 'Drug': 'Drug',
                            'Target_ID': 'PID', 'Target': 'Target',
                            'Y': 'Bioactivity'})
# Drug or Target embeddings
'''分别对应计算embeddings'''
structures = list(data['Drug'].unique())
unique_ids = list(data['MID'].unique())
print(len(structures))
print(len(unique_ids))

encoding_fn = precompute_drug_embeddings

embeddings = encoding_fn(structures, encoder_name=drug_encoder, batch_size=4)
embedding_dict = {}

for pid, emb in zip(unique_ids, embeddings):
    print(emb)
    embedding_dict[pid] = emb

# print(len(embedding_dict))
# # print(embedding_dict)
# with open(os.path.join(md_datapath, f'processed/Molecule/{drug_encoder}/{dataset.split(".")[0]}_{drug_encoder}_encoding.pickle'), 'wb') as handle:
#     pickle.dump(embedding_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
