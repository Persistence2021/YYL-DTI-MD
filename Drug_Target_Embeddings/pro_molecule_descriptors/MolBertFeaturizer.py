import pickle
import os
import gc
import sys
import numpy as np
from rdkit import Chem
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from multiprocessing import freeze_support

'''
需要针对不同的编码方式配置不同的conda环境，避免python包的版本冲突
'''
#显示所有列
# pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)

def encode_drug(batch, encoder, name):
    """ Wraps encoding of drug compounds, i.e. drugs, from different encoders. """
    embeddings = encoder(batch)
    return embeddings if name == 'CDDD' else embeddings[0]

def precompute_drug_embeddings(drugs, encoder_name, batch_size):
    gc.collect()
    if encoder_name == 'CDDD':          # Server conda env cddd not work on server
        from cddd.inference import InferenceModel
        CDDD_MODEL_DIR = os.path.join(data_path, 'checkpoints/CDDD/default_model')
        assert os.path.exists(CDDD_MODEL_DIR), \
            'Error: default model should be downloaded according to CDDD github.' \
            'Place the default_model folder under checkpoints/CDDD/'
        cddd_model = InferenceModel(CDDD_MODEL_DIR)
        mol_encoder = cddd_model.seq_to_emb
    elif encoder_name == 'MolBert':     # Server conda env molbert
        from molbert.utils.featurizer.molbert_featurizer import MolBertFeaturizer
        MOLBERT_MODEL_DIR = os.path.join(data_path, 'checkpoints/MolBert/molbert_100epochs/checkpoints/last.ckpt')
        assert os.path.exists(MOLBERT_MODEL_DIR), \
            'Error: checkpoint should be downloaded according to CDDD github.' \
            'Place the file last.ckpt under checkpoints/MolBert/molbert_100epochs/checkpoints/last.ckpt'
        molbert_model = MolBertFeaturizer(MOLBERT_MODEL_DIR, max_seq_len=500, embedding_type='average-1-cat-pooled') #
        print("Model initialization!")
        mol_encoder = molbert_model.transform
        print("Model pre-processing!")
    else:
        print(f'Target encoder {encoder_name} currently not supported.')
        sys.exit()

    embeddings = np.array([])
    with ThreadPoolExecutor(max_workers=8) as executor:
        desc = f'Pre-computing drug encodings with {encoder_name} : '
        batches = (drugs[i:i + batch_size] for i in range(0, len(drugs), batch_size))
        threads = [executor.submit(encode_drug, batch, mol_encoder, encoder_name) for batch in batches]
        for t in tqdm(threads, desc=desc, colour='white'):
            emb = t.result()
            embeddings = emb if len(embeddings) == 0 else np.append(embeddings, emb, axis=0)
    return embeddings

# 获取从Uniprot网站获得关于精神类疾病的Entry
md_datapath = "/home/cpua212/code/ylyao/Data/Mental_disorder/"
data_path = '/home/cpua212/code/ylyao/MDDTI_YYL/Data/'
datafiles = ['KIBA_md.csv', 'BindingDBIC50_md.csv', 'Lenselink_md.csv']
dataset = datafiles[2]
data = pd.read_csv(os.path.join(md_datapath, dataset))
md_data = pd.read_csv(os.path.join(md_datapath, dataset), sep=',')
# 数据集修改列名
# md_data = md_data.rename(columns={'Drug_ID': 'MID', 'Drug': 'Drug',
#                             'Target_ID': 'PID', 'Target': 'Target',
#                             'Y': 'Bioactivity'})
# Drug or Target embeddings
'''分别对应计算embeddings'''
drug_type = ['CDDD', 'MolBert']
drug_encoder = drug_type[1]
structures = list(md_data['Drug'].unique())
unique_ids = list(md_data.MID.unique())
print(len(unique_ids))
# structures = [Chem.MolFromSmiles(smiles) for smiles in structures]

encoding_fn = precompute_drug_embeddings
embeddings = encoding_fn(structures, drug_encoder, batch_size=10)

embedding_dict = {}
for pid, emb in zip(unique_ids, embeddings):
    embedding_dict[pid] = emb

# print(embedding_dict)
with open(os.path.join(md_datapath, f'processed/Molecule/{drug_encoder}/{dataset.split(".")[0]}_{drug_encoder}_encoding.pickle'), 'wb') as handle:
    pickle.dump(embedding_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)