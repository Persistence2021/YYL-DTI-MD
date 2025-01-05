import pickle

import pandas as pd


import os
import gc
import sys
import torch
import numpy as np
from tqdm import tqdm
from itertools import tee
from typing import Any, Dict, List, Generator, Iterable
from concurrent.futures import ThreadPoolExecutor

'''
需要针对不同的编码方式配置不同的conda环境，避免python包的版本冲突
'''
# 重新定义类
class UniRepEmbedder:

    _params: Dict[str, Any]

    def __init__(self):
        from jax_unirep.utils import load_params
        self._params = load_params()

    def embed_batch(self, batch):
        from jax_unirep.featurize import get_reps
        h, h_final, c_final = get_reps(batch)
        return h

    @staticmethod
    def reduce_per_target(embedding):
        return embedding.mean(axis=0)

def encode_target(batch, encoder):
    """ Wraps encoding of protein targets, i.e. targets, from different encoders. """
    full_embeddings = encoder.embed_batch(batch)
    embeddings = np.array([])
    for emb in full_embeddings:
        # emb = encoder.reduce_per_target(emb)
        emb = encoder.reduce_per_protein(emb)
        emb = np.expand_dims(emb, axis=0)
        embeddings = emb if len(embeddings) == 0 else np.append(embeddings, emb, axis=0)
    return embeddings
def precompute_target_embeddings(targets, encoder_name, batch_size, n_jobs=8):
    gc.collect()
    if encoder_name == 'UniRep':
        bio_encoder = UniRepEmbedder()
    elif encoder_name == 'Glove':  # Server Conda env bio_embeddings. local requires batch_size 4
        from bio_embeddings.embed import GloveEmbedder
        bio_encoder = GloveEmbedder()
    elif encoder_name == 'SeqVec':  # Server Conda env bio_embeddings. local requires batch_size 4
        from bio_embeddings.embed import SeqVecEmbedder
        bio_encoder = SeqVecEmbedder()
    else:
        print(f'Target encoder {encoder_name} currently not supported.')
        sys.exit()

    embeddings = np.array([])

    desc = f'Pre-computing target encodings with {encoder_name}: '
    batches = (targets[i:i + batch_size] for i in range(0, len(targets), batch_size))
    if n_jobs > 0:
        assert n_jobs < 9, f'{n_jobs} are too many workers for parallel computing.'
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:# 绘制进度条
            if encoder_name == 'UniRep':
                threads = [executor.submit(bio_encoder.embed_batch, batch) for batch in batches]
            else:
                threads = [executor.submit(encode_target, batch, bio_encoder) for batch in batches]
            for t in tqdm(threads, desc=desc, colour='white'):
                emb = t.result()
                embeddings = emb if len(embeddings) == 0 else np.append(embeddings, emb, axis=0)
    else:
        results = [encode_target(batch, bio_encoder) for batch in batches]
        for emb in tqdm(results, desc=desc, colour='white'):
            embeddings = emb if len(embeddings) == 0 else np.append(embeddings, emb, axis=0)

    del bio_encoder
    return embeddings

# 3种靶标氨基酸序列的处理方法,无transformers模'
# 可选数据集LenseLink，KIBA，BindingDB
md_datapath = "/home/cpua212/code/ylyao/MDDTI_YYL/Data/Mental_disorder/"
data_path = '/home/cpua212/code/ylyao/MDDTI_YYL/Data/'
datafiles = ['KIBA_md.csv', 'BindingDBIC50_md.csv', 'Lenselink_md.csv']
dataset = datafiles[0]
target_type = ["SeqVec", 'Glove', 'UniRep']
target_encoder = target_type[-1]
data = pd.read_csv(os.path.join(md_datapath, dataset))
print(data.head())

data = data.rename(columns={'Drug_ID': 'MID', 'Drug': 'Drug',
                            'Target_ID': 'PID', 'Target': 'Target',
                            'Y': 'Bioactivity'})
# Drug or Target embedding

structures = list(data['Target'].unique())
unique_ids = list(data['PID'].unique())

print(f'The amount of target is {len(structures)}')
print(f'The amount of id is {len(unique_ids)}')


encoding_fn = precompute_target_embeddings

embeddings = encoding_fn(structures[:5], encoder_name=target_encoder, batch_size=4, n_jobs=6)
print(type(embeddings))
embedding_dict = {}
for pid, emb in zip(unique_ids, embeddings):
    embedding_dict[pid] = emb
print(f'The length of embedding dictionary is {len(embedding_dict)}')

with open(os.path.join(md_datapath, f'processed/Target/{target_encoder}/{dataset.split(".")[0]}_{target_encoder}_encoding.pickle'), 'wb') as handle:
    pickle.dump(embedding_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
