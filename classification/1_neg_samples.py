from tqdm import tqdm
import argparse
import pandas as pd
import logging
from data_utils import *
import random
import numpy as np
import gc

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int)
parser.add_argument("--seed", type=int)
args = parser.parse_args()

# set seeds
random.seed(args.seed)
np.random.seed(args.seed)

# load all data
for i in range(4):
    if i == 0:
        samples = pd.read_json(path_or_buf=f'../final_dataset/dataset_cs{i}.json', dtype= {'principal_id': int, 'cited_id': int})
    else:
        samples_i = pd.read_json(path_or_buf=f'../final_dataset/dataset_cs{i}.json', dtype= {'principal_id': int, 'cited_id': int})
        samples = pd.concat([samples, samples_i], ignore_index=True)
        del samples_i
        gc.collect()

print(f'Original samples: {samples.shape[0]}')

model = SentenceTransformer('../cs_BERT/SentenceCSBert/')

# retrieve random principal papers
principal_ids = list(set(list(samples['principal_id'])))
random_idx = list(np.random.randint(low = 0, high = len(principal_ids), size=args.n))
ind_paper = [principal_ids[i] for i in random_idx]
print(ind_paper)

# determine corresponding random cited papers
neg_samples = []
for p_id in tqdm(ind_paper):
    df_principal = samples[samples['principal_id'] == p_id].iloc[0,:]
    df_filtered = samples[(samples['principal_id'] != p_id) & (samples['cited_id'] != p_id)]

    # maximum distance
    p_title = df_principal["principal_title"]
    c_titles = list(df_filtered['principal_title']) + list(df_filtered['cited_title'])

    # randomly select 50 paper titles
    c_titles_idx = list(np.random.randint(low = 0, high = len(c_titles), size=50))
    c_titles_text = [c_titles[c] for c in c_titles_idx]

    embeddings1 = model.encode(p_title, convert_to_tensor=True)
    embeddings2 = model.encode(c_titles_text, convert_to_tensor=True)

    cos_scores = util.cos_sim(embeddings1, embeddings2)
    ind_max = int(torch.argmax(cos_scores)) # selected cited paper as maximum distance
    neg_ind = c_titles_idx[ind_max]

    if neg_ind > len(list(df_filtered['principal_title'])):
        neg_ind -= len(list(df_filtered['principal_title']))

        df_cited = df_filtered.iloc[neg_ind,:] # cited paper

        neg_samples.append({
            "principal_id": df_principal["principal_id"],
            "principal_title": df_principal["principal_title"],
            "principal_abstracts": df_principal["principal_abstracts"],
            "principal_text": df_principal["principal_text"],

            "cited_id": df_cited["cited_id"],
            "cited_title": df_cited["cited_title"],
            "cited_abstract": df_cited["cited_abstract"],
            "cited_text": df_cited["cited_text"]
        })

    else:
        df_cited = df_filtered.iloc[neg_ind,:] # principal paper

        neg_samples.append({
            "principal_id": df_principal["principal_id"],
            "principal_title": df_principal["principal_title"],
            "principal_abstracts": df_principal["principal_abstracts"],
            "principal_text": df_principal["principal_text"],

            "cited_id": df_cited["principal_id"],
            "cited_title": df_cited["principal_title"],
            "cited_abstract": df_cited["principal_abstracts"],
            "cited_text": df_cited["principal_text"]
        })

neg_samples = pd.DataFrame(neg_samples)
neg_samples.to_pickle("data/neg_samples.pkl")  