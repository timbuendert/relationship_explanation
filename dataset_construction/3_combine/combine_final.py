import pandas as pd
import pickle
import numpy as np
from collections import Counter
import gc 
import argparse
from tqdm import trange

def filter_length(df, section, l):
    ind = []
    for o in range(df.shape[0]):
        text = ' '.join([df[section][o][i]['text'] for i in range(len(df[section][o]))])
        if len(text) > l:
            ind.append(o)
    return df.iloc[ind,:].reset_index(drop = True)

argparser = argparse.ArgumentParser()
argparser.add_argument('--start', type=int)
argparser.add_argument('--end', type=int)
argparser.add_argument('--n', type=str)
args = argparser.parse_args()

# import samples
samples = pd.read_json(path_or_buf='samples_all.jsonl', lines=True, dtype= {'principal_id': int, 'cited_id': int})
samples = samples.iloc[args.start:args.end, :]

# import data and insert retrieved data into final dataframe
with open('data/p_title.pkl', 'rb') as f:
    p_title = pickle.load(f)    
p_title = p_title[args.start:args.end]
samples.insert(loc=1, column='principal_title', value=p_title)
del p_title
gc.collect()

with open('data/p_abs.pkl', 'rb') as f:
    p_abs = pickle.load(f)    
p_abs = p_abs[args.start:args.end]
samples.insert(loc=2, column='principal_abstracts', value=p_abs)
del p_abs
gc.collect()

with open('data/p_text.pkl', 'rb') as f:
    p_text = pickle.load(f)    
p_text = p_text[args.start:args.end]
samples.insert(loc=3, column='principal_text', value=p_text)
del p_text
gc.collect()


with open('data/c_title.pkl', 'rb') as f:
    c_title = pickle.load(f)
c_title = c_title[args.start:args.end]
samples.insert(loc=8, column='cited_title', value=c_title)
del c_title
gc.collect()

with open('data/c_abs.pkl', 'rb') as f:
    c_abs = pickle.load(f)
c_abs = c_abs[args.start:args.end]
samples.insert(loc=9, column='cited_abstract', value=c_abs)
del c_abs
gc.collect()

with open('data/c_text.pkl', 'rb') as f:
    c_text = pickle.load(f) 
c_text = c_text[args.start:args.end]
samples.insert(loc=10, column='cited_text', value=c_text)
del c_text
gc.collect()


print(f'Shape of samples: {samples.shape}')

samples = pd.read_json(path_or_buf=f'dataset_cs_unfiltered{args.n}.json')

# only keep samples where the texts of the principal and cited document are present (or via data = data[data['body_text'].map(len) > 0])
samples = samples[samples['principal_abstracts'].apply(lambda x: x != None) & samples['principal_text'].apply(lambda x: x != None) & samples['cited_text'].apply(lambda x: x != None) & samples['cited_abstract'].apply(lambda x: x != None)].reset_index(drop = True)
print(f'Shape of samples after filtering (1): {samples.shape}')

samples = filter_length(samples, 'principal_abstracts', 300)
samples = filter_length(samples, 'principal_text', 750)
samples = filter_length(samples, 'cited_text', 750)
samples = filter_length(samples, 'cited_abstract', 300)
samples = samples[samples['explanation'].apply(lambda x: len(x) > 25)].reset_index(drop = True)

print(f'Shape of samples after filtering (2): {samples.shape}')

keep_idx = []
for i in trange(samples.shape[0]):
    conc = False
    intro = False
    try:
        sections = [samples['principal_text'][i][j]['section'].lower() for j in range(len(samples['principal_text'][i]))]
    except:
        continue
    for s in sections: 
        if 'conclusion' in s:
            conc = True
        if 'introduction' in s:
            intro = True
    if conc and intro:
        keep_idx.append(i)
samples = samples.iloc[keep_idx,:]
print(f'Shape of samples after filtering (3): {samples.shape}')


samples.to_json(f'dataset_cs{args.n}.json', orient = "records")