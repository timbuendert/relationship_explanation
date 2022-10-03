import pandas as pd
from tqdm import trange, tqdm
import gc
import argparse
import pickle

def none_counter(ls1, ls2 = 0):
    counter = 0
    if ls2 != 0:
        for o in range(len(ls1)):
            if (ls1[o] == None) or (ls2[o] == None):
                counter +=1
    else:
        for o in ls1:
            if o == None:
                counter += 1
    return counter


# import samples
samples = pd.read_json(path_or_buf='samples_all.jsonl', lines=True, dtype= {'principal_id': int, 'cited_id': int})
print(f'Shape of samples: {samples.shape}')

argparser = argparse.ArgumentParser()
argparser.add_argument('--type', type=str)
args = argparser.parse_args()

if args.type == 'p_title':
    # add titles of principal papers
    p_title = samples.shape[0] * [None]
    for i in trange(10):
        open_samples = [ind for ind in range(len(p_title)) if p_title[ind] == None]
        meta = pd.read_json(path_or_buf=f'../CS_papers/data/metadata_full{i}.jsonl', lines=True, dtype= {'paper_id': int})
        for j in tqdm(open_samples):
            sample_id = samples['principal_id'][j]
            meta_sample = meta[meta['paper_id'] == sample_id]
            if meta_sample.shape[0] > 0:
                assert meta_sample.shape[0] == 1, print(meta_sample)
                p_title[j] = meta_sample["title"].values[0]
        del meta
        gc.collect()
    print(f'Principal titles: {none_counter(p_title)} None values')
    with open('data/p_title.pkl', 'wb') as f:
        pickle.dump(p_title, f)


elif args.type == 'p_text':
    # add principal_abstract & principal_body
    p_abs = samples.shape[0] * [None]
    p_text = samples.shape[0] * [None]
    for i in trange(10):
        open_samples = [ind for ind in range(len(p_abs)) if (p_abs[ind] == None) and (p_text[ind] == None)]
        pdf = pd.read_json(path_or_buf=f'../CS_papers/pdf_parses_full{i}.jsonl', lines=True, dtype= {'paper_id': int})
        for j in tqdm(open_samples):
            sample_id = samples['principal_id'][j]
            pdf_sample = pdf[pdf['paper_id'] == sample_id]
            if pdf_sample.shape[0] > 0:
                assert pdf_sample.shape[0] == 1, print(pdf_sample)
                p_abs[j] = pdf_sample['abstract'].values[0]
                p_text[j] = pdf_sample['body_text'].values[0]
        del pdf
        gc.collect()
    print(f'Principal texts: {none_counter(p_abs, p_text)} None values')     
    with open('data/p_abs.pkl', 'wb') as f:
        pickle.dump(p_abs, f)
    with open('data/p_text.pkl', 'wb') as f:
        pickle.dump(p_text, f)       

elif args.type == 'c_title':
    # add titles of cited papers
    c_title = samples.shape[0] * [None]
    for i in trange(25):
        open_samples = [ind for ind in range(len(c_title)) if c_title[ind] == None]
        meta = pd.read_json(path_or_buf=f'../CS_papers/cited_papers/output/metadata_full_cited{i}.jsonl', lines=True, dtype= {'paper_id': int})
        for j in tqdm(open_samples):
            sample_id = samples['cited_id'][j]
            meta_sample = meta[meta['paper_id'] == sample_id]
            if meta_sample.shape[0] > 0:
                assert meta_sample.shape[0] == 1, print(meta_sample)
                c_title[j] = meta_sample["title"].values[0]
        del meta
        gc.collect()
    print(f'Cited titles: {none_counter(c_title)} None values')  
    with open('data/c_title.pkl', 'wb') as f:
        pickle.dump(c_title, f)

elif args.type == 'c_text':
    # add cited_abstract & cited_body
    c_abs = samples.shape[0] * [None]
    c_text = samples.shape[0] * [None]
    for i in trange(25):
        open_samples = [ind for ind in range(len(c_abs)) if (c_abs[ind] == None) and (c_text[ind] == None)]
        pdf = pd.read_json(path_or_buf=f'../CS_papers/cited_papers/output/pdf_parses_full_cited{i}.jsonl', lines=True, dtype= {'paper_id': int})
        for j in tqdm(open_samples):
            sample_id = samples['cited_id'][j]
            pdf_sample = pdf[pdf['paper_id'] == sample_id]
            if pdf_sample.shape[0] > 0:
                assert pdf_sample.shape[0] == 1, print(pdf_sample)
                c_abs[j] = pdf_sample['abstract'].values[0]
                c_text[j] = pdf_sample['body_text'].values[0]
        del pdf
        gc.collect()
    print(f'Cited texts: {none_counter(c_abs, c_text)} None values')
    with open('data/c_abs.pkl', 'wb') as f:
        pickle.dump(c_abs, f)
    with open('data/c_text.pkl', 'wb') as f:
        pickle.dump(c_text, f)

else:
    print('Please enter a valid type.')