import pandas as pd
import pickle
import numpy as np
from collections import Counter
import gc
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument("--filter", action='store_true')
argparser.add_argument('--intent', type=str)
args = argparser.parse_args()

# import samples
for i in range(4):
    if i == 0:
        samples = pd.read_json(path_or_buf=f'dataset_cs{i}.json')
        if args.filter:
            samples = samples[samples['discourse'] == args.intent].reset_index(drop = True)
    else:
        samples_i = pd.read_json(path_or_buf=f'dataset_cs{i}.json')
        if args.filter:
            samples_i = samples_i[samples_i['discourse'] == args.intent].reset_index(drop = True)
        samples = pd.concat([samples, samples_i], ignore_index=True)
        print(i, samples_i.shape)
        del samples_i
        gc.collect()

'''
# print random sampels from discourses
for d in list(set(samples['discourse'])):
    print('\n', d)
    samples_d = samples[samples['discourse'] == d].reset_index(drop = True)
    samples_d = samples_d.sample(n=10, random_state=42).reset_index(drop = True)
    for di in range(samples_d.shape[0]):
        print(samples_d['explanation'][di])
'''

if not args.filter:
    print(f'Full shape: {samples.shape}')

    c_discourses = Counter(samples['discourse'])
    c_spantype = Counter(samples['span_type'])

    # print distributions
    print([(i, c_discourses[i] / len(samples['discourse']) * 100.0, c_discourses[i]) for i, count in c_discourses.most_common(10)])
    print([(i, c_spantype[i] / len(samples['span_type']) * 100.0, c_spantype[i]) for i, count in c_spantype.most_common()], '\n')

    for discourse in list(set(samples['discourse'])):
        span_types_discourse = [samples['span_type'][j] for j in range(len(samples['discourse'])) if samples['discourse'][j] == discourse]
        counter_st = Counter(span_types_discourse)
        print(f'For {discourse}:')
        print([(i, counter_st[i] / len(span_types_discourse) * 100.0, counter_st[i]) for i, _ in counter_st.most_common()])


if args.filter: # Filter citations per intent

    output_name = str(args.intent).lower()
    
    samples.to_json(f'data/dataset_{output_name}.jsonl', orient = "records", lines=True)

    train_data, val_data, test_data = np.split(samples.sample(frac=1, random_state=42), [int(.6*samples.shape[0]), int(.8*samples.shape[0])])
    print(f'Train: {train_data.shape}') # 60%
    print(f'Val: {val_data.shape}') # 20%
    print(f'Test: {test_data.shape}') # 20%

    train_data.to_json(f'data/{output_name}_train.jsonl', orient = "records", lines=True)
    val_data.to_json(f'data/{output_name}_val.jsonl', orient = "records", lines=True)
    test_data.to_json(f'data/{output_name}_test.jsonl', orient = "records", lines=True)

    splits_idx = {'train':list(train_data.index),
                  'test':list(test_data.index),
                  'val':list(val_data.index)}

    with open(f'data/{output_name}_splits_idx.pkl', 'wb') as f:
        pickle.dump(splits_idx, f)