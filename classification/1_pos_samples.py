import pickle
import gc
import pandas as pd

# load all data
for i in range(4):
    if i == 0:
        samples = pd.read_json(path_or_buf=f'../final_dataset/dataset_cs{i}.json')
    else:
        samples_i = pd.read_json(path_or_buf=f'../final_dataset/dataset_cs{i}.json')
        samples = pd.concat([samples, samples_i], ignore_index=True)
        del samples_i
        gc.collect()

print(f'All samples: {samples.shape}')

# load samples
with open(f'data/labels.pkl', 'rb') as f:
    labels = pickle.load(f)

ind_samples = [i for i in range(len(labels)) if labels[i] > -1]
print(f'Labels of dataset: {len(ind_samples)}')

# get samples where positive relationship is present
samples_dataset = samples.iloc[ind_samples]
print(f'Final dataset: {samples_dataset.shape}')

samples_dataset.to_pickle("data/pos_samples.pkl")
