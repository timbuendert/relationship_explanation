import pickle
import pandas as pd
import gc

explanations, discourses, principal_titles = [], [], []
for i in range(4):
    samples = pd.read_json(path_or_buf=f'../final_dataset/dataset_cs{i}.json')
    explanations += list(samples['explanation'])
    discourses += list(samples['discourse'])
    principal_titles += list(samples['principal_title'])
    del samples
    gc.collect()

print(f'Number of explanations: {len(explanations)}')
print(f'Number of discourses: {len(discourses)}')
print(f'Number of principal titles: {len(principal_titles)}')

dict_samples = [{'explanation': explanations[j], 'discourse': discourses[j], 'principal_title': principal_titles[j]} for j in range(len(explanations))]

with open(f'data/samples.pkl', 'wb') as f:
    pickle.dump(dict_samples, f)