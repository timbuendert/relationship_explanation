import pickle

with open(f'data/samples.pkl', 'rb') as f:
    samples = pickle.load(f)  

with open(f'data/labels.pkl', 'rb') as f:
    labels = pickle.load(f)

print(f'Number of labels loaded: {len([i for i in labels if i != -1])}')

# list of mapping from duplicate explanations to indices
exp = [s['explanation'] for s in samples]
rem_idx = [exp.index('(Simons et al. 2010 : example 10)'), exp.index('(Jiang, Xu, and Liang 2017)')]  # samples for which there is no cond_sum input


idx = 12513
print(f'Previous label: {labels[idx]}')

explanation = samples[idx]['explanation']
p_title = samples[idx]['principal_title']

l = ''
while l == '':
    if samples[idx]['discourse'].lower() == 'narrative_cite':
        l = int(input(f"\n{idx}: Enter the label (0: background, 1: comparison, -2: cannot tell // 2: single_summ, 100: break) [probably background]\nPrincipal title: {p_title}\n{explanation}\nLabel: "))
    elif samples[idx]['discourse'].lower() == 'reflection':
        l = int(input(f"\n{idx}: Enter the label (0: background, 1: comparison, -2: cannot tell // 2: single_summ, 100: break) [probably comparison]\nPrincipal title: {p_title}\n{explanation}\nLabel: "))
    elif samples[idx]['discourse'].lower() == 'single_summ':
        l = int(input(f"\n{idx}: Enter the label (0: background, 1: comparison, -2: cannot tell // 2: single_summ, 100: break) [single_summ]\nPrincipal title: {p_title}\n{explanation}\nLabel: "))
    else:
        l = int(input(f"\n{idx}: Enter the label (0: background, 1: comparison, -2: cannot tell // 2: single_summ, 100: break)\nPrincipal title: {p_title}\n{explanation}\nLabel: "))

labels[idx] = l  

with open(f'data/labels.pkl', 'wb') as f:
    pickle.dump(labels, f)

print(f'\nSuccessfully updated index {idx} to {labels[idx]}!')

# background: to understand tools/papers -> use generation model trained on narrative_cite 
# direct comparison: also extension -> use generation model trained on reflection