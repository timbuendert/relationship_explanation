import pickle
from collections import defaultdict, Counter

def list_duplicates(seq):
    tally = defaultdict(list)
    for i,item in enumerate(seq):
        tally[item].append(i)
    dup_dict = {}
    for key,locs in tally.items():
        if len(locs) > 1:
            dup_dict[key] = locs
    return dup_dict

def find_dup_label(idx):
    for k in dup_exp.keys():
        if idx in dup_exp[k]:
            original_idx = dup_exp[k][0]
    return labels[original_idx]

with open(f'data/samples.pkl', 'rb') as f:
    samples = pickle.load(f)  

try:
    with open(f'data/labels.pkl', 'rb') as f:
        labels = pickle.load(f)
except FileNotFoundError:
    labels = [-1] * len(samples)

print(f'Number of labels loaded: {len([i for i in labels if i != -1])}')

# list of mapping from duplicate explanations to indices
exp = [s['explanation'] for s in samples]
rem_idx = [exp.index('(Simons et al. 2010 : example 10)'), exp.index('(Jiang, Xu, and Liang 2017)')]  # samples for which there is no cond_sum input

dup_exp = list_duplicates(exp)
dup_exp_idx = [item for sublist in dup_exp.values() for item in sublist]
dup_ind_labeled = [item for sublist in [r[1:] for r in dup_exp.values()] for item in sublist] # list of indices whether label is already available

with open(f'data/samples_final.pkl', 'rb') as f:  # load all combined samples
    contexts = pickle.load(f)
dup_cont = list_duplicates(contexts)
dup_cont_idx = [item for sublist in dup_cont.values() for item in sublist]

####################################################
####################################################

method = 1 # narrative_cite and reflection only
#method = 2 # all other intents
#method = 3 # duplicate ones

# background: to understand tools/papers -> use generation model trained on narrative_cite 
# direct comparison: also extension -> use generation model trained on reflection

####################################################
####################################################

for i in range(len(samples)):
    if i in rem_idx or labels[i] != -1:
        continue

    if i in dup_ind_labeled:
        labels[i] = find_dup_label(i)
        continue

    explanation = samples[i]['explanation']
    p_title = samples[i]['principal_title']

    if method == 1:
        if not(samples[i]['discourse'] == 'Narrative_cite' or samples[i]['discourse'] == 'Reflection') or i in dup_cont_idx:
            continue

    elif method == 2:
        if samples[i]['discourse'] == 'Narrative_cite' or samples[i]['discourse'] == 'Reflection' or i in dup_cont_idx:
            continue

    if method == 1 or method == 2:
        l = ''
        while l == '':
            if samples[i]['discourse'].lower() == 'narrative_cite':
                l = int(input(f"\n{i}: Enter the label (0: background, 1: comparison, -2: cannot tell // 100: break) [probably background]\nPrincipal title: {p_title}\n{explanation}\nLabel: "))
            elif samples[i]['discourse'].lower() == 'reflection':
                l = int(input(f"\n{i}: Enter the label (0: background, 1: comparison, -2: cannot tell // 100: break) [probably comparison]\nPrincipal title: {p_title}\n{explanation}\nLabel: "))
            elif samples[i]['discourse'].lower() == 'single_summ':
                l = int(input(f"\n{i}: Enter the label (0: background, 1: comparison, -2: cannot tell // 100: break) [single_summ]\nPrincipal title: {p_title}\n{explanation}\nLabel: "))
            else:
                l = int(input(f"\n{i}: Enter the label (0: background, 1: comparison, -2: cannot tell // 100: break)\nPrincipal title: {p_title}\n{explanation}\nLabel: "))
        
        if l == 100:
            break

        labels[i] = l
    
    elif method == 3: # give duplicate indices one label by looking at all of them at once (probably comparison)
        if i not in dup_cont_idx:
            continue

        ind_sample = [l for l in list(dup_cont.values()) if i in l][0]
        print(f"\n{ind_sample}: Enter the label (0: background, 1: comparison, -1: cannot tell // 100: break)\nPrincipal title: {p_title}")
        for n, e in enumerate(ind_sample):
            print(f"{n}: {samples[e]['explanation']}")
        l = int(input("Label: "))

        if l == 100:
            break

        for e in ind_sample:
            labels[e] = l
        

with open(f'data/labels.pkl', 'wb') as f:
    pickle.dump(labels, f)

print(f'\nSaved {len([i for i in labels if i != -1])} labels ({len([i for i in labels if i > -1])} real ones)')
counter_l = Counter(labels)
print(counter_l)
