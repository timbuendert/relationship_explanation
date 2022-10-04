import pandas as pd
import pickle

# retrieve part of overall corpus
i = 0
df = pd.read_json(path_or_buf=f'../CS_papers/data/pdf_parses_full{i}.jsonl', lines=True)

# join and export all body texts
texts = [sample[j]['text'] for sample in df["body_text"] for j in range(len(sample))]
print(len(texts))

with open('texts.pkl', 'wb') as f:
    pickle.dump(texts, f)