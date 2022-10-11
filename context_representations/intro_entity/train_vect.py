from sklearn.feature_extraction.text import TfidfVectorizer
import json 
import pickle
import argparse
import spacy
from dygie.spacy_interface.spacy_interface import DygieppPipe

argparser = argparse.ArgumentParser()
argparser.add_argument('--intent', type=str)
args = argparser.parse_args()

def load_data(data_path):
    with open(data_path, "r") as f:
        for line in f:
            yield json.loads(line)

data = list(load_data(f'../../final_dataset/data/dataset_{args.intent}.jsonl'))
print('Successfully loaded data')


## Regular TFIDF-Vectorizer
alltexts = [' '.join([sample['cited_text'][i]['text'] for i in range(len(sample['cited_text']))]) for sample in data]
alltexts += [' '.join([sample['principal_text'][i]['text'] for i in range(len(sample['principal_text']))]) for sample in data]

tfidf = TfidfVectorizer(stop_words = 'english')
vect = tfidf.fit(alltexts)

print(vect)

with open('tfidf_vect.pkl', 'wb') as f:
    pickle.dump(vect, f)



## Entity-TFIDF-Vectorizer
all_cited_titles = [sample['cited_title'] for sample in data]
all_cited_abstracts = [' '.join([sample['cited_abstract'][i]['text'] for i in range(len(sample['cited_abstract']))]) for sample in data]

nlp = spacy.load('en_core_web_sm')
component = DygieppPipe(nlp,pretrained_filepath="scierc.tar.gz", dataset_name="scierc")
nlp.add_pipe(component)

all_entities = []
for abs, title in zip(all_cited_abstracts, all_cited_titles):
    text = f"Title: {title} Section: Abstract {abs}"
    doc = nlp(text)
    if len(list(doc.ents)) > 2:
        all_entities += [' '.join([str(i) for i in list(doc.ents)])]

tfidf_ent = TfidfVectorizer(stop_words = 'english')
vect_ent = tfidf_ent.fit(all_entities)

print(vect_ent)

with open('entity_vect.pkl', 'wb') as f:
    pickle.dump(vect_ent, f)
    
    