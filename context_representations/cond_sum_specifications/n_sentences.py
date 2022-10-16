import json
from nltk.tokenize import sent_tokenize
import numpy as np


def load_data(data_path):
    with open(data_path, "r") as f:
        for line in f:
            yield json.loads(line)   

def text2sentences(text: str):
    sentences = sent_tokenize(text)
    drop_idx = []
    drop_counter = 0
    
    for n, sent in enumerate(sentences):
        sent = sent.strip()
        if n == 0 and sent[0].islower():
            sent = sent[0].upper() + sent[1:]
            sentences[0] = sent
        if sent[0].islower() or sent[0] == '(' or len(sent) < 20:
            drop_counter += 1
            sentences[n-drop_counter] = sentences[n-drop_counter] + ' ' + sent
            drop_idx.append(n)
        else: 
            drop_counter = 0
    return [s for n,s in enumerate(sentences) if n not in drop_idx]

def check_section(section):
    rel_sections = ['introduction', 'conclusion']

    if section == '1':
        return True

    for s in rel_sections:
        if s in section:
            return True
    else:
        return False

#################################################################

sentence_lengths = []

for s in ['train', 'test', 'val']:
    samples = list(load_data(f'../final_dataset/data/reflection_{s}.jsonl'))

    src_tgt_pairs = []
    for cleaned_citance in samples:
        src_tgt_pairs.append({
            "citingPaperId": cleaned_citance["principal_id"],
            "citingTitle": cleaned_citance["principal_title"],
            "citingAbstract": cleaned_citance["principal_abstracts"],
            "citingBody": cleaned_citance["principal_text"],

            "citedPaperId": cleaned_citance["cited_id"],
            "citedTitle": cleaned_citance["cited_title"],
            "citedAbstract": cleaned_citance["cited_abstract"],
            "citedBody": cleaned_citance["cited_text"],

            "citation_text": cleaned_citance["explanation"].replace("[BOS] ", ''),
            "intent": cleaned_citance["discourse"]
        })


    for src_tgt_pair in src_tgt_pairs:
        doc1_abs = ' '.join([src_tgt_pair['citingAbstract'][i]['text'] for i in range(len(src_tgt_pair['citingAbstract']))])
        doc1_sec = ' '.join([src_tgt_pair['citingBody'][i]['text'] for i in range(len(src_tgt_pair['citingBody'])) if check_section(src_tgt_pair['citingBody'][i]['section'].lower())])
        doc1 = doc1_abs + ' ' + doc1_sec
        doc2 = ' '.join([src_tgt_pair['citedBody'][i]['text'] for i in range(len(src_tgt_pair['citedBody'])) if ('related' not in src_tgt_pair['citedBody'][i]['section'].lower()) or ('previous' not in src_tgt_pair['citedBody'][i]['section'].lower())])
        
        sentences1 = text2sentences(doc1) # sentences to choose from using cond_sum
        sentences2 = text2sentences(doc2) # -"-

        sentence_lengths.append(len(sentences1)+len(sentences2))

print(f'Average number of sentences: {np.mean(sentence_lengths)} ({len(sentence_lengths)} samples)')