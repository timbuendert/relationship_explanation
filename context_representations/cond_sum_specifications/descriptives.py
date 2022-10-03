from nltk.tokenize import sent_tokenize
import numpy as np

def text2sentences(text: str):
    sentences = sent_tokenize(text)
    drop_idx = []
    drop_counter = 0
    
    for n, sent in enumerate(sentences):
        sent = sent.strip()
        if n == 0 and sent[0].islower():
            sent = sent[0].upper() + sent[1:]
            sentences[0] = sent
        if sent[0].islower() or sent[0] == '(':
            drop_counter += 1
            sentences[n-drop_counter] = sentences[n-drop_counter] + ' ' + sent
            drop_idx.append(n)
        else: 
            drop_counter = 0
    return [s for n,s in enumerate(sentences) if n not in drop_idx]

def retrieve_desc(lst_text):
    char_lengths = [len(i) for i in lst_text]
    word_lengths = [len(i.split(' ')) for i in lst_text]
    sen_lengths = [len(text2sentences(i)) for i in lst_text]

    print(f'{path}:\nAvg. character length = {np.mean(char_lengths)}\nAvg. word length = {np.mean(word_lengths)}\nAvg. sentence length = {np.mean(sen_lengths)}\n\n')
    pass

n_sentences = [1,2,5]
n_matches = [3,5,8]
title = ['_t','']

for s in n_sentences:
    for m in n_matches:
        for t in title:
            path = f'cond_sum_{s}_{m}{t}'

            examples = []
            examples += list(open(f'{path}/train.source').readlines())
            examples += list(open(f'{path}/val.source').readlines())
            examples += list(open(f'{path}/test.source').readlines())

            print('All:')
            retrieve_desc(examples)

            print('Principal Parts:')
            p_contexts = [i[:i.index(' <DOC_SEP>')] for i in examples]
            retrieve_desc(p_contexts)

            print('Cited Parts:')
            c_contexts = [i[i.index('<DOC_SEP>') + len('<DOC_SEP> '):] for i in examples]
            retrieve_desc(c_contexts)