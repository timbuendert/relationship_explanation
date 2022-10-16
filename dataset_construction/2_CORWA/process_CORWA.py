import pandas as pd
import numpy as np
from tqdm import tqdm, trange
import copy

###############################################################################################
# Define functions

# process sample of CORWA into dataset samples
# - correct mistakes made by sentence-level analysis of corwa (e.g., "etc." viewed as end of sentence)
# - adjust discourse tags, citation spans, text ("[BOS]" appearances)

def process_sample(sample):
    text = sample['paragraph']
    discourse_tags = sample['discourse_tags']
    mapping = copy.deepcopy(sample['span_citation_mapping'])
                
    sentences = ['[BOS]'+e for e in text.split('[BOS]') if e] 
    
    drop_idx = []
    change_map = []
    is_etc = False
    count_drop = 1
    for n, sen in enumerate(sentences):
        # if 1 or 2 tokens in sentence -> merge with sentence before
        if sen[-5:] == 'etc. ' or sen[-5:] == 'etc.) ':
            is_etc = True
        if len(sen) < 25 or (is_etc and sen.split()[1][0].islower()):
            if n == 0:
                continue
            sentences[n-count_drop] = sentences[n-count_drop] + ' '.join([token for token in list(sen.split())[1:]])
            drop_idx.append(n)
            count_drop += 1
            change_map.append(text.index(sentences[n]))
            if is_etc:
                is_etc = False
        else:
            count_drop = 1
                
    sentences = [j.strip() for i, j in enumerate(sentences) if i not in drop_idx]
    
    text_new = ' '.join([s for s in sentences])
    discourse_tags_new = [t for i, t in enumerate(discourse_tags) if i not in drop_idx]
        
    assert len(sentences) == len(discourse_tags_new)
    assert len(sentences) == text_new.count('[BOS]')
    
    for i, j in enumerate(mapping):
        for c in change_map:
            if mapping[i]['char_start'] > c:
                mapping[i]['char_start'] -= 6
            if mapping[i]['char_end'] > c:
                mapping[i]['char_end'] -= 6
                
         
    return {'id': sample['id'], 
            'paragraph': text_new, 
            'discourse_tags': discourse_tags_new, 
            'span_citation_mapping': mapping, 
            'idx':sample['idx']}
    

# Find citations with their labels to produce final samples
# - if span ranges over several sentences: each sentence = one sample
# - only want citations of one paper and where paper id is present

def cited_papers(mapping):
    cited = []
    for s in mapping.values():
        cited += list(s.values())
    return list(set(cited))

def check_sentence(text, sentences, span_mapping):
    cited_sentences = []
    span_mapping_updated = copy.deepcopy(span_mapping)
    drop_idx = []

    char_ends = (np.cumsum([len(sen) for sen in sentences]))

    for n, span in enumerate(span_mapping):
        
        # only want citations with one cited ID
        if len(cited_papers(span['span_citation_mapping'])) != 1:
            continue
        
        if span['char_start'] == -1:
            citation = list(span['span_citation_mapping'][span['span_type']].keys())
            if len(citation) != 1:
                continue
            span['char_start'] = text.find(citation[0])


        cited_text = text[span['char_start']:span['char_end']]
        n_sentence = [i for i in range(len(sentences)) if cited_text in sentences[i]]
        
        # one citation with many sentences -> many citations with one sentence
        if (len(n_sentence) != 1) and (cited_text.count('[BOS]') > 0):
            citations = [e for e in cited_text.split('[BOS]') if e]
            
            assert cited_text.count('[BOS]'), print(cited_text)

            start_char = span['char_start']
            
            drop_idx.append(n)
            for k, c in enumerate(citations):
                cited_text_new = text[start_char+cited_text.find(c):start_char+cited_text.find(c)+ len(c)]
                n_sentence = [[i for i in range(len(char_ends)) if (char_ends > start_char+cited_text.find(c))[i]][0]]


                assert len(n_sentence) == 1, print(sentences)

                
                # if 1 or 2 tokens in sentence
                if len(c) < 30:
                    if (k == 0) and (len(sentences[n_sentence[0]]) > 40):
                        span_mapping_updated += [{'char_start': span['char_start'],
                                                  'char_end': span['char_start'] + len(c),
                                                  'span_type': span['span_type'],
                                                  'span_citation_mapping': span['span_citation_mapping']}]
                        cited_sentences.append(n_sentence[0])                    
                    
                    elif (len(n_sentence) == 1) and (len(sentences[n_sentence[0]]) > 40):
                        span_mapping_updated += [{'char_start': text.find(c),
                                                  'char_end': text.find(c) + len(c),
                                                  'span_type': span['span_type'],
                                                  'span_citation_mapping': span['span_citation_mapping']}]

                        cited_sentences.append(n_sentence[0])
                
                
                else:
                    assert not (len(cited_text_new.split('[BOS]')) > 1), print('here', cited_text_new)
                    #print(cited_text_new, cited_text, cited_text.count('[BOS]') >0, '\n',sentences,len(sentences), '\n', n_sentence, '\n', citations, '\n', span)
                        
                    span_mapping_updated += [{'char_start': text.find(c),
                                            'char_end': text.find(c) + len(c),
                                            'span_type': span['span_type'],
                                            'span_citation_mapping': span['span_citation_mapping']}]

                    cited_sentences.append(n_sentence[0])

        
        # e.g. cited span appears two times in text
        elif len(n_sentence) != 1:
            start_idx = [text.index(sentences[i]) for i in range(len(sentences))] # start indices of all sentences
            try:
                sentence_start_char = [s for s in start_idx if s < span['char_start']][-1] # start index of citing sentence
            except:
                print(span, text)
            n_sentence = start_idx.index(sentence_start_char) # number of sentence
            cited_sentences.append(n_sentence)
        
        else:
            cited_sentences.append(n_sentence[0])
    
    span_mapping_updated = [j for i, j in enumerate(span_mapping_updated) if i not in drop_idx]
    return [i for i in range(len(sentences)) if cited_sentences.count(i) == 1], span_mapping_updated
    
def get_citation_citedID(row):
    text = row['paragraph']
    span_mapping = row['span_citation_mapping']
    principal_id = str(row['id'])

    d = '[BOS]'
    sentences = [d+e for e in text.split(d) if e] 
    
    citations = []
    
    admissible_sent_idx, span_mapping = check_sentence(text, sentences, span_mapping)
    
    for n, span in enumerate(span_mapping):
        cited_text = text[span['char_start']:span['char_end']]
        span_type = span['span_type']
        for sent_idx in range(len(sentences)):
            if cited_text in sentences[sent_idx]:
                cited_ids = list(span['span_citation_mapping'][span_type].values())
                if sent_idx in admissible_sent_idx and len(cited_ids) == 1 and cited_ids[0] is not None: # ??
                    citations.append({'principal_id': principal_id,
                                      'explanation': sentences[sent_idx].replace('[BOS] ', ''), 
                                      'discourse': row['discourse_tags'][sent_idx], 
                                      'span_type': span_type,
                                      'cited_id': str(cited_ids[0])})

    return citations


###############################################################################################
# Execute for all tagged files

for i in trange(10):
    # load and preproces tagged samples
    data_tagged = pd.read_json(path_or_buf=f'../CORWA/related_works_tagged/tagged_related_works{i}.jsonl', lines=True, dtype= {'id': str})
    data_tagged[['id', 'idx']] = data_tagged['id'].str.split('_', 1, expand = True)
    data_tagged['id'] = data_tagged['id'].apply(lambda text: text.split('-', 1)[0])


    # ensure that each [BOS] tag is associated with a discourse tag
    for j in range(data_tagged.shape[0]):
        if data_tagged['paragraph'][j].count('[BOS]') != len(data_tagged['discourse_tags'][j]):
            del data_tagged['discourse_tags'][j][-10] # manually evaluated by analyzing example 93258 from file #7

    # apply function to all samples
    data_tagged_processed = pd.DataFrame([process_sample(data_tagged.iloc[k,:]) for k in range(data_tagged.shape[0])])
    data_tagged_processed.shape


    # apply function to all data samples
    outputs = []
    for l in tqdm(range(data_tagged_processed.shape[0])):
        if not data_tagged_processed['span_citation_mapping'][l]:
            continue  
        outputs += get_citation_citedID(data_tagged_processed.iloc[l,:])


    # combine and show final samples
    samples = pd.DataFrame(outputs)
    print(f'Shape of iteration {i}: {samples.shape}')

    if i == 0:
        samples_all = samples
    else:
        samples_all = pd.concat([samples_all, samples], ignore_index=True)

print(samples_all.head())
print(f'Final shape: {samples_all.shape}')

# export
samples_all.to_json(f'samples_all.jsonl', orient = "records", lines=True)