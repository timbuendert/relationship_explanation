import os
import re
import math
import json
from sklearn.feature_extraction.text import TfidfVectorizer
#import spacy
#from intro_entity.dygie.spacy_interface.spacy_interface import DygieppPipe

#import scispacy


#nlp = spacy.load('en_core_sci_lg')


def load_data(data_path):
    with open(data_path, "r") as f:
        for line in f:
            yield json.loads(line)


def process_citation_with_cite_pattern(text, cite_pattern):
    if text.count('et al.') > 10:
        tokens = text.split(';')
        texts_new = []
        for t in tokens:
            reference_sign_matches = re.finditer(cite_pattern, t)
            for reference_sign_match in reference_sign_matches:
                original_reference_sign = reference_sign_match.group(0)
                t = t.replace(original_reference_sign, "#REF")
            texts_new.append(t)
        return ';'.join(texts_new)

    else:
        reference_sign_matches = re.finditer(cite_pattern, text)
        for reference_sign_match in reference_sign_matches:
            original_reference_sign = reference_sign_match.group(0)
            text = text.replace(original_reference_sign, "#REF")
        return text

def process_apa_citation(text):
    cite_pattern = "(([^\(\)]+,?\d{2,4}[a-z]?))"
    apa_incitation_pattern = f"\((\s*)?{cite_pattern}(; {cite_pattern})*(\s*)?\)"
        
    text = process_citation_with_cite_pattern(
        text, apa_incitation_pattern)
    return text


def process_ieee_citation(text):
    cite_pattern = "((\d{1,}(-\d{1,})*[a-z]?)|(#REF))"
    ieee_incitation_pattern = f"\[{cite_pattern}(,\s?{cite_pattern})*\]"
    text = process_citation_with_cite_pattern(
        text, ieee_incitation_pattern)
    return text


def process_reference_marker(text):

    #bad_string = '(Boroditsky, 2000; Singer et al., 2004 Singer et al., , 2006 Aziz-Zadeh et al., 2006; Gibbs, 2006; Wilson and Gibbs, 2007; Casasanto, 2008; Boulenger et al., 2009; IJzerman and Semin, 2009; Schubert and Koole, 2009; Landau et al., 2010; Sapolsky, 2010; Desai et al., 2011; Schwarz, 2011, 2012; Saygin et al., 2011; Fay and Maner, 2012; Mattingly and Lewandowski, 2013; Pitts et al., 2013; Deckman et al., 2014; Galinsky et al., 2014; Knowles et al., 2014; Masicampo and Ambady, 2014; Sassenrath et al., 2014; Schoel et al., 2014; Slepian et al., 2014; Stellar and Willer, 2014 '
    #if bad_string in text:
    #    text = text.replace(bad_string, "#REF")

    # Handle IEEE citation
    processed_text = process_ieee_citation(text)

    # Handle APA citation
    processed_text = process_apa_citation(processed_text)

    
    tokens = processed_text.split()

    drop_idx = []
    
    for n,t in enumerate(tokens):
        if '#REF' in t:
            if 'al' in tokens[n-1].lower():
                drop_idx += [n-1, n-2, n-3]
            elif 'al' in tokens[n].lower():
                tokens[n] = '#REF'
                drop_idx += [n-1, n-2]

            elif tokens[n-1][0].isupper() and tokens[n-2].lower() == 'and':
                drop_idx += [n-1, n-2, n-3]          

            elif tokens[n-1] == '#REF':
                drop_idx += [n]

        if ('al.' in t) and ('et' in tokens[n-1]):
            tokens[n] = '#REF'
            drop_idx += [n-1, n-2]
    
    processed_text = ' '.join([t for i,t in enumerate(tokens) if i not in drop_idx]).strip()

    return processed_text


# preprocess citing or cited input content


def process_source_input(text):
    text = text.replace(
        '\n', ' ').replace("\r", " ").strip().lower()

    return text


def process_seciton_name(section_name):
    section_name = str(section_name).lower()
    section_name = re.sub(
        "(\d+\.?)+\d?[\s\t]+(.+)", r"\2", section_name)
    return section_name


def get_citing_paper_input(data_pair, citing_input_mode='abstract'):
    if citing_input_mode == "title":
        return_input = data_pair["citingTitle"]
    elif citing_input_mode == "abstract":
        return_input = data_pair["citingAbstract"]
    else:
        raise TypeError("Unknown input mode")
    return return_input


def is_valid_citation_text(text):
    # sentence starts with "...", it's an incomplete sentence
    invalid_charaters = (b'\xe2\x80\xa6'.decode(), ",")
    return not text.startswith(invalid_charaters)

def list2text(string, cat):
    return ' '.join([string[cat][i]['text'] for i in range(len(string[cat]))])

#######################################################################################################################################

import pandas as pd
from nltk.tokenize import sent_tokenize
import numpy as np
import torch
from sentence_transformers import util
#from thefuzz import process
from sentence_transformers import SentenceTransformer, models

############### Helper functions 

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
    

def process_output_comp(sel_sent, doc):
    sentences = text2sentences(doc)
    sel_idx = []
    for s_s in sel_sent:
        idx = []
        for n, s in enumerate(sentences):
            if s in s_s:
                idx.append(n)
        assert len(idx) < 2, print(len(idx), s_s, sentences)
        sel_idx += idx

    sel_sentences = [s for n,s in enumerate(sentences) if n in sel_idx]
    return ' '.join(sel_sentences)

def process_output(sentences, max_idx, n_sentences): # input sentences1 or ?max_indices 
    #sentences = text2sentences(doc)
    n_s = len(sentences)

    all_seq = [list(range(a,a+n_sentences)) for a in range(n_s-(n_sentences-1) +1)]
    sel_idx = []
    for j in max_idx:
        try:
            sel_idx+=all_seq[j]
        except IndexError:
            print(j, all_seq, n_s)
    sel_idx = list(set(sel_idx))
    #print(sel_idx)

    sel_sentences = [s for n,s in enumerate(sentences) if n in sel_idx]

    #for n,s in enumerate(sentences):
    #    if n in sel_idx:
    #        print(s, '\n')
    
    #print(' '.join(sel_sentences))

    return ' '.join(sel_sentences)

def check_token(token, exst_tokens):
    if exst_tokens:
        if token in exst_tokens:
            return False
        else:
            if token.isdigit():
                return False
            try:
                token.encode(encoding='utf-8').decode('ascii')
            except UnicodeDecodeError:
                return False
            else:
                return True

    else:
        if token.isdigit():
            return False

        try:
            token.encode(encoding='utf-8').decode('ascii')
        except UnicodeDecodeError:
            return False
        else:
            return True

    
def get_top_tf_idf_words(response, feature_names, top_n=2, exst_tokens = None):
    #sorted_nzs = np.argsort(response.data)[:-(top_n+100):-1]
    sorted_nzs = np.argsort(response.data)[::-1]
    rel_tokens = list(feature_names[response.indices[sorted_nzs]])
    tokens = [t for t in rel_tokens if check_token(t, exst_tokens)][:top_n] #100
    return tokens

def check_section(section):
    rel_sections = ['introduction', 'conclusion']

    if section == '1':
        return True

    for s in rel_sections:
        if s in section:
            return True
    else:
        return False

############### Main function

def cond_summaries(doc1: str, doc2: str, model, n_sentences = 1, n_matches = 1, agg_mode = None, semantic_search: bool = False):
    sentences1 = text2sentences(doc1)
    sentences2 = text2sentences(doc2)

    if n_sentences != 1:
        sentences1 = [' '.join(sentences1[n:n + n_sentences]) for n in range(len(sentences1))]
        sentences2 = [' '.join(sentences2[n:n + n_sentences]) for n in range(len(sentences2))]

    n_matches = min(min(len(sentences1), len(sentences2)), n_matches) # min((len(sentences1) * len(sentences2)), n_matches) ?
    
    embeddings1 = model.encode(sentences1, convert_to_tensor=True)
    embeddings2 = model.encode(sentences2, convert_to_tensor=True)
 
    cosine_scores = util.cos_sim(embeddings1, embeddings2) # or via normalizing vetors & dot_product -> more efficient?
    # print(cosine_scores)
    
    if agg_mode is not None: # or average/sum over all cosine similarities of other doc with this one sentence?
        # worth experimenting
        if agg_mode == 'mean':
            agg_doc1 = torch.mean(cosine_scores, dim = 1) # or sum
            agg_doc2 = torch.mean(cosine_scores, dim = 0) # or sum
        elif agg_mode == 'sum':
            agg_doc1 = torch.sum(cosine_scores, dim = 1)
            agg_doc2 = torch.sum(cosine_scores, dim = 0)
            
        _, idx_1 = torch.topk(agg_doc1, n_matches)
        _, idx_2 = torch.topk(agg_doc2, n_matches)
        
        #sel_sent1 = [sentences1[idx_1[j]] for j in range(len(idx_1))]
        #sel_sent2 = [sentences2[idx_2[k]] for k in range(len(idx_2))]
    
    
    elif semantic_search == True:
        # semantic search (https://www.youtube.com/watch?v=ewlCCB7EFPs; https://www.sbert.net/examples/applications/semantic-search/README.html)
        #embeddings1 = embeddings1.to('cuda')
        embeddings1 = util.normalize_embeddings(embeddings1)
        #embeddings2 = embeddings2.to('cuda')
        embeddings2 = util.normalize_embeddings(embeddings2)
        
        # can add following parameters: query_chunk_size, corpus_chunk_size, score_function
        hits = util.semantic_search(embeddings1, embeddings2, score_function=util.dot_score, top_k = n_matches)
        
        scores = [hits[j][0]['score'] for j in range(len(hits))]
        sort_indices = np.array(scores).argsort().tolist()[::-1]
        idx_1 = sort_indices[:n_matches]
        idx_2 = [hits[i][0]['corpus_id'] for i in idx_1]
        
        #sel_sent1 = [sentences1[idx_1[j]] for j in range(len(idx_1))]
        #sel_sent2 = [sentences2[idx_2[k]] for k in range(len(idx_2))]
            
    
    else:
        _, i = torch.topk(cosine_scores.flatten(), n_matches) # or see: https://www.sbert.net/docs/usage/semantic_textual_similarity.html
        max_indices = np.array(np.unravel_index(i.cpu().numpy(), cosine_scores.shape)).T        

        # for case n = 1: 
        #max_value = torch.amax(cosine_scores)
        #max_indices = (cosine_scores == max_value).nonzero(as_tuple=True)
                        
        sel_sent1 = [sentences1[max_indices[j,0]] for j in range(max_indices.shape[0])]
        sel_sent2 = [sentences2[max_indices[k,1]] for k in range(max_indices.shape[0])]

        idx_1 = max_indices[:,0]
        idx_2 = max_indices[:,1]
            

    cond_sent1 = process_output(text2sentences(doc1), idx_1, n_sentences) #doc1
    cond_sent2 = process_output(text2sentences(doc2), idx_2, n_sentences)

    
    return cond_sent1, cond_sent2


def get_context_input(data, model, tfidf, entity, strategy: str, n_match = 1, n_sentence = 1, title = False):

    if strategy == 'intro_abs':
        #sections_doc1 = list(set([data['citingBody'][i]['section'] for i in range(len(data['citingBody']))]))
        #intro_section = sections_doc1[0]
        #for s in sections_doc1:
        #    if 'intro' in s.lower():
        #        intro_section = s

        # or: intro_text = [ item for item in js_dict['grobid_parse']['body_text'] if item["section"] and "intro" in item["section"].lower() ]
        sen1 = ' '.join([data['citingBody'][i]['text'] for i in range(len(data['citingBody'])) if 'introduction' in data['citingBody'][i]['section'].lower()])
        #sen1 = ' '.join([data['citingBody'][i]['text'] for i in range(len(data['citingBody'])) if data['citingBody'][i]['section'] == intro_section])

        sen2 = ' '.join([data['citedAbstract'][i]['text'] for i in range(len(data['citedAbstract']))])

    elif strategy == 'title_abs':
        sen1 = data['citingTitle']
        sen2 = ' '.join([data['citedAbstract'][i]['text'] for i in range(len(data['citedAbstract']))])

    elif strategy == 'intro_tfidf':
        #sections_doc1 = list(set([data['citingBody'][i]['section'] for i in range(len(data['citingBody']))]))
        #intro_section = sections_doc1[0]
        #for s in sections_doc1:
        #    if 'intro' in s.lower():
        #        intro_section = s

        sen1 = ' '.join([data['citingBody'][i]['text'] for i in range(len(data['citingBody'])) if 'introduction' in data['citingBody'][i]['section'].lower()])
        #sen1 = ' '.join([data['citingBody'][i]['text'] for i in range(len(data['citingBody'])) if data['citingBody'][i]['section'] == intro_section])

        sen2_alltext = [' '.join([data['citedBody'][i]['text'] for i in range(len(data['citedBody']))])]
        tfidf_sen2 = tfidf.transform(sen2_alltext)
        feature_names = np.array(list(tfidf.get_feature_names_out()))
        sen2 = get_top_tf_idf_words(tfidf_sen2, feature_names, 100)
        #assert len(sen2) == 100, print(len(sen2))
        if len(sen2) < 100:
            print(f'Only {len(sen2)} TFIDF tokens')
        for b in range(1, len(sen2)*2-1, 2): 
            sen2.insert(b,'<TFIDF>')

        sen2 = ' '.join(sen2)
        

    elif strategy == 'intro_entity':
        sen1 = ' '.join([data['citingBody'][i]['text'] for i in range(len(data['citingBody'])) if 'introduction' in data['citingBody'][i]['section'].lower()])

        doc2_title = data['citedTitle']
        doc2_abs = ' '.join([data['citedAbstract'][i]['text'] for i in range(len(data['citedAbstract']))])
        text = f"Title: {doc2_title} Section: Abstract {doc2_abs}"

        nlp = spacy.load('en_core_web_sm')
        component = DygieppPipe(nlp,pretrained_filepath="scierc.tar.gz", dataset_name="scierc")
        nlp.add_pipe(component)
        doc = nlp(text)
        ent = [' '.join([str(en) for en in list(doc.ents)])]

        ent_sen2 = entity.transform(ent)
        feature_names = np.array(list(entity.get_feature_names()))
        sen2 = get_top_tf_idf_words(ent_sen2, feature_names, 100)
        #print(sen2, len(sen2))
        n = len(sen2)

        if n < 100:
            needed_tfidf_tokens = (100 - n)
            #print(f'Need another {needed_tfidf_tokens} TFIDF tokens')

            doc2_alltext = [' '.join([data['citedBody'][i]['text'] for i in range(len(data['citedBody']))])]
            tfidf_sen2 = tfidf.transform(doc2_alltext)
            feature_names_tfidf = np.array(list(tfidf.get_feature_names()))
            sen2_tfidf = get_top_tf_idf_words(tfidf_sen2, feature_names_tfidf, needed_tfidf_tokens, sen2)

            for b in range(1, n*2-1, 2): 
                sen2.insert(b,'<ENT>')
            for d in range(0, len(sen2_tfidf)*2-1, 2): 
                sen2_tfidf.insert(d,'<TFIDF>')
            sen2 += sen2_tfidf

        else:
            for b in range(1, n*2-1, 2): 
                sen2.insert(b,'<ENT>')

        sen2 = ' '.join(sen2)


    elif strategy == 'cond_sum':

        doc1_abs = ' '.join([data['citingAbstract'][i]['text'] for i in range(len(data['citingAbstract']))])
        doc1_sec = ' '.join([data['citingBody'][i]['text'] for i in range(len(data['citingBody'])) if check_section(data['citingBody'][i]['section'].lower())])
        doc1 = doc1_abs + ' ' + doc1_sec
        doc2 = ' '.join([data['citedBody'][i]['text'] for i in range(len(data['citedBody'])) if ('related' not in data['citedBody'][i]['section'].lower()) and ('previous' not in data['citedBody'][i]['section'].lower())])

        sen1, sen2 = cond_summaries(doc1, doc2, model = model, n_matches = n_match, n_sentences = n_sentence, semantic_search = False) # agg_mode = 'sum',
        #print(sen1, sen2)

        if title:
            doc1_title = data['citingTitle']
            doc2_title = data['citedTitle']
            sen1 = '<PT> ' + doc1_title + ' <PC> ' + sen1
            sen2 = '<CT> ' + doc2_title + ' <CC> ' + sen2
            #print(sen1)
            #print(sen2)


    # as in https://github.com/Kel-Lu/SciGen/blob/master/data_processing/make_datafiles_from_dicts.py
    if len(sen1.split()) > 500:
        sen1 = ' '.join(sen1.split()[:500])
    if len(sen2.split()) > 500:
        sen2 = ' '.join(sen2.split()[:500])

    return sen1, sen2

