import argparse

import json
from tqdm import tqdm
import numpy as np

import os
import sys

from util import *
from data_util import *
from joint_tagger import run_prediction

from transformers import AutoTokenizer

from collections import Counter

import nltk
nltk.download('punkt')
     
if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--repfile', type=str, default = "allenai/scibert_scivocab_uncased", help="Word embedding file")
    argparser.add_argument('--related_work_file', type=str) # "20200705v1/acl/related_work.jsonl"
    argparser.add_argument('--output_file', type=str) # "tagged_related_works.jsonl"
    argparser.add_argument('--dropout', type=float, default=0, help="embedding_dropout rate")
    argparser.add_argument('--bert_dim', type=int, default=768, help="bert_dimension")
    argparser.add_argument('--MAX_SENT_LEN', type=int, default=512)
    argparser.add_argument('--checkpoint', type=str, default = "joint_tagger_train_scibert_final.model")
    argparser.add_argument('--batch_size', type=int, default=32) # roberta-large: 2; bert: 8
    argparser.add_argument('--intent', type=str)
    argparser.add_argument('--model', type=str)
    argparser.add_argument('--context', type=str)
    argparser.add_argument("--cond_sum_analysis", action='store_true')
    args = argparser.parse_args()
        
    if not sys.warnoptions:
        import warnings
        warnings.simplefilter("ignore")
    
    tokenizer = AutoTokenizer.from_pretrained(args.repfile)
    additional_special_tokens = {'additional_special_tokens': ['[BOS]']}
    tokenizer.add_special_tokens(additional_special_tokens)

    if not args.intent:
        related_work_jsons = read_related_work_jsons(args.related_work_file)
        print(len(related_work_jsons))
            
        paragraphs = {}
        for paper_id, related_work_dict in related_work_jsons.items():
            for pi, para in enumerate(related_work_dict["related_work"]):
                paragraph_id = paper_id + "_" + str(pi)
                paragraphs[paragraph_id] = " ".join(scientific_sent_tokenize(para["text"]))

        discourse_predictions, citation_predictions, span_predictions, dataset = run_prediction(paragraphs, tokenizer, args) # for each paragraph of all papers

        all_span_citation_mappings = annotate_related_work(discourse_predictions, citation_predictions, span_predictions, dataset, related_work_jsons, tokenizer)

    #    print(len(all_span_citation_mappings))
        
        with open(args.output_file,"w") as f:
            for mapping in all_span_citation_mappings:
                json.dump(mapping,f)
                f.write("\n")
        
    else:
        if args.model == 'BART':
            path = f"../BART/experiments/bart_{args.intent}_{args.context}/test.output"
        elif args.model == 'SciGPT2':
            path = f"../SciGen/outputs/eval_{args.intent}_{args.context}.outputs" # _scigpt
        elif args.model == 'SciGen':
            path = f"../SciGen/outputs/scigen_{args.intent}_{args.context}.outputs"
        elif args.model == 'OPT':
            path = f"../OPT/outputs/eval_{args.intent}_{args.context}.outputs" # _opt
        elif args.model == 'PEGASUS':
            path = f"../PEGASUS/experiments/pegasus_{args.intent}_{args.context}/test.output"

        if args.cond_sum_analysis:
            path = f"outputs/opt_{args.model}.outputs" 
        
        if args.intent == 'single_summ':
            if not os.path.isfile(path):
                path = path.replace('_single_summ','')

        with open(path, "r") as data:
            outputs = ['[BOS] ' + line.replace('\n','') for line in data] #.replace('<|CITE|>', '#REF').replace('<|PAD|>', '#REF')

        discourse_predictions, _, _, dataset = run_prediction(outputs, tokenizer, args) # for each paragraph of all papers
        for i in range(len(discourse_predictions)):
            if len(discourse_predictions[i]) != 1:
                print(len(discourse_predictions[i]))
        discourse_predictions = [d[0] for d in discourse_predictions]
        print(f'Length dataset = {len(dataset)}')
        print(f'Length discourse predictions = {len(discourse_predictions)}')
        
        results = [1 if discourse_predictions[i].lower() == args.intent else 0 for i in range(len(discourse_predictions))]
        acc = sum(results)/len(results)
        for n, r in enumerate(results):
            if r == 0:
                print(f'\n{discourse_predictions[n]}:\n{dataset[n]}\n')

        c_discourses = Counter(discourse_predictions)
        print([(i, c_discourses[i] / len(discourse_predictions) * 100.0, c_discourses[i]) for i, count in c_discourses.most_common(10)])
        
        print(f'\n\nIntent accuracy: {acc}')

        


# /Users/timbundert/opt/anaconda3/envs/py_env/bin/python3 /Users/timbundert/Desktop/pipeline/pipeline.py --related_work_file='related_work.jsonl' --output_file='tagged_related_works.jsonl' --batch_size=4
