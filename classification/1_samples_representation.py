import argparse
import pandas as pd
import os
import logging
from data_utils import *
import pickle

logger = logging.getLogger(__name__)


class DataPreprocessor:
    def __init__(self, samples, n_start, n_end):
        self._citances = samples
        print('Successfully loaded {} samples to be processed!'.format(self._citances.shape))

    def get_src_tgt_pairs(self):
        src_tgt_pairs = []

        for i in range(self._citances.shape[0]):

            cleaned_citance = self._citances.iloc[i,:]

            src_tgt_pairs.append({
                "citingPaperId": cleaned_citance["principal_id"],
                "citingTitle": cleaned_citance["principal_title"],
                "citingAbstract": cleaned_citance["principal_abstracts"],
                "citingBody": cleaned_citance["principal_text"],

                "citedPaperId": cleaned_citance["cited_id"],
                "citedTitle": cleaned_citance["cited_title"],
                "citedAbstract": cleaned_citance["cited_abstract"],
                "citedBody": cleaned_citance["cited_text"],
            })

        return src_tgt_pairs


    def _preprocess(self, src_tgt_pairs, intent=None,
                    prepend_token=False, context_input_mode='cond_sum', context_n_sentences = 2, context_n_matches = 2, title = False, cs_model = 'SciBERT'):
        preprocessd_src_tgt_pairs = []

        if context_input_mode == 'cond_sum':
            if cs_model == '../cs_BERT/SentenceCSBert/':
                model_cs = SentenceTransformer(cs_model)
                
            else:
                if cs_model == 'SciBERT':
                    word_embedding_model = models.Transformer('allenai/scibert_scivocab_uncased', max_seq_length=512)
                else:
                    word_embedding_model = models.Transformer(cs_model, max_seq_length=512)

                pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
                model_cs = SentenceTransformer(modules=[word_embedding_model, pooling_model])
                
            tfidf = 0
            entity = 0


        elif context_input_mode == 'intro_entity':

            with open("../contexts_single_summ/intro_entity/tfidf_vect.pkl", 'rb') as f:
                tfidf = pickle.load(f)      
            
            with open("../contexts_single_summ/intro_entity/entity_vect.pkl", 'rb') as f:
                entity = pickle.load(f)   

            model_cs = 0

        else:
            tfidf = 0
            entity = 0
            model_cs = 0


        for i, src_tgt_pair in enumerate(src_tgt_pairs):
                
            ## get contexts based on selected mode
            citing_src, cited_src = get_context_input(src_tgt_pair, model_cs, tfidf, entity, context_input_mode, n_match = context_n_matches, n_sentence = context_n_sentences, title = title)
            
            # put #REF instead of reference
            citing_src = process_reference_marker(citing_src).strip()
            cited_src = process_reference_marker(cited_src).strip()
            
            src = citing_src + " <DOC_SEP> " + cited_src

            control_codes = ""

            # prepend citation intent token
            if prepend_token:
                control_codes += f'@{src_tgt_pair["intent"]} '
                src = f"{control_codes} {src}"


            if src != "":
                preprocessd_src_tgt_pairs.append({
                    "citingPaperId": src_tgt_pair["citingPaperId"],
                    "citedPaperId": src_tgt_pair["citedPaperId"],
                    "citingTitle": src_tgt_pair["citingTitle"],
                    "citingAbstract": src_tgt_pair["citingAbstract"],
                    "citedTitle": src_tgt_pair["citedTitle"],
                    "citedAbstract": src_tgt_pair["citedAbstract"],
                    "model_src": src,
                })
            else:
                print(i, src)

        return preprocessd_src_tgt_pairs

    def get_preprocessed_src_tgt_pairs(self, intent=None, prepend_token=False,
                                       context_input_mode='cond_sum', context_n_sentences = 2, context_n_matches = 2, title = False, cs_model = 'SciBERT'):
        return self._preprocess(self.get_src_tgt_pairs(), intent=intent, prepend_token=prepend_token,
                                context_input_mode=context_input_mode, context_n_sentences = context_n_sentences, context_n_matches = context_n_matches, title = title, cs_model = cs_model)

########### EXPORT

    def _export_to_transformers_input_file(self, dataset, dataset_type,
                                           out_dir='data/preprocessed'):
        data_path = os.path.join(out_dir, f'{dataset_type}_samples')

        with open(f"{data_path}.source", "w") as f_src:
            for data in dataset:
                f_src.write(data["model_src"])
                f_src.write('\n')

    def export_to_transformers_input_file(self, dataset_type, out_dir='data/preprocessed',
                                          intent=None, prepend_token=False, context_input_mode='cond_sum', context_n_sentences = 2, context_n_matches = 2,  title = None, cs_model = 'SciBERT'):
        dataset = self.get_preprocessed_src_tgt_pairs( 
            intent=intent, prepend_token=prepend_token, context_input_mode=context_input_mode, context_n_sentences = context_n_sentences, context_n_matches = context_n_matches,  title = title, cs_model = cs_model)


        if (not os.path.isdir(out_dir)) and (out_dir != ''):
            os.mkdir(out_dir)

        self._export_to_transformers_input_file(
            dataset, dataset_type, out_dir=out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str)
    parser.add_argument("--context", type=str)
    parser.add_argument("--cs_model", type=str)
    parser.add_argument("--n_sentences", type=int)
    parser.add_argument("--n_matches", type=int)
    parser.add_argument("--title", action='store_true')
    args = parser.parse_args()

    if args.context == 'cond_sum':
        if args.title:
            tgt_directory = f'cond_sum_{args.n_sentences}_{args.n_matches}_t'
        else:
            tgt_directory = f'cond_sum_{args.n_sentences}_{args.n_matches}'
    else:
        tgt_directory = f'{args.context}'
    os.makedirs(f'data/{tgt_directory}', exist_ok=True)

    if args.context == 'intro_entity':
        if args.split == 'pos':
            samples = pd.read_json('data/pos_samples_ie.json', orient='records')
        elif args.split == 'neg':
            samples = pd.read_json('data/neg_samples_ie.json', orient='records')
        else:
            raise ValueError
    else:
        if args.split == 'pos':
            samples = pd.read_pickle("data/pos_samples.pkl")
        elif args.split == 'neg':
            samples = pd.read_pickle("data/neg_samples.pkl")
        else:
            raise ValueError


    print(f'Samples to be processed: {samples.shape[0]}')
    
    dataset_processed = DataPreprocessor(samples, 0, samples.shape[0]+10)

    dataset_processed.export_to_transformers_input_file(
        args.split, out_dir=f'data/{tgt_directory}',
        prepend_token=False, context_input_mode=args.context,
        context_n_sentences=args.n_sentences, context_n_matches=args.n_matches, title = args.title, cs_model=args.cs_model)