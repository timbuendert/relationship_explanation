import os
import json
import logging
import argparse
from data_utils import *
from tqdm import tqdm
import pickle

logger = logging.getLogger(__name__)


class DataPreprocessor:
    def __init__(self,
                 file_path, n_start, n_end):
        self._citances = list(load_data(file_path))

        if (n_end > len(self._citances)) or (n_end == -1):
            n_end = len(self._citances)

        self._citances = self._citances[n_start:n_end]
        print('Successfully loaded {} samples to be processed (from {} to {})!'.format(len(self._citances), n_start, n_end))

    def get_src_tgt_pairs(self):
        src_tgt_pairs = []

        for cleaned_citance in self._citances:

            # filter out data without abstract
            #if cleaned_citance["cited_abstract"][0]['text'].lower() not in ("", "abstract"):
            src_tgt_pairs.append({
                "citingPaperId": cleaned_citance["principal_id"],
                "citingTitle": cleaned_citance["principal_title"],
                "citingAbstract": cleaned_citance["principal_abstracts"], #list2text(cleaned_citance["principal_abstracts"], 'principal_abstracts'),
                "citingBody": cleaned_citance["principal_text"], #list2text(cleaned_citance["principal_text"], 'principal_text'),

                "citedPaperId": cleaned_citance["cited_id"],
                "citedTitle": cleaned_citance["cited_title"],
                "citedAbstract": cleaned_citance["cited_abstract"], #list2text(cleaned_citance["cited_abstract"], 'cited_abstract'),
                "citedBody": cleaned_citance["cited_text"], #list2text(cleaned_citance["cited_text"], 'cited_text'),

                "citation_text": cleaned_citance["explanation"].replace("[BOS] ", ''),
                "intent": cleaned_citance["discourse"]
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
                    word_embedding_model = models.Transformer('allenai/scibert_scivocab_uncased', max_seq_length=512) #256
                else:
                    word_embedding_model = models.Transformer(cs_model, max_seq_length=512) #256

                pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
                model_cs = SentenceTransformer(modules=[word_embedding_model, pooling_model])

            tfidf = 0
            entity = 0

        elif context_input_mode == 'intro_tfidf':

            with open("tfidf_vect.pkl", 'rb') as f:
                tfidf = pickle.load(f)      
            
            # ngram_range = (1,1) -> only unigrams
            #tfidf = TfidfVectorizer(stop_words = 'english')
            #alltexts = [' '.join([sample['citedBody'][i]['text'] for i in range(len(sample['citedBody']))]) for sample in src_tgt_pairs]
            #alltexts += [' '.join([sample['citingBody'][i]['text'] for i in range(len(sample['citingBody']))]) for sample in src_tgt_pairs]
            #X = tfidf.fit_transform(alltexts)
            
            entity = 0
            model_cs = 0

        elif context_input_mode == 'intro_entity':

            with open("tfidf_vect.pkl", 'rb') as f:
                tfidf = pickle.load(f)      
            
            with open("entity_vect.pkl", 'rb') as f:
                entity = pickle.load(f)   

            model_cs = 0

        else:
            tfidf = 0
            entity = 0
            model_cs = 0

        for src_tgt_pair in tqdm(src_tgt_pairs):
            # Check if valid citation text
            #if is_valid_citation_text(src_tgt_pair["citation_text"]):
                #if intent is not None and intent != src_tgt_pair["intent"]:
                #    continue
                
            ## get contexts based on selected mode
            citing_src, cited_src = get_context_input(src_tgt_pair, model_cs, tfidf, entity, context_input_mode, n_match = context_n_matches, n_sentence = context_n_sentences, title = title)
            
            # put #REF instead of reference
            citing_src = process_reference_marker(citing_src).strip()
            cited_src = process_reference_marker(cited_src).strip()
            tgt = process_reference_marker(src_tgt_pair["citation_text"]).replace('\n', ' ').strip()
            
            src = citing_src + " <DOC_SEP> " + cited_src

            control_codes = ""

            # prepend citation intent token
            if prepend_token:
                control_codes += f'@{src_tgt_pair["intent"]} '
                src = f"{control_codes} {src}"


            if src != "" and tgt != "":
                preprocessd_src_tgt_pairs.append({
                    "citingPaperId": src_tgt_pair["citingPaperId"],
                    "citedPaperId": src_tgt_pair["citedPaperId"],
                    "citingTitle": src_tgt_pair["citingTitle"],
                    "citingAbstract": src_tgt_pair["citingAbstract"],
                    "citedTitle": src_tgt_pair["citedTitle"],
                    "citedAbstract": src_tgt_pair["citedAbstract"],
                    "intent": src_tgt_pair["intent"],
                    "model_src": src,
                    "citation_text": tgt
                })

        return preprocessd_src_tgt_pairs

    def get_preprocessed_src_tgt_pairs(self, intent=None, prepend_token=False,
                                       context_input_mode='cond_sum', context_n_sentences = 2, context_n_matches = 2, title = False, cs_model = 'SciBERT'):
        return self._preprocess(self.get_src_tgt_pairs(), intent=intent, prepend_token=prepend_token,
                                context_input_mode=context_input_mode, context_n_sentences = context_n_sentences, context_n_matches = context_n_matches, title = title, cs_model = cs_model)

########### EXPORT

    def _export_to_transformers_input_file(self, dataset, dataset_type,
                                           out_dir='data/preprocessed'):
        data_path = os.path.join(out_dir, dataset_type)

        with open(f"{data_path}.source", "w") as f_src, open(f"{data_path}.target", "w") as f_tgt:
            for data in dataset:
                f_src.write(data["model_src"])
                f_src.write('\n')

                f_tgt.write(data["citation_text"])
                f_tgt.write('\n')

    def _export_to_jsonl_file(self, dataset, dataset_type,
                              out_dir='data/preprocessed'):
        data_path = os.path.join(out_dir, dataset_type)

        with open(f"{data_path}.jsonl", "w") as f_src, open(f"{data_path}.jsonl", "w") as f_tgt:
            for data in dataset:
                f_src.write(json.dumps(data))
                f_src.write('\n')

                f_tgt.write(json.dumps(data))
                f_tgt.write('\n')


    def export_to_transformers_input_file(self, dataset_type, out_dir='data/preprocessed',
                                          intent=None, prepend_token=False, context_input_mode='cond_sum', context_n_sentences = 2, context_n_matches = 2,  title = None, cs_model = 'SciBERT'):
        dataset = self.get_preprocessed_src_tgt_pairs( 
            intent=intent, prepend_token=prepend_token, context_input_mode=context_input_mode, context_n_sentences = context_n_sentences, context_n_matches = context_n_matches,  title = title, cs_model = cs_model)


        if (not os.path.isdir(out_dir)) and (out_dir != ''):
            os.mkdir(out_dir)

        self._export_to_transformers_input_file(
            dataset, dataset_type, out_dir=out_dir)

    def export_to_jsonl_file(self, dataset_type, out_dir='data/preprocessed',
                             intent=None, prepend_token=False, context_input_mode='cond_sum', context_n_sentences = 2, context_n_matches = 2, title = None, cs_model = 'SciBERT'):
        dataset = self.get_preprocessed_src_tgt_pairs( 
            intent=intent, prepend_token=prepend_token, context_input_mode=context_input_mode, context_n_sentences = context_n_sentences, context_n_matches = context_n_matches, title = title, cs_model = cs_model)

        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)

        self._export_to_jsonl_file(
            dataset, dataset_type, out_dir=out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str,
                        help="Path of parsed dataset to be exported to transformers input format")
    parser.add_argument("--out_dir", type=str, default='',
                        help="Path of output diretory")
    parser.add_argument("--dataset_type", type=str, required=False,
                        help="Dataset type (train/val/test)")
    parser.add_argument("--intent", type=str, required=False,
                        help="Citation dataset with <intent>")
    parser.add_argument("--context_input_mode", type=str, default="cond_sum",
                        help="The context input mode for principal and cited paper")
    parser.add_argument("--context_n_sentences", type=int, default=2,
                        help="Level of aggregation in cond_sum method (if selected)")
    parser.add_argument("--context_n_matches", type=int, default=2,
                        help="Number of matches in cond_sum method (if selected)")
    parser.add_argument("--title", action='store_true')
    parser.add_argument("--prepend_token", action="store_true",
                        help="Prepend intent token")
    parser.add_argument("--n_start", type=int, required=False, default = 0,
                        help="starting index of observations to be processed")
    parser.add_argument("--n_end", type=int, required=False, default = 1000000,
                        help="end index of observations to be processed")    
    parser.add_argument("--cs_model", type=str, required=False)
    parser.add_argument("--outfile_type", type=str, required=True,
                        help="export file type (huggingface transformers format (hf) or jsonl)")
    args = parser.parse_args()

    os.makedirs(f'{args.out_dir}', exist_ok=True)


    dataset_processed = DataPreprocessor(args.input_file, args.n_start, args.n_end)

    if args.dataset_type is None:
        raise ValueError("Please input dataset type!")

    if args.outfile_type == "jsonl":
        dataset_processed.export_to_jsonl_file(
            args.dataset_type, out_dir=args.out_dir, intent=args.intent,
            context_input_mode=args.context_input_mode,context_n_sentences=args.context_n_sentences, context_n_matches=args.context_n_matches, title=args.title, cs_model=args.cs_model)
    elif args.outfile_type == "hf":
        dataset_processed.export_to_transformers_input_file(
            args.dataset_type, out_dir=args.out_dir, intent=args.intent,
            prepend_token=args.prepend_token, context_input_mode=args.context_input_mode,
            context_n_sentences=args.context_n_sentences, context_n_matches=args.context_n_matches, title = args.title, cs_model=args.cs_model)


'''
conda activate cctgm (local)

python preprocessing/data_preprocess.py \
    --input_file=data/original/train.jsonl \
    --out_dir=data/preprocessed/single_summ \
    --dataset_type=train2 \
    --context_input_mode=cond_sum  \
    --context_n_sentences=2 \
    --context_n_matches=5 \
    --n_start=0 \
    --n_end=5 \
    --outfile_type=hf


# --prepend_token \
# --intent <background/method/result>
'''