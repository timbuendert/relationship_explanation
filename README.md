# Explaining Relationships Between Academic Documents

This repository contains the practical implementations of the graduate thesis **"Explaining Relationships Between Academic Documents Using Generative Transformer Models"** as part of the *MSc. Data Science in Business and Economics* at *University of Tübingen*. In this work, a novel conditional context representation for the two papers is proposed which is evaluated across several state-of-the-art Transformer language models. For more details, please [contact me](mailto:tim.moritz.buendert@googlemail.com) to receive the entire thesis.

To use the provided scripts, please create a new `Python 3` environment and execute ```pip install -r requirements.txt``` to install all packages and dependencies.

______

scierc.tar.gz in classification/dygie, context_representations/intro_entity: https://github.com/dwadden/dygiepp#pretrained-models
joint_tagger_train_scibert_final.model in dataset_construction/2_CORWA: https://github.com/jacklxc/CORWA
scigen & scigpt folders in language_generation/SciGen: https://github.com/Kel-Lu/SciGen

dygie:https://github.com/dwadden/dygiepp
BART/PEGASUS: https://github.com/BradLin0819/Automatic-Citation-Text-Generation-with-Citation-Intent-Control
OPT/SciGen: https://github.com/Kel-Lu/SciGen
CORWA: https://github.com/jacklxc/CORWA
S2ORC: https://github.com/allenai/s2orc
Finetune CSBERT: https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling.ipynb#scrollTo=3R1RA5w5eZ5E
TSDAE: https://www.sbert.net/examples/unsupervised_learning/TSDAE/README.html
data_preprocess /utils: https://github.com/BradLin0819/Automatic-Citation-Text-Generation-with-Citation-Intent-Control/tree/main/utils
classification.py: https://towardsdatascience.com/text-classification-with-bert-in-pytorch-887965e5820f

TODO: download data from cluster!

Find larger data files in [Google Drive Folder](https://drive.google.com/drive/folders/1uGxfWfnK_PtNfKEfuc2EbCuEQpZpjnQJ?usp=sharing)

# Documentation for dataset construction pipeline (process)

## 1) Retrieve relevant principal documents (Cluster): 0_filter_principal
Via `0_sh_script.sh`: use `0_filter_papers.py` to generate *pdf_parses_full{0-9}.jsonl* (and same for metadata) to filter for only CS (& NLP papers)
[Optional: paste received S2ORC credentials in *S2ORC/current_credentials.txt* and process them using `S20RC/process_credentials.py`]

## 2) Extract relevant information (e.g., cited paper IDs) and related work sections from principal documents (Cluster): 1_setup
Via `1_setup.sh`: use these new jsonl files with `1_setup_relWork.py` to create the related work jsonl documents for subsequent CORWA tagging -> results in *related_work{0-9}.jsonl* and *all_ref_paper_ids{0-9}.pkl*


## 3.1) Retrieve cited papers from S2ORC (Cluster): 0_filter_cited
Use all *all_ref_paper_ids{0-9}.pkl* to get cited papers via `0_filter_papers_cited.py` to obtain *pdf_parses_full_cited{0-24}.jsonl* (same for metadata)

## 3.2) Tag extracted related work sections using CORWA (Cluster)
- **2_CORWA**: Tag *related_work{0-9}.jsonl* to obtain *tagged_related_works{0-9}.jsonl*
- **3_process_CORWA** (local): Process *tagged_related_works{0-9}.jsonl* via `3_process_CORWA.py` (*as in **5)**  of `dataset.ipynb` (code/dataset_construction/dataset_construction.ipynb?)*)
    - (*`3_principal_titles.py` to save titles of principal papers ?*)


## 4) Combine (principal and cited) papers and tagged related work sections to construct final dataset: 4_combine
- Via `4_add.sh`: use `4_add_data.py` to filter the correct titles, abstracts and texts for each principal and cited paper ID
- Via `4_combine.sh`: use `4_combine_final.py` to chunckwise-add the paired data and filter such thath no empty abstracts, body texts, etc. 
- Via `4_merge.sh`: use `4_merge_parts.py` to merge all filtered data chunks
    - Determine which intent to filter for!


## 5) Use dataset to derive spearate datsets based on context types
- *cond_sum*: via `all_cond_sum.sh` (calling `preprocess.sh` as often as needed) on Cluster -> use `combine_chunks.py` to merge them one dataset
- *intro_tfidf*: use `train_vect.sh` to generate the trained TF-IDF vectorizer and subsequently `intro_tfidf.sh` to generate the samples -> use `combine_chunks.py` to merge them one dataset
- *intro_entity*: set up the python environment `python_env_dygie` via `setup_dygie.sh` -> `train_vect.sh` to generate the trained TF-IDF vectorizer for entities and unigrams -> and subsequently `intro_entity.sh` to generate the samples (*dygie* folder has to be in two places: *intro_entity* and *contexts_${INTENT}*) -> use `combine_chunks.py` to merge them one dataset
- *title_abs*: via `title_abs.sh`
- *cited_only* from *cond_sum* and *title_abs* via `cited_only.sh`


## 6) Train models based on different data contexts
- **SciGen** 
    - use `train_evaluate.sh` to fine-tune the pre-trained **SciGPT2** model and subsequently evaluate it on the test set
    - use `evaluate.sh` to only evaluate the pre-trained **SciGen** model
- **BART**: use `train_eval` to fine-tune the base *BART* model and subsequently evaluate it on the test set
- **OPT**: see **SciGen**
- **GPT3**: use `https://elicit.org/tasks`

## 7) Evaluate model outputs
- see model outputs + contents of papers via `retrieve_outputs.sh`
- evaluate performance by using **CORWA** to re-classify generated sentences (`eval_intent.sh`)
- use `compare_intent_generation.sh` to analyze the output of models trained on different intents for the same input (are reflection models more reflective in nature?)