# Dataset Construction

This folder contains the code associated with the construction of our final data corpus. 


______

joint_tagger_train_scibert_final.model in dataset_construction/2_CORWA: https://github.com/jacklxc/CORWA

CORWA: https://github.com/jacklxc/CORWA
S2ORC: https://github.com/allenai/s2orc


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
