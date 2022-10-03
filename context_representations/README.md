# Context Representations


______

scierc.tar.gz in classification/dygie, context_representations/intro_entity: https://github.com/dwadden/dygiepp#pretrained-models

dygie: https://github.com/dwadden/dygiepp
data_preprocess /utils: https://github.com/BradLin0819/Automatic-Citation-Text-Generation-with-Citation-Intent-Control/tree/main/utils

Find larger data files in [Google Drive Folder](https://drive.google.com/drive/folders/1uGxfWfnK_PtNfKEfuc2EbCuEQpZpjnQJ?usp=sharing)


## 5) Use dataset to derive spearate datsets based on context types
- *cond_sum*: via `all_cond_sum.sh` (calling `preprocess.sh` as often as needed) on Cluster -> use `combine_chunks.py` to merge them one dataset
- *intro_tfidf*: use `train_vect.sh` to generate the trained TF-IDF vectorizer and subsequently `intro_tfidf.sh` to generate the samples -> use `combine_chunks.py` to merge them one dataset
- *intro_entity*: set up the python environment `python_env_dygie` via `setup_dygie.sh` -> `train_vect.sh` to generate the trained TF-IDF vectorizer for entities and unigrams -> and subsequently `intro_entity.sh` to generate the samples (*dygie* folder has to be in two places: *intro_entity* and *contexts_${INTENT}*) -> use `combine_chunks.py` to merge them one dataset
- *title_abs*: via `title_abs.sh`
- *cited_only* from *cond_sum* and *title_abs* via `cited_only.sh`
