# csBERT

This folder contains the code for generating the *BERT*-based **csBERT** and the *SentenceBERT*-based **csSBERT** language models. 

While **csBERT** is constructed by fine-tuning [`SciBERT`](https://github.com/allenai/scibert) to the computer science domain (using `finetune.py`), **csBERT** is fine-tuned to a sentence Transformer model using `tsdae.py` to constitute **csSBERT**.
