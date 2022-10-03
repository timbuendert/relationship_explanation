# Explaining Relationships Between Academic Documents

This repository contains the practical implementations of the graduate thesis **"Explaining Relationships Between Academic Documents Using Generative Transformer Models"** as part of the *MSc. Data Science in Business and Economics* at *University of Tübingen*. In this work, a novel conditional context representation for the two papers is proposed which is evaluated across several state-of-the-art Transformer language models. For more details, please [contact me](mailto:tim.moritz.buendert@googlemail.com) to receive the entire thesis.

To use the provided scripts, please create a new `Python 3` environment and execute the following command to to install all packages and dependencies.

```pip install -r requirements.txt```

Apart from the provided scripts, larger data files are made available in a corresponding [Google Drive Folder](https://drive.google.com/drive/folders/1uGxfWfnK_PtNfKEfuc2EbCuEQpZpjnQJ?usp=sharing). This includes the pre-trained **dygie** model `scierc.tar.gz`, the pre-trained joint tagger from the *CORWA* pipeline as well as the **scigen** and **scigpt2** folders

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
