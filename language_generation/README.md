# Language Generation

This folder contains the code associated with the language generation task for explaining relationships. It is divided into the following two subfolders.

In `models/`, all training and evaluation scripts of the generation task are provided. In addition, it contains the scripts for evaluating the intent accuracy as well as retrieving random samples to showcase in the thesis. To integrate the pre-trained *SciGen* and *SciGPT2* models from Luu et al. (2020), the **scigen** and **scigpt2** folders from the [Google Drive folder](https://drive.google.com/drive/folders/1uGxfWfnK_PtNfKEfuc2EbCuEQpZpjnQJ?usp=sharing) need to be saved in `models/SciGen`.

In `qualitative study/`, the survey distributed to the jurors to rate the generated explanations on various dimensions is presented. In addition, the scripts for generating this survey as well as evaluating it are provided.

____________________________________

BART/PEGASUS: https://github.com/BradLin0819/Automatic-Citation-Text-Generation-with-Citation-Intent-Control
OPT/SciGen: https://github.com/Kel-Lu/SciGen



## 6) Train models based on different data contexts
- **SciGen** 
    - use `train_evaluate.sh` to fine-tune the pre-trained **SciGPT2** model and subsequently evaluate it on the test set
    - use `evaluate.sh` to only evaluate the pre-trained **SciGen** model
- **BART**: use `train_eval` to fine-tune the base *BART* model and subsequently evaluate it on the test set

## 7) Evaluate model outputs
- see model outputs + contents of papers via `retrieve_outputs.sh`
- evaluate performance by using **CORWA** to re-classify generated sentences (`eval_intent.sh`)
- use `compare_intent_generation.sh` to analyze the output of models trained on different intents for the same input (are reflection models more reflective in nature?)