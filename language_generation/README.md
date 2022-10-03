# Language Generation



______

scigen & scigpt folders in language_generation/SciGen: https://github.com/Kel-Lu/SciGen

BART/PEGASUS: https://github.com/BradLin0819/Automatic-Citation-Text-Generation-with-Citation-Intent-Control
OPT/SciGen: https://github.com/Kel-Lu/SciGen



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