# Language Generation

This folder contains the code associated with the language generation task for explaining relationships. It is divided into the following two subfolders.

In `models/`, the code for training and evaluating the models is present. In more detail, the subfolders `models/BART`, `models/OPT`, `models/PEGASUS` and `models/SciGen` contain the corresponding training and evaluation scripts for the specific models. In addition, it contains the scripts for evaluating the intent accuracy (`models/eval_intents`) as well as retrieving random samples to showcase in the thesis (`models/samples`). To integrate the pre-trained *SciGen* and *SciGPT2* models from Luu et al. (2020), the **scigen** and **scigpt2** folders from the [Google Drive folder](https://drive.google.com/drive/folders/1uGxfWfnK_PtNfKEfuc2EbCuEQpZpjnQJ?usp=sharing) need to be saved in `models/SciGen`. Also note that `models/BART/transformers_src` has been adapted from [Jung et al. (2022)](https://github.com/BradLin0819/Automatic-Citation-Text-Generation-with-Citation-Intent-Control). 

In `qualitative study/`, the survey distributed to the jurors to rate the generated explanations on various dimensions is presented. In addition, the scripts for generating this survey as well as evaluating it are provided.
