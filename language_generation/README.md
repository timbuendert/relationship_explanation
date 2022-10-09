# Language Generation

This folder contains the code associated with the language generation task for explaining relationships. It is divided into the following two subfolders.

In `models/`, all training and evaluation scripts of the generation task are provided. In addition, it contains the scripts for evaluating the intent accuracy as well as retrieving random samples to showcase in the thesis. To integrate the pre-trained *SciGen* and *SciGPT2* models from Luu et al. (2020), the **scigen** and **scigpt2** folders from the [Google Drive folder](https://drive.google.com/drive/folders/1uGxfWfnK_PtNfKEfuc2EbCuEQpZpjnQJ?usp=sharing) need to be saved in `models/SciGen`. Also note that `models/BART/transformers_src` has been adapted from [Jung et al. (2022)](https://github.com/BradLin0819/Automatic-Citation-Text-Generation-with-Citation-Intent-Control). 
In more detail, the subfolders **BART**, **OPT**, **PEGASUS** and **SciGen** contain the corresponding training and evaluation scripts for the specific models. On the other hand, **eval_intents** is used to evaluate the intents of the generated samples on the test set. The subfolder **samples** generally contains the scripts to retrieve the generated samples presented in the thesis.

In `qualitative study/`, the survey distributed to the jurors to rate the generated explanations on various dimensions is presented. In addition, the scripts for generating this survey as well as evaluating it are provided.
