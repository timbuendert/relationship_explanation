# Context Representations

This folder contains the code for constructing the different context representations which are used as input to the language models. While previous approaches used fixed portions of the principal and cited papers, this work argues that the conditional nature of the explanation task requires a more flexible method to represent both papers dependent upon each other. Therefore, we propose *conditional summaries* as a way to identify the most relevant sentences from both papers.

For comparison, the previously proposed **Title-Abstract** and **Intro-Entity** approaches are also implemented. For the latter, the [**dygie**](https://github.com/dwadden/dygiepp) pipeline is used. Therefore, the pre-trained `scierc.tar.gz` needs to be downloaded from the [Google Drive folder](https://drive.google.com/drive/folders/1uGxfWfnK_PtNfKEfuc2EbCuEQpZpjnQJ?usp=sharing) to `intro_entity/dygie/`. Please note, that using **dygie** also requires a different Python environment as it is built on a different Python version.

In turn, the folder `cond_sum_specifications/` contains the code associated with the construction of all the different **Cond-Sum** specifications used in the thesis along with their descriptives. The final configuration which turned out to performs best in indicated in `cond_sum/`.

Due to the large dataset sizes, the context representations might be constructed in chunks for optimal efficiency. Based on the individual chunks, `combine_chunks.py` is used to merge them into one dataset.