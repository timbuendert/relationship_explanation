# Classification

This folder contains the code for framing the relationship explanation as a *classification* task. To this end, positive and negative samples are retrieved and represented by the respective contexts as described in the thesis (**1_**). This is based on a smaller subset which was was labelled using the *true* explanation and the principal title as basis for the decision (**2_**). Finally, these inputs are used to address the classification task.

For the *intro-entity* context representation, the [**dygie**](https://github.com/dwadden/dygiepp) pipeline is used. Therefore, the pre-trained `scierc.tar.gz` needs to be downloaded from the [Google Drive folder](https://drive.google.com/drive/folders/1uGxfWfnK_PtNfKEfuc2EbCuEQpZpjnQJ?usp=sharing) to `dygie/`.
