# Explaining Relationships Between Academic Documents

This repository contains the practical implementations corresponding to the graduate thesis **"Explaining Relationships Between Academic Documents Using Generative Transformer Models"** as part of the *MSc. Data Science in Business and Economics* at the *University of Tübingen*. In this work, amongst others, a novel conditional context representation for the two papers is proposed and evaluated across several state-of-the-art Transformer language models. A schematic overview of this new method is shown below.

![Schematic Overview of Conditional Summary Context Construction](https://github.com/timbuendert/relationship_explanation/blob/main/context_representations/cond_sum/cond_sum_schema.png)

For more details, please [contact me](mailto:tim.moritz.buendert@googlemail.com) to receive the entire thesis.
 

The demo notebook [**demo_notebook.ipynb**](https://github.com/timbuendert/relationship_explanation/blob/main/demo_notebook.ipynb) contains an exemplary application of the methods discussed and developed in the corresponding paper and invites for further experiments. To use it and experiment with the fine-tuned models, please download the corresponding Python module <b>relationship_explanation</b> from [Google Drive](https://drive.google.com/drive/folders/1uGxfWfnK_PtNfKEfuc2EbCuEQpZpjnQJ?usp=sharing).


Generally, to use the provided scripts, please create a new `Python 3` environment and execute the following command to to install all packages and dependencies.

```pip install -r requirements.txt```

Apart from the provided scripts, larger data files are made available in a corresponding [Google Drive folder](https://drive.google.com/drive/folders/1uGxfWfnK_PtNfKEfuc2EbCuEQpZpjnQJ?usp=sharing). This includes the pre-trained **dygie** model `scierc.tar.gz` ([source](https://github.com/dwadden/dygiepp#pretrained-models)), the pre-trained joint tagger `joint_tagger_train_scibert_final.model` ([source](https://github.com/jacklxc/CORWA)) from the *CORWA* pipeline as well as the `scigen` and `scigpt2` folders ([source](https://github.com/Kel-Lu/SciGen)). For more details, please refer to the **README** files in the separate folders.