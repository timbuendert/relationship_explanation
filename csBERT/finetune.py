# adapted from: https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling.ipynb#scrollTo=3R1RA5w5eZ5E

from transformers import AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling, AutoModelForMaskedLM
from datasets import Dataset, DatasetDict
import math
import argparse
import os
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=str)
args = parser.parse_args()

# load data
with open('texts.pkl', 'rb') as f:
    texts = pickle.load(f)    

# create Dataset object with corresponding splits
dataset = Dataset.from_dict({'text': texts})
train_dataset, validation_dataset = dataset.train_test_split(test_size=0.3, seed = 42).values()
dataset = DatasetDict({'train': train_dataset, 'val': validation_dataset})


# Load tokenizer and Model for  Masked-language Modelling
tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased', use_fast=True)
model = AutoModelForMaskedLM.from_pretrained('allenai/scibert_scivocab_uncased')


# tokenize input texts
def tokenize_function(examples):
    return tokenizer(examples["text"])

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])


# create blocks of input tokens
block_size = 128
def group_texts(examples):
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()} # concatenate all texts
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    total_length = (total_length // block_size) * block_size # drop small remainder -> could also be padded
    result = { # Split by chunks of max_len
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    batch_size=1000
)

# setup for training
training_args = TrainingArguments(
    args.output_dir,
    overwrite_output_dir = True,
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    save_steps = 100000
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets["train"],
    eval_dataset=lm_datasets["val"],
    data_collator=data_collator,
)

####### TRAINING
# n = 100 -> 2:30m for 3 epochs

print('TRAINING')
train_result = trainer.train(resume_from_checkpoint = True) # or resume from path / resume_from_checkpoint = True

trainer.save_model(args.output_dir)
trainer.state.save_to_json(os.path.join(args.output_dir, "trainer_state.json"))
tokenizer.save_pretrained(args.output_dir)


####### EVALUATION
print('Evaluating')
eval_results = trainer.evaluate()
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
print(f'Evaluation metrics: {eval_results}')

