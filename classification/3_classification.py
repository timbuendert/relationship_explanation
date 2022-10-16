# Adapted from https://towardsdatascience.com/text-classification-with-bert-in-pytorch-887965e5820f (https://github.com/marcellusruben/medium-resources/blob/main/Text_Classification_BERT/bert_medium.ipynb)

from collections import Counter
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
import argparse
import os
import pandas as pd
import torch.nn.functional as F
from sklearn.metrics import f1_score
import pickle
import copy

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str)
parser.add_argument("--context", type=str)
parser.add_argument("--lr", type=float)
parser.add_argument("--epochs", type=int)
args = parser.parse_args()

tgt_path = f'classifications/{args.context}'
os.makedirs(tgt_path, exist_ok=True)


# load data

with open(f'data/labels.pkl', 'rb') as f:
    labels = pickle.load(f)  

labels = [i for i in labels if i > -1]
print(f'Positive labels of dataset: {len(labels)}')

pos_examples = list(open(f'data/{args.context}/pos_samples.source').readlines())
print(f'Positive number of samples: {len(pos_examples)}')

classes = {'background':0,
           'comparison':1,
           'negative':2
          }

df = pd.DataFrame({'text': pos_examples, 'label': labels})

# add negative samples
neg_examples = list(open(f'data/{args.context}/neg_samples.source').readlines())
neg_labels = [2]*len(neg_examples)

df_labels = df.append(pd.DataFrame({'text': neg_examples, 'label': neg_labels}), ignore_index=True)
print(f'Number of all samples: {df_labels.shape[0]}')


# remove duplicate samples
df_labels = df_labels.drop_duplicates()
print(f'Number of non-duplicate samples: {df_labels.shape[0]}')

counter = Counter(df_labels['label'])
print(f'Class distribution: {counter}')

# create dataset partitions
np.random.seed(112)
df_train, df_val, df_test = np.split(df_labels.sample(frac=1, random_state=42), 
                                     [int(.6*len(df_labels)), int(.8*len(df_labels))])
print(f'Data split: Train = {len(df_train)}, Val = {len(df_val)}, Test = {len(df_test)}')


tokenizer = AutoTokenizer.from_pretrained(args.model_path)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, df):

        self.labels = df['label']
        self.texts = [tokenizer(text, 
                                padding='max_length', max_length = 512, truncation=True,
                                return_tensors="pt") for text in df['text']]

    def classes(self):
        return list(set(self.labels))

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y

# create classifier
class CS_Classifier(nn.Module):
    def __init__(self, dropout=0.5):
        super(CS_Classifier, self).__init__()
        self.lm = AutoModel.from_pretrained(args.model_path)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, len(classes))

    def forward(self, input_id, mask):
        _, pooled_output = self.lm(input_ids=input_id, attention_mask=mask, return_dict=False) # pooler output
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = F.softmax(linear_output, dim=-1)
        return final_layer


def train(model, train_data, val_data, learning_rate, epochs):
    train, val = Dataset(train_data), Dataset(val_data)

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=4, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=4)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    val_losses = []
    best_val_loss = 100

    # training
    for epoch_num in range(epochs):
        total_acc_train = 0
        total_loss_train = 0
        total_train_f1 = 0

        for train_input, train_label in tqdm(train_dataloader):
            model.zero_grad()

            train_label = train_label.to(device)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)
            
            batch_loss = criterion(output, train_label.long())
            total_loss_train += batch_loss.item()
            
            y_pred = output.argmax(dim = 1)

            acc = (y_pred == train_label).sum().item()
            total_acc_train += acc

            total_train_f1 += f1_score(train_label.cpu(), y_pred.cpu(), average = 'weighted')

            batch_loss.backward()
            optimizer.step()
        
        total_acc_val = 0
        total_loss_val = 0
        total_test_f1 = 0

        # validation
        with torch.no_grad():
            for val_input, val_label in val_dataloader:
                val_label = val_label.to(device)
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)

                batch_loss = criterion(output, val_label.long())
                total_loss_val += batch_loss.item()

                y_pred = output.argmax(dim = 1)

                acc = (y_pred == val_label).sum().item()
                total_acc_val += acc
        
                total_test_f1 += f1_score(val_label.cpu(), y_pred.cpu(), average = 'weighted')

        if total_loss_val / len(val_data) < best_val_loss:
            best_val_loss = total_loss_val / len(val_data)
            best_model = copy.deepcopy(model)

        print(
            f'Epochs: {epoch_num + 1}\nTrain Loss: {total_loss_train / len(train_data): .3f} | Train Accuracy: {total_acc_train / len(train_data): .3f} | Train F1 Score: {total_train_f1 / len(train_dataloader): .3f}\n\
                Val Loss: {total_loss_val / len(val_data): .3f} | Val Accuracy: {total_acc_val / len(val_data): .3f} | Val F1 Score: {total_test_f1 / len(val_dataloader): .3f}')

        # early stopping
        if epoch_num > 2:
            if total_loss_val / len(val_data) > val_losses[-1] and total_loss_val / len(val_data) > val_losses[-2]:
                break

        val_losses.append(total_loss_val / len(val_data))

    return best_model
                  
def evaluate(model, test_data):
    test = Dataset(test_data)
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=2)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.cuda()

    total_acc_test = 0
    total_f1_test = 0

    # evaluation
    with torch.no_grad():
        for test_input, test_label in test_dataloader:
            test_label = test_label.to(device)
            mask = test_input['attention_mask'].to(device)
            input_id = test_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)

            y_pred = output.argmax(dim = 1)

            acc = (y_pred == test_label).sum().item()
            total_acc_test += acc
            
            total_f1_test += f1_score(test_label.cpu(), y_pred.cpu(), average = 'weighted')
    
    print(f'\nTest Accuracy: {total_acc_test / len(test_data): .3f} | Test F1 Score: {total_f1_test / len(test_dataloader): .3f}')


model = CS_Classifier()
              
tuned_model = train(model, df_train.reset_index(drop=True), df_val.reset_index(drop=True), args.lr, args.epochs)
torch.save(tuned_model.state_dict(), f'{tgt_path}/model.pth')
tokenizer.save_pretrained(f'{tgt_path}/')
evaluate(tuned_model, df_test.reset_index(drop=True))