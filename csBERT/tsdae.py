# adapted from https://www.sbert.net/examples/unsupervised_learning/TSDAE/README.html

import argparse
import os
from sentence_transformers import SentenceTransformer, models, datasets, losses
from torch.utils.data import DataLoader
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", type=str)
parser.add_argument("--output_dir", type=str)
parser.add_argument("--bs", type=int)
parser.add_argument("--n", type=int)
args = parser.parse_args()

os.makedirs(f'{args.output_dir}', exist_ok=True)

# load text corpus
with open('texts_1.pkl', 'rb') as f:
    train_sentences = pickle.load(f)    

train_sentences = train_sentences[:args.n]

# Define sentence transformer model using average pooling
word_embedding_model = models.Transformer(args.input_dir, max_seq_length=512)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# Create special denoising dataset that adds noise on-the-fly
train_dataset = datasets.DenoisingAutoEncoderDataset(train_sentences)

# DataLoader to batch data
train_dataloader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, drop_last=True)

# Use the denoising auto-encoder loss
train_loss = losses.DenoisingAutoEncoderLoss(model, decoder_name_or_path=args.input_dir, tie_encoder_decoder=True)

# Train model
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=1,
    weight_decay=0,
    scheduler='constantlr',
    optimizer_params={'lr': 3e-5},
    show_progress_bar=True,
    checkpoint_path=args.output_dir
)

model.save(f'{args.output_dir}/tsdae-model')