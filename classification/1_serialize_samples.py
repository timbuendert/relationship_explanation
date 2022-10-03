import pandas as pd
import argparse
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("--split", type=str)
args = parser.parse_args()

samples = pd.read_pickle(f"data/{args.split}_samples.pkl")
print(samples.shape)

samples.to_json(f"data/{args.split}_samples_ie.json", orient = 'records')