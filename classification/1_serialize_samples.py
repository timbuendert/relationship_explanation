import pandas as pd
import argparse

# serialize samples for the intro-entity context representation because it requires a different Python version which in incompatible with the the saved pickle files

parser = argparse.ArgumentParser()
parser.add_argument("--split", type=str)
args = parser.parse_args()

samples = pd.read_pickle(f"data/{args.split}_samples.pkl")
print(samples.shape)

# save json file
samples.to_json(f"data/{args.split}_samples_ie.json", orient = 'records')