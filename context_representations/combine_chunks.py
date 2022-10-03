from tqdm import tqdm
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument('--type', type=str)
argparser.add_argument('--n_train', type=int)
argparser.add_argument('--n_val', type=int)
argparser.add_argument('--n_test', type=int)
args = argparser.parse_args()

dataset_types = {'train': args.n_train, 'val': args.n_val, 'test': args.n_test}

src_path = f'{args.type}/chunks'
file_types = ['source', 'target']

for dataset_type in tqdm(dataset_types.keys()):
    for file_type in file_types:
        with open(f"{args.type}/{dataset_type}.{file_type}", "w") as f_all:
            for i in range(dataset_types[dataset_type]):
                with open(f"{src_path}/{dataset_type}_{i}.{file_type}", "r") as dataset:
                    for line in dataset:
                        f_all.write(line)
