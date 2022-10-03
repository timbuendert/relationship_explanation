#!/bin/bash

sbatch <<EOT
#!/bin/sh

#SBATCH --job-name=chunks_"$1".job
#SBATCH --output=chunks_"$1".out
#SBATCH --export=All

#SBATCH --partition=fat
#SBATCH --nodes=1

#SBATCH --mem=5gb
#SBATCH --time=01:00:00

#SBATCH --mail-user=tim-moritz.buendert@student.uni-tuebingen.de
#SBATCH --mail-type=BEGIN,END,FAIL


../python_env/bin/python3 combine_chunks.py --type=$1 --n_train=$2 --n_val=$3 --n_test=$4

# cond_sum: 4 2 2
# intro_tfidf: 10 4 4

EOT
