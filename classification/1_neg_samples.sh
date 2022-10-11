#!/bin/bash

N=$1
SEED=42

sbatch <<EOT
#!/bin/sh

#SBATCH --job-name=neg_${N}.job
#SBATCH --output=out/neg_${N}.out
#SBATCH --export=All

#SBATCH --partition=gpu_8
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

#SBATCH --mem=40gb
#SBATCH --time=3:00:00

#SBATCH --mail-user=tim-moritz.buendert@student.uni-tuebingen.de
#SBATCH --mail-type=BEGIN,END,FAIL

../python_env/bin/python3 1_neg_samples.py \
    --n=${N} \
    --seed=${SEED}

EOT