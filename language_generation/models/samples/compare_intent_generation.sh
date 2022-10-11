#!/bin/bash

MODEL=$1
CONTEXT=$2
N=$3
SEED=42

sbatch <<EOT
#!/bin/sh

#SBATCH --job-name=intents_${MODEL}_${CONTEXT}.job
#SBATCH --output=intents_${MODEL}_${CONTEXT}.out
#SBATCH --export=All

#SBATCH --partition=gpu_8
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

#SBATCH --mem=15gb
#SBATCH --time=02:00:00

#SBATCH --mail-user=tim-moritz.buendert@student.uni-tuebingen.de
#SBATCH --mail-type=BEGIN,END,FAIL

../python_env/bin/python3 compare_intent_generation.py --model=${MODEL} --context=${CONTEXT} --n_samples=${N} --seed=${SEED}

EOT