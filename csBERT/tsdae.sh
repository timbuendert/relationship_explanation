#!/bin/bash

INPUT_DIR=SciBERT-finetuned
OUTPUT_DIR=SentenceCSBert
BATCH_SIZE=2
N=2

sbatch <<EOT
#!/bin/sh

#SBATCH --job-name=tsdae.job
#SBATCH --output=tsdae.out
#SBATCH --export=All

#SBATCH --partition=gpu_8
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

#SBATCH --mem=30gb
#SBATCH --time=04:00:00

#SBATCH --mail-user=tim-moritz.buendert@student.uni-tuebingen.de
#SBATCH --mail-type=BEGIN,END,FAIL

../python_env/bin/python3 tsdae.py \
    --input_dir=${INPUT_DIR} \
    --output_dir=${OUTPUT_DIR} \
    --bs=${BATCH_SIZE} \
    --n=${N}

EOT