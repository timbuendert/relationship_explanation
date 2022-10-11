#!/bin/bash

OUTPUT_DIR=SciBERT-finetuned

sbatch <<EOT
#!/bin/sh

#SBATCH --job-name=finetune.job
#SBATCH --output=finetune.out
#SBATCH --export=All

#SBATCH --partition=gpu_8
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

#SBATCH --mem=20gb
#SBATCH --time=06:00:00

#SBATCH --mail-user=tim-moritz.buendert@student.uni-tuebingen.de
#SBATCH --mail-type=BEGIN,END,FAIL

../python_env/bin/python3 finetune.py \
    --output_dir=${OUTPUT_DIR}

EOT