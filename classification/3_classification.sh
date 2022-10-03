#!/bin/bash

#MODEL_PATH=bert-base-uncased
#MODEL_PATH=allenai/scibert_scivocab_uncased
MODEL_PATH=../cs_BERT/SciBERT-finetuned/

CONTEXT=$1
EPOCHS=15
LEARNING_RATE=1e-6

sbatch <<EOT
#!/bin/sh

#SBATCH --job-name=class_${CONTEXT}.job
#SBATCH --output=out/class_${CONTEXT}.out
#SBATCH --export=All

#SBATCH --partition=gpu_4
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

#SBATCH --mem=15gb
#SBATCH --time=1:00:00

#SBATCH --mail-user=tim-moritz.buendert@student.uni-tuebingen.de
#SBATCH --mail-type=BEGIN,END,FAIL

../python_env/bin/python3 3_classification.py \
    --model_path=${MODEL_PATH} \
    --context=${CONTEXT} \
    --lr=${LEARNING_RATE} \
    --epochs=${EPOCHS} 

EOT

# sh 3_classification.sh {title_abs|intro_entity|cond_sum_2_5_t}
