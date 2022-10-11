#!/bin/bash

INTENT=reflection

N_SENTENCES=1
N_MATCHES=5
SPLIT=$1
MODEL=../cs_BERT/SentenceCSBert/
NAME=cond_sum_${N_SENTENCES}_${N_MATCHES}_${SPLIT}

sbatch <<EOT
#!/bin/sh

#SBATCH --job-name=${NAME}.job
#SBATCH --output=cond_sum_${N_SENTENCES}_${N_MATCHES}/${NAME}.out
#SBATCH --export=All

#SBATCH --partition=gpu_8
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

#SBATCH --mem=15gb
#SBATCH --time=10:00:00

#SBATCH --mail-user=tim-moritz.buendert@student.uni-tuebingen.de
#SBATCH --mail-type=BEGIN,END,FAIL

../python_env/bin/python3 data_preprocess.py \
    --input_file=../final_dataset/data/${INTENT}_${SPLIT}.jsonl \
    --out_dir=cond_sum_${N_SENTENCES}_${N_MATCHES} \
    --dataset_type=${SPLIT} \
    --context_input_mode=cond_sum \
    --context_n_sentences=${N_SENTENCES} \
    --context_n_matches=${N_MATCHES} \
    --cs_model=${MODEL} \
    --outfile_type=hf

EOT