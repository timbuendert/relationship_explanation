#!/bin/bash

INTENT=reflection
SPLIT=5000 
#50000

N_START=$(($2 * ${SPLIT}))
N_END=$((($2 * ${SPLIT}) + ${SPLIT}))

sbatch <<EOT
#!/bin/sh

#SBATCH --job-name=cs_"$1"_"$2".job
#SBATCH --output=cs_"$1"_"$2".out
#SBATCH --export=All

#SBATCH --partition=gpu_8
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

#SBATCH --mem=20gb
#SBATCH --time=14:00:00
# 60gb

#SBATCH --mail-user=tim-moritz.buendert@student.uni-tuebingen.de
#SBATCH --mail-type=BEGIN,END,FAIL

echo ${N_START}
echo ${N_END}

../../python_env/bin/python3 ../data_preprocess.py \
    --input_file=../../final_dataset/data/${INTENT}_"$1".jsonl \
    --out_dir=chunks \
    --dataset_type="$1"_"$2" \
    --context_input_mode=cond_sum \
    --context_n_sentences=2 \
    --context_n_matches=5 \
    --n_start=${N_START} \
    --n_end=${N_END} \
    --outfile_type=hf

EOT

# --prepend_token \

#sh cond_sum.sh train 1

#Train: (192006, 11)
#Val: (64002, 11)
#Test: (64003, 11)

#Full shape: (12375, 11)
#Train: (7425, 11)
#Val: (2475, 11)
#Test: (2475, 11)