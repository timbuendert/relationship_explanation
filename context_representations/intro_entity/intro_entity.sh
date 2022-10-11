#!/bin/bash

INTENT=reflection
SPLIT=5000 

N_START=$(($2 * ${SPLIT}))
N_END=$((($2 * ${SPLIT}) + ${SPLIT}))

sbatch <<EOT
#!/bin/sh

#SBATCH --job-name=ie_"$1"_"$2".job
#SBATCH --output=out/ie_"$1"_"$2".out
#SBATCH --export=All

#SBATCH --partition=fat
#SBATCH --nodes=1

#SBATCH --mem=800gb
#SBATCH --time=40:00:00

#SBATCH --mail-user=tim-moritz.buendert@student.uni-tuebingen.de
#SBATCH --mail-type=BEGIN,END,FAIL

echo ${N_START}
echo ${N_END}

../../python_env_dygie/bin/python3 ../data_preprocess.py \
    --input_file=../../final_dataset/data/${INTENT}_"$1".jsonl \
    --out_dir=chunks \
    --dataset_type="$1"_"$2" \
    --context_input_mode=intro_entity \
    --n_start=${N_START} \
    --n_end=${N_END} \
    --outfile_type=hf

EOT