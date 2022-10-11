#!/bin/bash

INTENT = reflection

sbatch <<EOT
#!/bin/sh

#SBATCH --job-name=t_a_"$1".job
#SBATCH --output=t_a_"$1".out
#SBATCH --export=ALL

#SBATCH --partition=fat
#SBATCH --nodes=1
#SBATCH --mem=130gb
#SBATCH --time=24:00:00

#SBATCH --mail-user=tim-moritz.buendert@student.uni-tuebingen.de
#SBATCH --mail-type=BEGIN,END,FAIL

../../python_env/bin/python3 ../data_preprocess.py \
    --input_file=../../final_dataset/data/${INTENT}_"$1".jsonl \
    --dataset_type=$1 \
    --context_input_mode=title_abs  \
    --outfile_type=hf

EOT