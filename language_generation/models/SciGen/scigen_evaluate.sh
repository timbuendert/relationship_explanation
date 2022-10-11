#!/bin/bash

INTENT=$1
CONTEXT=$2

SEED=42

# decoding parameters
NUM_SAMPLES=1
LENGTH=60
TEMPERATURE=1.0
TOP_K=0
TOP_P=0.9

sbatch <<EOT
#!/bin/sh

#SBATCH --job-name=scigen_${INTENT}_${CONTEXT}.job
#SBATCH --output=out/scigen_${INTENT}_${CONTEXT}.out
#SBATCH --export=ALL

#SBATCH --partition=gpu_8
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

#SBATCH --mem=10gb
#SBATCH --time=4:00:00

#SBATCH --mail-user=tim-moritz.buendert@student.uni-tuebingen.de
#SBATCH --mail-type=BEGIN,END,FAIL

../../python_env/bin/python3 scigpt2_generate.py \
    --model_type=gpt2 \
    --base_dir=../../contexts_${INTENT} \
    --model_name_or_path=scigen \
    --output_file=outputs/scigen_${INTENT}_${CONTEXT} \
    --input_type=${CONTEXT} \
    --seed=${SEED} \
    --split=test \
    --num_samples=${NUM_SAMPLES} \
    --length=${LENGTH} \
    --temperature=${TEMPERATURE} \
    --top_k=${TOP_K} \
    --top_p=${TOP_P}


EOT