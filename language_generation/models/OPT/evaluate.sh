#!/bin/bash

INTENT=$1
CONTEXT=$2

BATCH_SIZE=4

# evaluation parameters
NUM_SAMPLES=1
LENGTH=55
TEMPERATURE=1.0
TOP_K=0
TOP_P=0.9

sbatch <<EOT
#!/bin/sh

#SBATCH --job-name=opt_${INTENT}_${CONTEXT}_eval.job
#SBATCH --output=out/opt_${INTENT}_${CONTEXT}_eval.out
#SBATCH --export=All

#SBATCH --partition=gpu_8
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

#SBATCH --mem=40gb
#SBATCH --time=45:00:00

#SBATCH --mail-user=tim-moritz.buendert@student.uni-tuebingen.de
#SBATCH --mail-type=BEGIN,END,FAIL

../../python_env/bin/python3 opt_generate.py \
    --model_type=opt \
    --base_dir=../../contexts_${INTENT} \
    --model_name_or_path=training/${INTENT}_${CONTEXT} \
    --output_file=outputs/eval_${INTENT}_${CONTEXT}_opt \
    --input_type=${CONTEXT} \
    --split=test \
    --num_samples=${NUM_SAMPLES} \
    --length=${LENGTH} \
    --temperature=${TEMPERATURE} \
    --top_k=${TOP_K} \
    --top_p=${TOP_P}

echo "Testing completed"

EOT

# For multiple samples: --n_obs=7, 1 fat node for 4h, 5 samples

# N_OBS=2678
# --n_obs=${N_OBS} \


# sh evaluate.sh reflection cond_sum
# if no intent specified: is single_summ