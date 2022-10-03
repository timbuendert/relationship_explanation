#!/bin/bash

INTENT=$1
CONTEXT=$2
MODEL=facebook/opt-350m

SEED=42

# training parameters
NUM_EPOCHS=5
LEARNING_RATE=1e-5
WEIGHT_DECAY=0.05
WARMUP_PROPORTION=0.1
BATCH_SIZE=4

# decoding parameters
NUM_SAMPLES=1
LENGTH=60
TEMPERATURE=1.0
TOP_K=0
TOP_P=0.9

sbatch <<EOT
#!/bin/sh

#SBATCH --job-name=opt_${INTENT}_${CONTEXT}.job
#SBATCH --output=out/opt_${INTENT}_${CONTEXT}.out
#SBATCH --export=All

#SBATCH --partition=gpu_8
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

#SBATCH --mem=10gb
#SBATCH --time=10:00:00

#SBATCH --mail-user=tim-moritz.buendert@student.uni-tuebingen.de
#SBATCH --mail-type=BEGIN,END,FAIL


../../python_env/bin/python3 opt_train.py \
    --model_type=opt \
    --model_name_or_path=${MODEL} \
    --base_dir=../../contexts_${INTENT} \
    --output_dir=training/${INTENT}_${CONTEXT} \
    --context_style=${CONTEXT} \
    --do_train \
    --do_eval \
    --per_gpu_train_batch_size=${BATCH_SIZE} \
    --per_gpu_eval_batch_size=${BATCH_SIZE} \
    --num_train_epochs=${NUM_EPOCHS} \
    --learning_rate=${LEARNING_RATE} \
    --overwrite_output_dir \
    --save_total_limit=5 \
    --seed=${SEED} \
    --weight_decay=${WEIGHT_DECAY} \
    --warmup_proportion=${WARMUP_PROPORTION}

echo "Training completed"

../../python_env/bin/python3 opt_generate.py \
    --model_type=opt \
    --base_dir=../../contexts_${INTENT} \
    --model_name_or_path=training/${INTENT}_${CONTEXT} \
    --output_file=outputs/eval_${INTENT}_${CONTEXT} \
    --input_type=${CONTEXT} \
    --seed=${SEED} \
    --split=test \
    --num_samples=${NUM_SAMPLES} \
    --length=${LENGTH} \
    --temperature=${TEMPERATURE} \
    --top_k=${TOP_K} \
    --top_p=${TOP_P}

echo "Testing completed"

EOT


# N_OBS=2678
# --n_obs=${N_OBS} \


# sh train_evaluate.sh reflection {cond_sum_1_5|title_abs|intro_entity}
# if no intent specified: is single_summ