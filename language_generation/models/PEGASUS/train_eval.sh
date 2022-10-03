#!/bin/bash

BASE_MODEL="google/pegasus-large"
MODEL_NAME="pegasus"
INTENT=$1
DATA_PROCESSING=$2

# Training
BASE_DIR="../../contexts_${INTENT}"
DATA_DIR="${BASE_DIR}/${DATA_PROCESSING}"
OUTPUT_DIR="models/${MODEL_NAME}_${INTENT}_${DATA_PROCESSING}"
TRANSFORMERS_PATH="../BART/transformers_src"

SEED=42
MAX_SOURCE_LENGTH=1024

# Training Arguments
NUM_EPOCHS=5
LEARNING_RATE=1e-5
WARMUP_RATIO=0.1
BATCH_SIZE=4
WEIGHT_DECAY=0.05

# Decoding parameters
MAX_NEW_TOKENS=60
TOP_P=0.9

TYPICAL_P=0.9

NUM_BEAMS=4
NO_REPEAT_NGRAM=3
LENGTH_PENALTY=2


# Evaluation
EXPERIMENT_PATH=experiments/${MODEL_NAME}_${INTENT}_${DATA_PROCESSING}
OUTPUT_PATH=${EXPERIMENT_PATH}/test.output
REFERENCE_PATH=${DATA_DIR}/test.target

if [ ! -d ${EXPERIMENT_PATH} ]; then
    mkdir -p ${EXPERIMENT_PATH}
fi

sbatch <<EOT
#!/bin/sh

#SBATCH --job-name=${MODEL_NAME}_${INTENT}_${DATA_PROCESSING}.job
#SBATCH --output=${MODEL_NAME}_${INTENT}_${DATA_PROCESSING}.out
#SBATCH --export=ALL

#SBATCH --partition=gpu_8
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

#SBATCH --mem=10gb
#SBATCH --time=15:00:00

#SBATCH --mail-user=tim-moritz.buendert@student.uni-tuebingen.de
#SBATCH --mail-type=BEGIN,END,FAIL

../../python_env/bin/python3 ${TRANSFORMERS_PATH}/finetune_trainer.py \
    --model_name_or_path ${BASE_MODEL} \
    --context_style ${DATA_PROCESSING} \
    --do_train \
    --do_eval \
    --task summarization \
    --data_dir ${DATA_DIR} \
    --logging_dir runs/${MODEL_NAME}_${INTENT}_${DATA_PROCESSING} \
    --save_strategy epoch \
    --evaluation_strategy epoch \
    --overwrite_output_dir \
    --predict_with_generate \
    --load_best_model_at_end \
    --metric_for_best_model loss \
    --seed=${SEED} \
    --save_total_limit 2 \
    --max_source_length=${MAX_SOURCE_LENGTH} \
    --val_max_target_length=${MAX_SOURCE_LENGTH} \
    --test_max_target_length=${MAX_SOURCE_LENGTH} \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs=${NUM_EPOCHS} \
    --learning_rate ${LEARNING_RATE} \
    --warmup_ratio=${WARMUP_RATIO} \
    --per_device_train_batch_size=${BATCH_SIZE} \
    --per_device_eval_batch_size=${BATCH_SIZE} \
    --weight_decay=${WEIGHT_DECAY}

#    --eval_beams=${NUM_BEAMS} \
#    --generation_num_beams=${NUM_BEAMS} \


echo "Training completed"


../../python_env/bin/python3 ${TRANSFORMERS_PATH}/run_eval.py ${MODEL_NAME} ${DATA_DIR}/test.source ${OUTPUT_PATH} \
    --pretrained_model_path=${OUTPUT_DIR} \
    --reference_path ${REFERENCE_PATH} \
    --score_path=${EXPERIMENT_PATH}/test_metric.json \
    --seed=${SEED} \
    --bs=${BATCH_SIZE} \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --top_p ${TOP_P} 


#    --length_penalty ${LENGTH_PENALTY} \
#    --no_repeat_ngram_size ${NO_REPEAT_NGRAM} \
#    --num_beams ${NUM_BEAMS}

echo "Testing completed"

EOT

# sh train_eval.sh reflection {cond_sum_1_5|title_abs|intro_entity}
# if no intent specified: is single_summ