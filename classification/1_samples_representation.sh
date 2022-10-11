#!/bin/bash

SPLIT=$1
CONTEXT=$2
N_SENTENCES=$3
N_MATCHES=$4
TITLE=$5

CS_MODEL=../cs_BERT/SentenceCSBert/

if [ "$TITLE" = true ] ; then
  NAME=${CONTEXT}_${SPLIT}_${N_SENTENCES}_${N_MATCHES}_t
else
  NAME=${CONTEXT}_${SPLIT}_${N_SENTENCES}_${N_MATCHES}
fi

INTRO_ENTITY=intro_entity

sbatch <<EOT
#!/bin/sh

#SBATCH --job-name=${NAME}.job
#SBATCH --output=out/${NAME}.out
#SBATCH --export=All

#SBATCH --partition=gpu_8
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

#SBATCH --mem=10gb
#SBATCH --time=2:00:00

#SBATCH --mail-user=tim-moritz.buendert@student.uni-tuebingen.de
#SBATCH --mail-type=BEGIN,END,FAIL


if [ "$TITLE" = true ] ; then
  ../python_env/bin/python3 1_samples_representation.py \
      --split=${SPLIT} \
      --context=${CONTEXT} \
      --cs_model=${CS_MODEL} \
      --n_sentences=${N_SENTENCES} \
      --n_matches=${N_MATCHES} \
      --title
elif [ "$CONTEXT" = "$INTRO_ENTITY" ]; then
  ../python_env_dygie/bin/python3 1_samples_representation.py \
      --split=${SPLIT} \
      --context=${CONTEXT} \
      --cs_model=${CS_MODEL} \
      --n_sentences=${N_SENTENCES} \
      --n_matches=${N_MATCHES}
else
  ../python_env/bin/python3 1_samples_representation.py \
      --split=${SPLIT} \
      --context=${CONTEXT} \
      --cs_model=${CS_MODEL} \
      --n_sentences=${N_SENTENCES} \
      --n_matches=${N_MATCHES}
fi

EOT