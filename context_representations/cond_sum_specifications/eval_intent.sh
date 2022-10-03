#!/bin/bash

MODEL=$1

sbatch <<EOT
#!/bin/sh

#SBATCH --job-name=intents_${MODEL}.job
#SBATCH --output=out/intents_${MODEL}.out
#SBATCH --export=ALL

#SBATCH --partition=fat
#SBATCH --nodes=1
#SBATCH --mem=20gb
#SBATCH --time=10:00:00

#SBATCH --mail-user=tim-moritz.buendert@student.uni-tuebingen.de
#SBATCH --mail-type=BEGIN,END,FAIL

../../python_env/bin/python3 ../../corwa/pipeline.py \
    --checkpoint ../../corwa/joint_tagger_train_scibert_final.model \
    --intent=reflection \
    --model=${MODEL} \
    --context=cond_sum \
    --cond_sum_analysis

EOT

# sh eval_intent.sh 2_3_t