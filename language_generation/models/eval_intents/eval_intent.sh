#!/bin/bash

INTENT=$1
MODEL=$2
CONTEXT=$3

sbatch <<EOT
#!/bin/sh

#SBATCH --job-name=intents_${INTENT}_${MODEL}_${CONTEXT}.job
#SBATCH --output=intents_${INTENT}_${MODEL}_${CONTEXT}.out
#SBATCH --export=ALL

#SBATCH --partition=fat
#SBATCH --nodes=1
#SBATCH --mem=20gb
#SBATCH --time=10:00:00

#SBATCH --mail-user=tim-moritz.buendert@student.uni-tuebingen.de
#SBATCH --mail-type=BEGIN,END,FAIL

../../python_env/bin/python3 ../../corwa/pipeline.py \
    --checkpoint ../../corwa/joint_tagger_train_scibert_final.model \
    --intent=${INTENT} \
    --model=${MODEL} \
    --context=${CONTEXT}
EOT

# sh eval_intent.sh reflection BART cond_sum