#!/bin/bash

SPLIT=10
START=$(($1 * ${SPLIT}))
END=$((($1 * ${SPLIT}) + ${SPLIT}))

sbatch <<EOT
#!/bin/sh

#SBATCH --job-name=filter_"$1".job
#SBATCH --output=filter_"$1".out

#SBATCH --export=ALL

#SBATCH --partition=fat
#SBATCH --nodes=1
#SBATCH --mem=3gb
#SBATCH --time=08:00:00

#SBATCH --mail-user=tim-moritz.buendert@student.uni-tuebingen.de
#SBATCH --mail-type=BEGIN,END,FAIL

../python_env/bin/python3 0_filter_papers.py --batches-start=${START} --batches-end=${END} --n=$1

# scripts from 0-9 (non-cited)
EOT