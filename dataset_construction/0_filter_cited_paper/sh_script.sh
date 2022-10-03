#!/bin/bash

SPLIT=4
START=$(($1 * ${SPLIT}))
END=$((($1 * ${SPLIT}) + ${SPLIT}))

sbatch <<EOT
#!/bin/sh

#SBATCH --job-name=filter_cited_"$1".job
#SBATCH --output=filter_cited_"$1".out

#SBATCH --export=ALL

#SBATCH --partition=fat
#SBATCH --nodes=1
#SBATCH --mem=12gb
#SBATCH --time=35:00:00

#SBATCH --mail-user=tim-moritz.buendert@student.uni-tuebingen.de
#SBATCH --mail-type=BEGIN,END,FAIL

../python_env/bin/python3 0_filter_papers_cited.py --batches-start=${START} --batches-end=${END} --n=$1

EOT