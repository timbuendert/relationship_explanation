#!/bin/bash

SPLIT=500000
START=$(($1 * ${SPLIT}))
END=$((($1 * ${SPLIT}) + ${SPLIT}))

sbatch <<EOT
#!/bin/sh

#SBATCH --job-name=combine_"$1".job
#SBATCH --output=combine_"$1".out
#SBATCH --export=ALL

#SBATCH --partition=fat
#SBATCH --nodes=1
#SBATCH --mem=300gb
#SBATCH --time=15:00:00

#SBATCH --mail-user=tim-moritz.buendert@student.uni-tuebingen.de
#SBATCH --mail-type=BEGIN,END,FAIL

../python_env/bin/python3 4_combine_final.py --start=${START} --end=${END} --n=$1

# total: 1,634,806 -> with SPLIT = 500,000: 0,1,2 & 3: 1500000 to 1634806

EOT