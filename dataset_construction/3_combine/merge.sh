#!/bin/bash

#SBATCH --job-name=merge.job
#SBATCH --output=merge.out
#SBATCH --export=ALL

#SBATCH --partition=fat
#SBATCH --nodes=1
#SBATCH --mem=150gb
#SBATCH --time=02:00:00

#SBATCH --mail-user=tim-moritz.buendert@student.uni-tuebingen.de
#SBATCH --mail-type=BEGIN,END,FAIL

../python_env/bin/python3 4_merge_parts.py --filter --Reflection