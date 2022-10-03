#!/bin/bash

N=$1
SEED=42

sbatch <<EOT
#!/bin/sh

#SBATCH --job-name=outputs_all_${N}.job
#SBATCH --output=outputs_all_${N}.out
#SBATCH --export=All

#SBATCH --partition=fat
#SBATCH --nodes=1

#SBATCH --mem=1gb
#SBATCH --time=20:00

#SBATCH --mail-user=tim-moritz.buendert@student.uni-tuebingen.de
#SBATCH --mail-type=BEGIN,END,FAIL

../python_env/bin/python3 retrieve_outputs.py --n_samples=${N} --seed=${SEED}


EOT

# sh retrieve_outputs.sh {N}