#!/bin/bash

N=$1
SEED=42

sbatch <<EOT
#!/bin/sh

#SBATCH --job-name=outputs_${N}.job
#SBATCH --output=outputs_${N}.out
#SBATCH --export=All

#SBATCH --partition=fat
#SBATCH --nodes=1

#SBATCH --mem=2gb
#SBATCH --time=10:00

#SBATCH --mail-user=tim-moritz.buendert@student.uni-tuebingen.de
#SBATCH --mail-type=BEGIN,END,FAIL


../python_env/bin/python3 retrieve_samples.py --n_samples=${N} --seed=${SEED}

EOT