#!/bin/bash

sbatch <<EOT
#!/bin/sh

#SBATCH --job-name=pos_samples.job
#SBATCH --output=out/pos_samples.out
#SBATCH --export=All

#SBATCH --partition=fat
#SBATCH --nodes=1

#SBATCH --mem=175gb
#SBATCH --time=1:00:00

#SBATCH --mail-user=tim-moritz.buendert@student.uni-tuebingen.de
#SBATCH --mail-type=BEGIN,END,FAIL

../python_env/bin/python3 1_pos_samples.py

EOT

# sh 1_pos_samples.sh

# for intro_entity: use 1_serialize_samples.py (or python_env_dygie and pos_samples_ie.pkl)