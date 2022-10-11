#!/bin/bash

sbatch <<EOT
#!/bin/sh

#SBATCH --job-name=pos_samples.job
#SBATCH --output=out/pos_samples.out
#SBATCH --export=All

#SBATCH --partition=fat
#SBATCH --nodes=1

#SBATCH --mem=75gb
#SBATCH --time=1:00:00

#SBATCH --mail-user=tim-moritz.buendert@student.uni-tuebingen.de
#SBATCH --mail-type=BEGIN,END,FAIL

../python_env/bin/python3 1_pos_samples.py

EOT