#!/bin/bash


sbatch <<EOT
#!/bin/sh

#SBATCH --job-name=texts.job
#SBATCH --output=texts.out
#SBATCH --export=All

#SBATCH --partition=fat
#SBATCH --nodes=1

#SBATCH --mem=30gb
#SBATCH --time=1:00:00

#SBATCH --mail-user=tim-moritz.buendert@student.uni-tuebingen.de
#SBATCH --mail-type=BEGIN,END,FAIL

../python_env/bin/python3 texts.py

EOT
