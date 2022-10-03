#!/bin/bash

sbatch <<EOT
#!/bin/sh

#SBATCH --job-name=explanations.job
#SBATCH --output=out/explanations.out
#SBATCH --export=All

#SBATCH --partition=fat
#SBATCH --nodes=1

#SBATCH --mem=100gb
#SBATCH --time=1:00:00

#SBATCH --mail-user=tim-moritz.buendert@student.uni-tuebingen.de
#SBATCH --mail-type=BEGIN,END,FAIL

../python_env/bin/python3 2_explanations.py

EOT

# sh 2_explanations.sh 
