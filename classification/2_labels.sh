#!/bin/bash

sbatch <<EOT
#!/bin/sh

#SBATCH --job-name=labels.job
#SBATCH --output=out/labels.out
#SBATCH --export=All

#SBATCH --partition=fat
#SBATCH --nodes=1

#SBATCH --mem=60gb
#SBATCH --time=20:00:00

#SBATCH --mail-user=tim-moritz.buendert@student.uni-tuebingen.de
#SBATCH --mail-type=BEGIN,END,FAIL

../python_env/bin/python3 2_labels.py

EOT

# sh labels.sh 
