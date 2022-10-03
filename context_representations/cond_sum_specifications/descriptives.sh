#!/bin/bash

sbatch <<EOT
#!/bin/sh

#SBATCH --job-name=descriptives.job
#SBATCH --output=out/descriptives.out
#SBATCH --export=All

#SBATCH --partition=fat
#SBATCH --nodes=1

#SBATCH --mem=3gb
#SBATCH --time=10:00

#SBATCH --mail-user=tim-moritz.buendert@student.uni-tuebingen.de
#SBATCH --mail-type=BEGIN,END,FAIL

../python_env/bin/python3 descriptives.py

EOT