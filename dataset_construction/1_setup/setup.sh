#!/bin/bash

sbatch <<EOT
#!/bin/sh

#SBATCH --job-name=combine_"$1".job
#SBATCH --output=combine_"$1".out
#SBATCH --export=ALL

#SBATCH --partition=fat
#SBATCH --nodes=1
#SBATCH --mem=60gb
#SBATCH --time=12:00:00

#SBATCH --mail-user=tim-moritz.buendert@student.uni-tuebingen.de
#SBATCH --mail-type=BEGIN,END,FAIL

../python_env/bin/python3 1_setup_relWork.py --n=$1

EOT