#!/bin/bash

sbatch <<EOT
#!/bin/sh

#SBATCH --job-name=add_"$1".job
#SBATCH --output=add_"$1".out
#SBATCH --export=ALL

#SBATCH --partition=fat
#SBATCH --nodes=1
#SBATCH --mem=105gb
#SBATCH --time=20:00:00

#SBATCH --mail-user=tim-moritz.buendert@student.uni-tuebingen.de
#SBATCH --mail-type=BEGIN,END,FAIL

../python_env/bin/python3 4_add_data.py --type=$1

EOT

# types: p_title, p_text, c_title, c_text
