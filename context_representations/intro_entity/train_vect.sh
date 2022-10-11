#!/bin/bash

INTENT=reflection

sbatch <<EOT
#!/bin/sh

#SBATCH --job-name=tfidf_vec.job
#SBATCH --output=tfidf_vec.out
#SBATCH --export=ALL

#SBATCH --partition=fat
#SBATCH --nodes=1
#SBATCH --mem=500gb
#SBATCH --time=12:00:00

#SBATCH --mail-user=tim-moritz.buendert@student.uni-tuebingen.de
#SBATCH --mail-type=BEGIN,END,FAIL

../../python_env_dygie/bin/python3 train_vect.py --intent=${INTENT}

EOT