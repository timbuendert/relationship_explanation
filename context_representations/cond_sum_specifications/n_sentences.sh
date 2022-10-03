#!/bin/bash

sbatch <<EOT
#!/bin/sh

#SBATCH --job-name=n_sent.job
#SBATCH --output=out/n_sent.out
#SBATCH --export=All

#SBATCH --partition=fat
#SBATCH --nodes=1

#SBATCH --mem=30gb
#SBATCH --time=20:00

#SBATCH --mail-user=tim-moritz.buendert@student.uni-tuebingen.de
#SBATCH --mail-type=BEGIN,END,FAIL

../python_env/bin/python3 n_sentences.py

EOT