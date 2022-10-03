#!/bin/bash

SPLIT=$1

sbatch <<EOT
#!/bin/sh

#SBATCH --job-name=serialize_${SPLIT}.job
#SBATCH --output=out/serialize_${SPLIT}.out
#SBATCH --export=All

#SBATCH --partition=fat
#SBATCH --nodes=1

#SBATCH --mem=10gb
#SBATCH --time=30:00

#SBATCH --mail-user=tim-moritz.buendert@student.uni-tuebingen.de
#SBATCH --mail-type=BEGIN,END,FAIL

../python_env/bin/python3 1_serialize_samples.py \
    --split=${SPLIT} 
fi

EOT

# sh serialize_samples.sh {pos|neg}
