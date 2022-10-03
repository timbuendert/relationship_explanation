#!/bin/bash

sbatch <<EOT
#!/bin/sh

#SBATCH --job-name=tag_"$1".job
#SBATCH --output=tag_"$1".out
#SBATCH --export=ALL

#SBATCH --partition=fat
#SBATCH --nodes=1
#SBATCH --mem=20gb
#SBATCH --time=40:00:00

#SBATCH --mail-user=tim-moritz.buendert@student.uni-tuebingen.de
#SBATCH --mail-type=BEGIN,END,FAIL

../python_env/bin/python3 pipeline.py --related_work_file='../related_works/related_work"$1".jsonl' --output_file='related_works_tagged/tagged_related_works"$1".jsonl'

EOT