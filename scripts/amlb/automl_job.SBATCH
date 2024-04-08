#!/bin/bash
#SBATCH -c 4
#SBATCH --mem 128GB
#SBATCH --time 12:00:00
#SBATCH --output logs/automl_job_%J.out
#SBATCH --mail-user=yfw215@nyu.edu
#SBATCH --no-kill

singularity exec --overlay overlay-50G-10M.ext3:ro /scratch/work/public/singularity/ubuntu-22.04.sif /bin/bash -c "source /ext3/env.sh; python automlbenchmark/runbenchmark.py ${1} ${2} 1h4c -f 0 -u user_config/ -i openml_datasets/ -o results/ -s skip"
