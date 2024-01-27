#!/bin/bash
#SBATCH -c 4
#SBATCH --mem 32GB
#SBATCH --time 01:15:00
#SBATCH --output logs/automl_job_%J.out
#SBATCH --mail-user=rl3725@nyu.edu

singularity exec --bind /scratch/rl3725/alphaautoml_experiments/experiments:/scratch/rl3725/alphaautoml_experiments/experiments --overlay overlay-15GB-500K.ext3:rw /scratch/work/public/singularity/ubuntu-20.04.4.sif /bin/bash -c "source /ext3/env.sh; python automlbenchmark/runbenchmark.py ${1} ${2} 1h4c -f 0 -u user_config/ -i openml_datasets/ -o results/"
