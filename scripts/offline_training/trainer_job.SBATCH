#!/bin/bash
#SBATCH -c 4
#SBATCH --mem 128GB
#SBATCH --time 12:00:00
#SBATCH --mail-user=yfw215@nyu.edu

singularity exec --overlay overlay-50G-10M.ext3:ro /scratch/work/public/singularity/ubuntu-20.04.4.sif /bin/bash -c "source /ext3/env.sh; python trainer.py -o results -t ${1} -T 5"