#!/bin/bash
  
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --mem=128GB
#SBATCH --job-name=torch
#SBATCH --output=fasttext_run_logs.out

module purge

singularity exec \
            --overlay /scratch/mp5583/automl_env/overlay-50G-10M.ext3:rw \
            /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif \
            /bin/bash -c "source /ext3/env.sh; python /scratch/mp5583/alpha-automl-latest/alpha-automl/examples/adding_huggingface_primitives.py"
