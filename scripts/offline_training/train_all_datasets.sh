#!/bin/bash

#datasets="10101 12 146195 146212 146606 146818 146821 146822 146825 14965 167119 167120 168329 168330 168331 168332 168335 168337 168338 168868 168908 168909 168910 168911 168912 189354 189355 189356 3 31 34539 3917 3945 53 7592 7593 9952 9977 9981"
# datasets="146825 168329 168330 168331 168332"
datasets="168910"
systems="Alpha-AutoML"

rm -rf tmp/logs/*
for dataset in $datasets
do
    echo "Training Alpha-AutoML for ${dataset} dataset"
    sbatch --output results/logs/automl_job_${dataset}.out trainer_job.SBATCH $dataset
done
