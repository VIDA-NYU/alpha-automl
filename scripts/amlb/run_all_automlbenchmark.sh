#!/bin/bash
  
datasets="openml/t/10101 openml/t/12 openml/t/146195 openml/t/146212 openml/t/146606 openml/t/146818 openml/t/146821 openml/t/146822 openml/t/146825 openml/t/14965 openml/t/167119 openml/t/167120 openml/t/168329 openml/t/168330 openml/t/168331 openml/t/168332 openml/t/168335 openml/t/168337 openml/t/168338 openml/t/168868 openml/t/168908 openml/t/168909 openml/t/168910 openml/t/168911 openml/t/168912 openml/t/189354 openml/t/189355 openml/t/189356 openml/t/3 openml/t/31 openml/t/34539 openml/t/3917 openml/t/3945 openml/t/53 openml/t/7592 openml/t/7593 openml/t/9952 openml/t/9977 openml/t/9981"
systems="autosklearn AutoGluon TPOT H2OAutoML AutoWEKA Alpha-AutoML"

for system in $systems
do
   for dataset in $datasets
   do
          echo "Running ${system} system in ${dataset} dataset"
          sbatch automl_job.SBATCH $system $dataset
   done
done