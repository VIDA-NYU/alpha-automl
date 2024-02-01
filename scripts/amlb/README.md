# AMLB Experiments



We used the AutoML Benchmark (AMLB) to run these experiments locally using Python 3.9. To reproduce these experiments, 
follow the next steps.


1. Clone and install AMLB. See the [AMLB repository](https://github.com/openml/automlbenchmark/tree/v2.0.6/) 
for additional details about the installation. 

```
git clone https://github.com/openml/automlbenchmark.git --branch stable --depth 1
cd automlbenchmark
pip install -r requirements.txt
cd ..
```

2. Create the *openml_datasets* and *results* folder (AMLB will use theses folders).
```
mkdir openml_datasets
mkdir results
```


3. To test the installation, run the following command. You should get valid ML pipelines after running it.
```
 python automlbenchmark/runbenchmark.py Alpha-AutoML  openml/t/12  test -f 0 -u user_config/ -i openml_datasets/ -o results/
```

4. We ran all the systems (AutoWEKA, TPOT, H2O, AutoGluon, Auto-Sklearn, and AlphaD3M) using Singularity containers in 
SLURM batch jobs in the [NYU Greene Cluster](https://sites.google.com/nyu.edu/nyu-hpc/hpc-systems/greene). To run the 
experiments in this cluster, run `bash ./run_all_automlbenchmark.sh`.
All the results will be stored in the `./results/results.csv` file.



