[![PyPI version](https://badge.fury.io/py/alpha-automl.svg)](https://pypi.org/project/alpha-automl)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Tests](https://github.com/VIDA-NYU/alpha-automl/actions/workflows/build.yml/badge.svg)](https://github.com/VIDA-NYU/alpha-automl/actions/workflows/build.yml)
[![Documentation Status](https://readthedocs.org/projects/alpha-automl/badge/?version=latest)](https://alpha-automl.readthedocs.io/en/latest/?badge=latest)


<img src="https://github.com/VIDA-NYU/alpha-automl/raw/devel/Alpha-AutoML_logo.png" width=30%>


Alpha-AutoML is an AutoML system that automatically searches for models and derives end-to-end pipelines that read, 
pre-process the data, and train the model. Alpha-AutoML leverages recent advances in deep reinforcement learning and is 
able to adapt to different application domains and problems through incremental learning.

Alpha-AutoML provides data scientists and data engineers the flexibility to address complex problems by leveraging the 
Python ecosystem, including open-source libraries and tools, support for collaboration, and infrastructure that enables 
transparency and reproducibility. 

This repository is part of New York University's implementation of the 
[Data Driven Discovery project (D3M)](https://datadrivendiscovery.org/).


## Documentation

Documentation is available [here](https://alpha-automl.readthedocs.io/). You can also try this [online demo](https://alpha-automl.hsrn.nyu.edu/) (1 hour limit).


## Installation

This package works with Python 3.6+ in Linux, Mac, and Windows.

You can install the latest stable version of this library from [PyPI](https://pypi.org/project/alpha-automl/):

```
pip install alpha-automl
```

To install the latest development version:

```
pip install git+https://github.com/VIDA-NYU/alpha-automl@devel
```


## Trying it using Docker

We provide pre-built docker images with Jupyter and Alpha-AutoML pre-installed that you can use to quickly test Alpha-AutoML.
To test it, you can run the following command in your machine, and open Jupyter Notebook on your browser:

```
docker run -p 8888:8888 ghcr.io/vida-nyu/alpha-automl
```
Using this command, Jupyter Notebook will auto-generate a security token. The correct URL to access the Jupyter will be printed in the console output and will look like: `http://127.0.0.1:8888/?token=70ace7fa017c35ba0134dc7931add12bf55a69d4d4e6e54f`.

Alternatively, if you want to provide a custom security token, you can run:
```
docker run -p 8888:8888 -e JUPYTER_TOKEN="<my-token>" ghcr.io/vida-nyu/alpha-automl
```

If you are running the Jupyter Notebook in a secure environment, the authentication can be disabled as follows:
```
docker run -p 8888:8888 ghcr.io/vida-nyu/alpha-automl --NotebookApp.token=''
```

## Building a Docker image from scratch

If you need to build an image from sources, you can use our [Dockerfile](https://github.com/VIDA-NYU/alpha-automl/blob/devel/Dockerfile). You can use a docker-build argument to select the packages that will be installed in the image (e.g., `full`, `timeseries`, `nlp`, etc) as follows:

```
docker build -t alpha-automl --build-arg BUILD_OPTION=full .
```

Or simply a base version using (this will use less disk space but will not provide support for some tasks such as NLP and timeseries):
```
docker build -t alpha-automl:latest --target alpha-automl .
```

You can also build an image to use with JupyterHub as follows:
```
docker build -t alpha-automl:latest-jupyterhub --target alpha-automl-jupyterhub .
```
See also the documentation on how to setup Alpha-AutoML + JupyterHub on [Kubernetes](https://github.com/VIDA-NYU/alpha-automl/tree/devel/kubernetes).