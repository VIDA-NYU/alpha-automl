'''Entrypoints for AlphaD3M.

This contains the multiple entrypoints for AlphaD3M, that get called from
different commands. They spin up a AutoML object and use it.
'''

import logging
import os
import sys
from alphad3m.automl import AutoML


logger = logging.getLogger(__name__)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s:%(levelname)s:AlphaD3M:%(name)s:%(message)s')


def main_serve():
    setup_logging()

    port = int(os.environ.get('D3MPORT', 45042))
    output_folder = os.environ['D3MOUTPUTDIR']

    automl = AutoML(output_folder)
    automl.run_server(port)


def main_search():
    setup_logging()
    timeout = None
    output_folder = None

    if len(sys.argv) == 3:
        output_folder = sys.argv[1]
        timeout = int(sys.argv[2])

    dataset = '/input/TRAIN/dataset_TRAIN/datasetDoc.json'
    problem_path = '/input/TRAIN/problem_TRAIN/problemDoc.json'
    automl = AutoML(output_folder)
    automl.run_search(dataset, problem_path=problem_path, timeout=timeout)







