'''Entrypoints for the TA2.

This contains the multiple entrypoints for the TA2, that get called from
different commands. They spin up a D3mTa2 object and use it.
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
    port = None
    output_folder = None

    if len(sys.argv) == 3:
        output_folder = sys.argv[1]
        port = int(sys.argv[2])

    # TODO: We use env variables in the code of AlphaD3M. So, temporally, we need to setup them,
    #  but we should get rid of them
    os.environ['D3MOUTPUTDIR'] = output_folder

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


def main_serve_dmc():
    setup_logging()
    port = None
    output_folder = None

    if len(sys.argv) == 2:
        port = int(sys.argv[1])  # TODO: Read the port from the env variable

    if 'D3MOUTPUTDIR' in os.environ:
        output_folder = os.environ['D3MOUTPUTDIR']

    logger.info('Config loaded from environment variables D3MOUTPUTDIR=%r', os.environ['D3MOUTPUTDIR'])

    automl = AutoML(output_folder)
    automl.run_server(port)






