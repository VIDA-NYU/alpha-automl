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


def main_search():
    setup_logging()
    timeout = None
    output_folder = None

    if 'D3MTIMEOUT' in os.environ:
        timeout = int(os.environ.get('D3MTIMEOUT'))

    if 'D3MOUTPUTDIR' in os.environ:
        output_folder = os.environ['D3MOUTPUTDIR']

    logger.info('Config loaded from environment variables D3MOUTPUTDIR=%r D3MTIMEOUT=%r',
                os.environ['D3MOUTPUTDIR'], os.environ.get('D3MTIMEOUT'))

    ta2 = AutoML(output_folder)
    dataset = '/input/TRAIN/dataset_TRAIN/datasetDoc.json'
    problem_path = '/input/TRAIN/problem_TRAIN/problemDoc.json'
    ta2.run_search(dataset, problem_path=problem_path, timeout=timeout)


def main_serve():
    setup_logging()
    port = None
    output_folder = None

    if len(sys.argv) == 2:
        port = int(sys.argv[1])

    if 'D3MOUTPUTDIR' in os.environ:
        output_folder = os.environ['D3MOUTPUTDIR']

    logger.info('Config loaded from environment variables D3MOUTPUTDIR=%r D3MTIMEOUT=%r',
                os.environ['D3MOUTPUTDIR'], os.environ.get('D3MTIMEOUT'))

    ta2 = AutoML(output_folder)
    ta2.run_server(port)
