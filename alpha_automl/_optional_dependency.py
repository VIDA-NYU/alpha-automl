import importlib
import logging

logger = logging.getLogger(__name__)


def import_optional_dependency(dependency_name):
    dependency_module = None
    try:
        dependency_module = importlib.import_module(dependency_name)
    except ImportError:
        logging.warning(f'Missing optional dependency "{dependency_name}". Use pip or conda to install it.')

    return dependency_module


def check_optional_dependency(dependency_name, ml_task):
    spec = importlib.util.find_spec(dependency_name)

    if spec is None:
        logging.warning(f'You tried to use "{dependency_name}", which is a optional dependency". '
                        f'Use: "pip install alpha-automl[{ml_task}]" to use primitives with {dependency_name}')
