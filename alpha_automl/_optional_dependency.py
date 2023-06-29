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
