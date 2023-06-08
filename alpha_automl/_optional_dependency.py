import importlib


def import_optional_dependency(dependency_name):
    try:
        dependency_module = importlib.import_module(dependency_name)
    except ImportError:
        raise ImportError(f'Missing optional dependency "{dependency_name}". Use pip or conda to install it.')

    return dependency_module
