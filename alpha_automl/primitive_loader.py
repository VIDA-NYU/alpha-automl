import logging
import json
from os.path import join, dirname

logger = logging.getLogger(__name__)

PRIMITIVES_HIERARCHY_PATH = join(dirname(__file__), 'resource/primitives_hierarchy.json')


def create_primitives_hierarchy():
    pass


def load_primitives_hierarchy():
    with open(PRIMITIVES_HIERARCHY_PATH) as fin:
        primitives = json.load(fin)
    logger.info('Hierarchy of all primitives loaded')

    return primitives


def load_primitives_types():
    primitive_types = {}
    with open(PRIMITIVES_HIERARCHY_PATH) as fin:
        primitives = json.load(fin)

    for primitive_type, primitive_names in primitives.items():
        for primitive_name in primitive_names:
            primitive_types[primitive_name] = primitive_type

    logger.info('Primitive types loaded')

    return primitive_types


PRIMITIVE_TYPES = load_primitives_types()

if __name__ == '__main__':
    # Run this to create the files for the list and hierarchy of primitives
    create_primitives_hierarchy()

