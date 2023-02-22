import logging
import json
from os.path import join, dirname

logger = logging.getLogger(__name__)

PRIMITIVES_HIERARCHY_PATH = join(dirname(__file__), 'resource/primitives_hierarchy.json')


def get_primitive_type(primitive_name):
    primitive_type = None
    # Changing the primitive families using some predefined rules
    if primitive_name in {'d3m.primitives.data_cleaning.quantile_transformer.SKlearn',
                          'd3m.primitives.data_cleaning.normalizer.SKlearn',
                          'd3m.primitives.normalization.iqr_scaler.DSBOX'}:
        primitive_type = 'FEATURE_SCALING'

    elif primitive_name in {'d3m.primitives.feature_extraction.feature_agglomeration.SKlearn',
                            'd3m.primitives.feature_selection.mutual_info_classif.DistilMIRanking'}:
        primitive_type = 'FEATURE_SELECTION'



    return primitive_type


def create_primitives_hierarchy():
    pass

def load_primitives_hierarchy():
    with open(PRIMITIVES_HIERARCHY_PATH) as fin:
        primitives = json.load(fin)
    logger.info('Hierarchy of all primitives loaded')

    return primitives


if __name__ == '__main__':
    # Run this to create the files for the list and hierarchy of primitives
    create_primitives_hierarchy()

