import re
import json
import pickle
import logging
import copy
import pandas as pd
from os.path import join, dirname, exists
from alphad3m.primitive_loader import load_primitives_by_name, load_primitives_by_id, load_primitives_types

logger = logging.getLogger(__name__)

METALEARNING_DB_PATH = join(dirname(__file__), '../resource/metalearning_db.json.gz')
PRECALCULATED_TASKKEYWORDS_PATH = join(dirname(__file__), '../resource/precalculated_taskkeywords.json')


IGNORE_PRIMITIVES = {
    # These primitives are static elements in the pipelines, not considered as part of the pattern
    'd3m.primitives.data_transformation.construct_predictions.Common',
    'd3m.primitives.data_transformation.extract_columns_by_semantic_types.Common',
    'd3m.primitives.data_transformation.dataset_to_dataframe.Common',
    'd3m.primitives.data_transformation.denormalize.Common',
    'd3m.primitives.data_transformation.flatten.DataFrameCommon',
    'd3m.primitives.data_transformation.column_parser.Common',
    'd3m.primitives.data_transformation.simple_column_parser.DataFrameCommon',
    'd3m.primitives.data_transformation.do_nothing.DSBOX',
    'd3m.primitives.data_transformation.do_nothing_for_dataset.DSBOX',
    'd3m.primitives.data_transformation.add_semantic_types.Common',
    'd3m.primitives.schema_discovery.profiler.Common',
    'd3m.primitives.schema_discovery.profiler.DSBOX',
    'd3m.primitives.data_cleaning.column_type_profiler.Simon',
    'd3m.primitives.operator.compute_unique_values.Common',
    'd3m.primitives.data_transformation.construct_confidence.Common',
    # We add these primitives internally because they require special connections
    'd3m.primitives.data_transformation.text_reader.Common',
    'd3m.primitives.data_transformation.image_reader.Common'
}


def create_csv_data(metalearningdb_pickle_path, datasets_path, pipelines_csv_path,  use_primitive_names=True):
    logger.info('Creating CSV file...')
    primitives_by_name = load_primitives_by_name(only_installed_primitives=False)
    primitives_by_id = load_primitives_by_id(only_installed_primitives=False)
    primitives_types = load_primitives_types(only_installed_primitives=False)
    available_datasets = load_precalculated_data('task_keywords')
    ignore_primitives_ids = {primitives_by_name[ignore_primitive]['id'] for ignore_primitive in IGNORE_PRIMITIVES}
    train_pipelines = []
    total_pipelines = 0

    with open(metalearningdb_pickle_path, 'rb') as fin:
        all_pipelines = pickle.load(fin)

    for pipeline_run in all_pipelines:
        dataset_id = get_dataset_id(pipeline_run['problem']['id'])

        if dataset_id not in available_datasets:
            continue

        pipeline_primitives = pipeline_run['steps']
        pipeline_primitives = filter_primitives(pipeline_primitives, ignore_primitives_ids)  # Get the IDs of primitives
        task_keywords = load_task_keywords(dataset_id, datasets_path)

        if len(pipeline_primitives) > 0:
            score = pipeline_run['scores'][0]['normalized']
            metric = pipeline_run['scores'][0]['metric']['metric']
            try:
                pipeline_primitive_types = [primitives_types[primitives_by_id[p]] for p in pipeline_primitives]
                if use_primitive_names:
                    pipeline_primitives = [primitives_by_id[p] for p in pipeline_primitives]
                pipeline_stages = generate_pipeline_stages(pipeline_primitives, pipeline_primitive_types)
                pipeline_stages = [(' '.join(ps), ' '.join(task_keywords), metric, score) for ps in pipeline_stages]
                train_pipelines += pipeline_stages
                total_pipelines += 1
            except:
                logger.warning(f'Primitives "{str(pipeline_primitives)}" are not longer available')

    logger.info(f'Loaded {len(all_pipelines)} pipelines')
    logger.info(f'Found {total_pipelines} pipelines')
    pipelines_df = pd.DataFrame.from_records(train_pipelines, columns=['pipeline', 'task_keywords', 'metric', 'score'])
    pipelines_df.to_csv(pipelines_csv_path, index=False)


def generate_pipeline_stages(primitive_items, primitive_types):
    combinations = []
    pre_pipeline = primitive_types
    combinations.append(copy.deepcopy(pre_pipeline))

    for index, primitive_item in enumerate(primitive_items):
        pre_pipeline[index] = primitive_item
        combinations.append(copy.deepcopy(pre_pipeline))

    return combinations


def load_task_keywords(dataset_id, datasets_path):
    possible_names = [join(datasets_path, dataset_id), join(datasets_path, dataset_id + '_MIN_METADATA'),
                      join(datasets_path, dataset_id.replace('_MIN_METADATA', ''))]
    # All possible names of the datasets on disk, with/without the suffix 'MIN_METADATA'

    for dataset_folder_path in possible_names:
        if exists(dataset_folder_path):
            break
    else:
        raise FileNotFoundError(f'Dataset {dataset_id} not found')

    problem_path = join(dataset_folder_path, 'TRAIN/problem_TRAIN/problemDoc.json')

    with open(problem_path) as fin:
        problem_doc = json.load(fin)
        task_keywords = problem_doc['about']['taskKeywords']

        return task_keywords


def get_dataset_id(problem_id):
    # Remove suffixes 'TRAIN' and 'problem' from the dataset name
    dataset_id = re.sub('_TRAIN$', '', problem_id)
    dataset_id = re.sub('_problem$', '', dataset_id)

    return dataset_id


def filter_primitives(pipeline_steps, ignore_primitives):
    primitives = []

    for pipeline_step in pipeline_steps:
        if pipeline_step['primitive']['id'] not in ignore_primitives:
                primitives.append(pipeline_step['primitive']['id'])

    if len(primitives) > 0 and primitives[0] == '7ddf2fd8-2f7f-4e53-96a7-0d9f5aeecf93':
        # Special case: Primitive to_numeric.DSBOX
        # This primitive should not be first because it only takes the numeric features, ignoring the remaining ones
        primitives = primitives[1:]

    return primitives


def load_precalculated_data(mode):
    if mode == 'task_keywords':
        file_path = PRECALCULATED_TASKKEYWORDS_PATH
    else:
        raise ValueError(f'Unknown mode "{mode}" to load data')

    if exists(file_path):
        with open(file_path) as fin:
            return json.load(fin)

    return {}


if __name__ == '__main__':
    # Download the metalearningdb.pkl file from https://drive.google.com/file/d/1WjY7iKkkKMZFeoiCqzamA_iqVOwQidXS/view
    # and D3M datasets from https://datasets.datadrivendiscovery.org/d3m/datasets/-/tree/master/seed_datasets_current
    metalearningdb_pickle_path = '/Users/rlopez/D3M/metalearning_db/metalearningdb.pkl'
    datasets_root_path = '/Users/rlopez/D3M/datasets/seed_datasets_current/'
    pipelines_csv_path = '/Users/rlopez/D3M/metalearning_db/marvin_pipelines.csv'

    create_csv_data(metalearningdb_pickle_path, datasets_root_path, pipelines_csv_path)
