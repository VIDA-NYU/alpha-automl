import pickle
import logging
import copy
import pandas as pd
from alphad3m.primitive_loader import load_primitives_by_name, load_primitives_by_id, load_primitives_types
from alphad3m.metalearning.resource_builder import get_dataset_id, filter_primitives, load_precalculated_data
from alphad3m.metalearning.dataset_profiler import DEFAULT_METAFEATURES

logger = logging.getLogger(__name__)


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


def create_csv_data(metalearningdb_pickle_path, pipelines_csv_path,  use_primitive_names=True):
    logger.info('Creating CSV file...')
    primitives_by_name = load_primitives_by_name(only_installed_primitives=False)
    primitives_by_id = load_primitives_by_id(only_installed_primitives=False)
    primitives_types = load_primitives_types(only_installed_primitives=False)
    dataset_task_keywords = load_precalculated_data('task_keywords')
    dataset_semantic_types = load_precalculated_data('dataprofiles')
    dataset_metafeatures = load_precalculated_data('metafeatures')
    ignore_primitives_ids = {primitives_by_name[ignore_primitive]['id'] for ignore_primitive in IGNORE_PRIMITIVES}
    train_pipelines = []
    total_pipelines = 0

    with open(metalearningdb_pickle_path, 'rb') as fin:
        all_pipelines = pickle.load(fin)

    for pipeline_run in all_pipelines:
        dataset_id = get_dataset_id(pipeline_run['problem']['id'])

        if dataset_id not in dataset_task_keywords:
            continue

        pipeline_primitives = pipeline_run['steps']
        pipeline_primitives = filter_primitives(pipeline_primitives, ignore_primitives_ids)  # Get the IDs of primitives
        task_keywords = ' '.join(dataset_task_keywords[dataset_id]['task_keywords'])
        semantic_types = ' '.join(dataset_semantic_types[dataset_id]['feature_types'])
        metafeatures = [dataset_metafeatures[dataset_id][mf]['value'] for mf in DEFAULT_METAFEATURES]

        if len(pipeline_primitives) > 0:
            score = pipeline_run['scores'][0]['normalized']
            metric = pipeline_run['scores'][0]['metric']['metric']
            try:
                pipeline_primitive_types = [primitives_types[primitives_by_id[p]] for p in pipeline_primitives]
                if use_primitive_names:
                    pipeline_primitives = [primitives_by_id[p] for p in pipeline_primitives]
                pipeline_stages = generate_pipeline_stages(pipeline_primitives, pipeline_primitive_types)
                pipeline_stages = [[' '.join(ps), task_keywords, semantic_types, metric, score] + metafeatures
                                   for ps in pipeline_stages]
                train_pipelines += pipeline_stages
                total_pipelines += 1
            except Exception:
                logger.warning(f'Primitives "{str(pipeline_primitives)}" are not longer available')

    logger.info(f'Loaded {len(all_pipelines)} pipelines')
    logger.info(f'Found {total_pipelines} pipelines')
    pipelines_df = pd.DataFrame.from_records(train_pipelines, columns=['primitives', 'task_keywords', 'semantic_types',
                                                                       'metric', 'score'] + DEFAULT_METAFEATURES)
    pipelines_df.to_csv(pipelines_csv_path, index=False)


def generate_pipeline_stages(primitive_items, primitive_types):
    combinations = []
    pre_pipeline = primitive_types
    combinations.append(copy.deepcopy(pre_pipeline))

    for index, primitive_item in enumerate(primitive_items):
        pre_pipeline[index] = primitive_item
        combinations.append(copy.deepcopy(pre_pipeline))

    return combinations


if __name__ == '__main__':
    # Download the metalearningdb.pkl file from https://drive.google.com/file/d/1WjY7iKkkKMZFeoiCqzamA_iqVOwQidXS/view
    # and D3M datasets from https://datasets.datadrivendiscovery.org/d3m/datasets/-/tree/master/seed_datasets_current
    metalearningdb_pickle_path = '/Users/rlopez/D3M/metalearning_db/metalearningdb.pkl'
    pipelines_csv_path = '/Users/rlopez/D3M/metalearning_db/marvin_pipelines.csv'

    create_csv_data(metalearningdb_pickle_path, pipelines_csv_path)
