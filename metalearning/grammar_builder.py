import logging
import statistics
from collections import OrderedDict
from alphad3m.primitive_loader import load_primitives_list
from alphad3m.metalearning.database import load_metalearningdb

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


IGNORE_PRIMITIVES = {'d3m.primitives.data_transformation.construct_predictions.Common',
                     'd3m.primitives.data_transformation.extract_columns_by_semantic_types.Common',
                     'd3m.primitives.data_transformation.dataset_to_dataframe.Common',
                     'd3m.primitives.data_transformation.denormalize.Common',
                     'd3m.primitives.data_transformation.flatten.DataFrameCommon',
                     'd3m.primitives.data_transformation.column_parser.Common',
                     'd3m.primitives.data_transformation.do_nothing.DSBOX',
                     'd3m.primitives.schema_discovery.profiler.DSBOX',
                     'd3m.primitives.data_cleaning.column_type_profiler.Simon',
                     'd3m.primitives.data_transformation.text_reader.Common',
                     'd3m.primitives.data_transformation.image_reader.Common',
                     'd3m.primitives.data_transformation.audio_reader.Common',
                     'd3m.primitives.data_transformation.dataframe_to_tensor.DSBOX'
                     }


def load_pipelines(task_keywords):
    primitives_by_id = load_primitives_by_id()
    primitives_by_name = load_primitives_by_name()
    all_pipelines = load_metalearningdb()
    ignore_primitives_ids = set()
    task_pipelines = []

    for ignore_primitive in IGNORE_PRIMITIVES:
        if ignore_primitive in primitives_by_name:
            ignore_primitives_ids.add(primitives_by_name[ignore_primitive]['id'])

    for pipeline_run in all_pipelines:
        pipeline_primitives = pipeline_run['steps']
        if is_target_task(pipeline_run['problem'], task_keywords) and is_available_primitive(pipeline_primitives, primitives_by_id):
            primitives = filter_primitives(pipeline_primitives, ignore_primitives_ids)
            primitives = [primitives_by_id[p] for p in primitives]  # Use the current names of primitives
            if len(primitives) > 0:
                score = pipeline_run['scores'][0]['value']
                metric = pipeline_run['scores'][0]['metric']['metric']
                dataset = pipeline_run['problem']['id']
                task_pipelines.append({'pipeline': primitives, 'score': score, 'metric': metric, 'dataset': dataset,
                                       'pipeline_repr': '_'.join(primitives)})

    logger.info('Found %d pipelines for task %s', len(task_pipelines), '_'.join(task_keywords))

    return task_pipelines


def create_grammar_from_metalearningdb(task_name, task_keywords):
    pipelines = load_pipelines(task_keywords)
    patterns, hierarchy_primitives = extract_patterns(pipelines)
    patterns, empty_elements = merge_patterns(patterns)
    grammar = format_grammar(task_name, patterns, empty_elements)

    return grammar, hierarchy_primitives


def format_grammar(task_name, patterns, empty_elements):
    grammar = 'S -> %s\n' % task_name
    grammar += task_name + ' -> ' + ' | '.join([' '.join(p) for p in patterns])

    for element in set([item for sublist in patterns for item in sublist]):
        production_rule = element + " -> 'primitive_terminal'"
        if element in empty_elements:
            production_rule += " | 'E'"

        grammar += '\n' + production_rule
    logger.info('Grammar obtained:\n%s', grammar)

    return grammar


def extract_patterns(pipelines, combine_encoders=False, min_frequency=5, avg_performance=0.5):
    available_primitives = load_primitives_by_name()
    pipelines = calculate_adtm(pipelines)
    patterns = {}

    for pipeline_data in pipelines:
        if pipeline_data['adtm'] > avg_performance:
            continue

        primitive_types = [available_primitives[p]['type'] for p in pipeline_data['pipeline']]
        if combine_encoders:
            primitive_types = combine_type_encoders(primitive_types)
        pattern_id = ' '.join(primitive_types)
        if pattern_id not in patterns:
            patterns[pattern_id] = {'structure': primitive_types, 'primitives': [], 'scores': [], 'frequency': 0}
        patterns[pattern_id]['primitives'].append(pipeline_data['pipeline'])
        patterns[pattern_id]['scores'].append(pipeline_data['score'])
        patterns[pattern_id]['frequency'] += 1
    logger.info('Found %d different patterns', len(patterns))
    # TODO: Group these removing conditions into a single loop
    # Remove patterns with fewer elements than the minimum frequency
    patterns = {k: v for k, v in patterns.items() if v['frequency'] >= min_frequency}
    logger.info('Found %d different patterns, after removing uncommon patterns', len(patterns))

    # Remove patterns with low performances
    blacklist_primitive_types = {'OPERATOR', 'ARRAY_CONCATENATION'}
    patterns = {k: v for k, v in patterns.items() if not blacklist_primitive_types & set(v['structure'])}
    logger.info('Found %d different patterns, after blacklisting primitive types', len(patterns))

    for pattern_id in patterns:
        scores = patterns[pattern_id].pop('scores')
        patterns[pattern_id]['average'] = statistics.mean(scores)

    # Remove patterns with low performances
    patterns = {k: v for k, v in patterns.items() if v['average'] >= avg_performance}
    logger.info('Found %d different patterns, after removing low-performance patterns', len(patterns))

    hierarchy_primitives = {}

    for pattern in patterns.values():
        for pipeline in pattern['primitives']:
            for primitive in pipeline:
                primitive_type = available_primitives[primitive]['type']
                if primitive_type not in hierarchy_primitives:
                    hierarchy_primitives[primitive_type] = set()
                hierarchy_primitives[primitive_type].add(primitive)

    patterns = sorted(patterns.values(), key=lambda x: x['average'], reverse=True)
    logger.info('Patterns:\n%s', '\n'.join(['structure: %s, frequency: %d' % (str(x['structure']), x['frequency']) for x in patterns]))
    patterns = [p['structure'] for p in patterns]

    return patterns, hierarchy_primitives


def calculate_adtm(pipelines):
    dataset_performaces = {}
    pipeline_performances = {}

    for pipeline_data in pipelines:
        id_dataset = pipeline_data['dataset'] + '_' + pipeline_data['metric']

        if id_dataset not in dataset_performaces:
            dataset_performaces[id_dataset] = {'min': float('inf'), 'max': 0}
        performance = pipeline_data['score']

        if performance > dataset_performaces[id_dataset]['max']:
            dataset_performaces[id_dataset]['max'] = performance

        if performance < dataset_performaces[id_dataset]['min']:
            dataset_performaces[id_dataset]['min'] = performance

        id_pipeline = pipeline_data['pipeline_repr']

        if id_pipeline not in pipeline_performances:
            pipeline_performances[id_pipeline] = {}

        if id_dataset not in pipeline_performances[id_pipeline]:
            pipeline_performances[id_pipeline][id_dataset] = pipeline_data['score']
        else:
            # A pipeline can have different performances for a given dataset, choose the best one
            if pipeline_data['score'] > pipeline_performances[id_pipeline][id_dataset]:
                pipeline_performances[id_pipeline][id_dataset] = pipeline_data['score']

    for pipeline_data in pipelines:
        id_pipeline = pipeline_data['pipeline_repr']
        id_dataset_pipeline = pipeline_data['dataset'] + '_' + pipeline_data['metric']
        dtm = 0

        for id_dataset in pipeline_performances[id_pipeline]:
            minimum = dataset_performaces[id_dataset]['min']
            maximum = dataset_performaces[id_dataset]['max']
            if id_dataset_pipeline == id_dataset:
                score = pipeline_data['score']
            else:
                score = pipeline_performances[id_pipeline][id_dataset]
            dtm += (maximum - score) / (maximum - minimum)
        adtm = dtm / len(pipeline_performances[id_pipeline])
        pipeline_data['adtm'] = adtm

    return pipelines


def merge_patterns(grammar_patterns):
    patterns = sorted(grammar_patterns, key=lambda x: len(x), reverse=True)
    empty_elements = set()
    skip_patterns = []

    for pattern in patterns:
        for element in pattern:
            modified_pattern = [e for e in pattern if e != element]
            for current_pattern in patterns:
                if modified_pattern == current_pattern:
                    empty_elements.add(element)
                    skip_patterns.append(modified_pattern)

    for skip_pattern in skip_patterns:
        if skip_pattern in patterns:
            patterns.remove(skip_pattern)

    return patterns, empty_elements


def combine_type_encoders(primitive_types):
    encoders = ['DATETIME_ENCODER', 'CATEGORICAL_ENCODER', 'TEXT_FEATURIZER']
    encoder_group = 'ENCODERS'
    new_primitive_types = []

    for primitive_type in primitive_types:
        if primitive_type in encoders:
            if encoder_group not in new_primitive_types:
                new_primitive_types.append(encoder_group)
        else:
            new_primitive_types.append(primitive_type)

    return new_primitive_types


def analyze_distribution(pipelines_metalearningdb):
    available_primitives = load_primitives_by_name()
    primitive_frequency = {}
    primitive_distribution = {}
    logger.info('Analyzing the distribution of primitives')

    for pipeline_data in pipelines_metalearningdb:
        for primitive_name in pipeline_data['pipeline']:
            primitive_type = available_primitives[primitive_name]['type']
            if primitive_type not in primitive_frequency:
                primitive_frequency[primitive_type] = {'primitives': {}, 'total': 0}
            if primitive_name not in primitive_frequency[primitive_type]['primitives']:
                primitive_frequency[primitive_type]['primitives'][primitive_name] = 0
            primitive_frequency[primitive_type]['primitives'][primitive_name] += 1
            primitive_frequency[primitive_type]['total'] += 1

    for primitive_type, primitives_info in primitive_frequency.items():
        if primitive_type not in primitive_distribution:
            primitive_distribution[primitive_type] = OrderedDict()
        for primitive, frequency in sorted(primitives_info['primitives'].items(), key=lambda x: x[1], reverse=True):
            distribution = float(frequency) / primitives_info['total']
            primitive_distribution[primitive_type][primitive] = distribution
        print(primitive_type)
        print(['%s %s' % (k, round(v, 4)) for k, v in primitive_distribution[primitive_type].items()])

    return primitive_distribution


def is_available_primitive(pipeline_primitives, current_primitives, verbose=False):
    for primitive in pipeline_primitives:
        if primitive['primitive']['id'] not in current_primitives:
            if verbose:
                logger.warning('Primitive %s is not longer available' % primitive['primitive']['python_path'])
            return False
    return True


def is_target_task(problem, task_keywords):
    problem_task_keywords = []
    if 'task_type' in problem:  # For old versions of the problem's schema
        problem_task_keywords = [problem['task_type']]
    elif 'task_keywords' in problem:
        problem_task_keywords = problem['task_keywords']

    # Skip semisupervised pipelines when it's not in the target task
    if 'SEMISUPERVISED' in problem_task_keywords and 'SEMISUPERVISED' not in task_keywords:
        return False

    if all(t in problem_task_keywords for t in task_keywords):
        return True

    return False


def filter_primitives(pipeline_steps, ignore_primitives):
    primitives = []

    for pipeline_step in pipeline_steps:
        if pipeline_step['primitive']['id'] not in ignore_primitives:
                primitives.append(pipeline_step['primitive']['id'])

    return primitives


def load_primitives_by_name():
    primitives_by_name = {}
    primitives = load_primitives_list()

    for primitive in primitives:
        primitives_by_name[primitive['python_path']] = {'id': primitive['id'], 'type': primitive['type']}

    return primitives_by_name


def load_primitives_by_id():
    primitives_by_id = {}
    primitives = load_primitives_list()

    for primitive in primitives:
        primitives_by_id[primitive['id']] = primitive['python_path']

    return primitives_by_id


if __name__ == '__main__':
    task_name = 'CLASSIFICATION_TASK'
    task_keywords = ['CLASSIFICATION', 'TABULAR', 'MULTICLASS']
    create_grammar_from_metalearningdb(task_name, task_keywords)
    #analyze_distribution(load_pipelines(task_keywords))
