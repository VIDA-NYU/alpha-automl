import logging
import numpy as np
from scipy import stats
from collections import OrderedDict
from alphad3m.metalearning.resource_builder import load_metalearningdb
from alphad3m.metalearning.dataset_similarity import get_similar_datasets
from alphad3m.primitive_loader import load_primitives_by_name, load_primitives_by_id

logger = logging.getLogger(__name__)


def load_related_pipelines(dataset_path, target_column, task_keywords):
    available_primitives = load_primitives_by_id()
    all_pipelines = load_metalearningdb()
    similar_datasets = get_similar_datasets('dataprofiles', dataset_path, target_column, task_keywords)
    task_pipelines = []

    for similar_dataset in similar_datasets.keys():
        if similar_dataset not in all_pipelines['pipeline_performances']:
            continue

        for pipeline_id, pipeline_performances in all_pipelines['pipeline_performances'][similar_dataset].items():
            primitive_ids = all_pipelines['pipeline_structure'][pipeline_id]
            if is_available_primitive(primitive_ids, available_primitives):
                for index in range(len(pipeline_performances['score'])):
                    primitives = [available_primitives[p] for p in primitive_ids]  # Use the current names of primitives
                    score = pipeline_performances['score'][index]
                    metric = pipeline_performances['metric'][index]
                    task_pipelines.append({'pipeline': primitives, 'score': score, 'metric': metric, 'dataset': similar_dataset,
                                           'pipeline_repr': '_'.join(primitives)})

    logger.info('Found %d related pipelines', len(task_pipelines))

    return task_pipelines


def create_metalearningdb_grammar(task_name, dataset_path, target_column, task_keywords, merge=False):
    pipelines = load_related_pipelines(dataset_path, target_column, task_keywords)
    patterns, primitives = extract_patterns(pipelines)
    empty_elements = []

    if merge:
        patterns, empty_elements = merge_patterns(patterns)

    grammar = format_grammar(task_name, patterns, empty_elements)

    return grammar, primitives


def format_grammar(task_name, patterns, empty_elements):
    if len(patterns) == 0:
        logger.warning('Empty patterns, no grammar have been generated')
        return None

    grammar = 'S -> %s\n' % task_name
    grammar += task_name + ' -> ' + ' | '.join([' '.join(p) for p in patterns])

    for element in sorted(set([e for sublist in patterns for e in sublist])):  # Sort to have a deterministic grammar
        production_rule = element + " -> 'primitive_terminal'"
        if element in empty_elements:
            production_rule += " | 'E'"

        grammar += '\n' + production_rule
    logger.info('Grammar obtained:\n%s', grammar)

    return grammar


def extract_patterns(pipelines, max_nro_patterns=15, min_frequency=3, adtm_threshold=0.5, mean_score_threshold=0.5, ratio_datasets=0.2):
    available_primitives = load_primitives_by_name()
    pipelines = calculate_adtm(pipelines)
    patterns = {}

    for pipeline_data in pipelines:
        if pipeline_data['adtm'] > adtm_threshold:
            # Skip pipelines with average distance to the minimum higher than the threshold
            continue

        primitive_types = [available_primitives[p]['type'] for p in pipeline_data['pipeline']]
        pattern_id = ' '.join(primitive_types)
        if pattern_id not in patterns:
            patterns[pattern_id] = {'structure': primitive_types, 'primitives': set(), 'datasets': set(), 'pipelines': [], 'scores': [], 'adtms': [], 'frequency': 0}
        patterns[pattern_id]['primitives'].update(pipeline_data['pipeline'])
        patterns[pattern_id]['datasets'].add(pipeline_data['dataset'])
        patterns[pattern_id]['pipelines'].append(pipeline_data['pipeline'])
        patterns[pattern_id]['scores'].append(pipeline_data['score'])
        patterns[pattern_id]['adtms'].append(pipeline_data['adtm'])
        patterns[pattern_id]['frequency'] += 1

    logger.info('Found %d different patterns, after creating the portfolio', len(patterns))
    # TODO: Group these removing conditions into a single loop
    # Remove patterns with fewer elements than the minimum frequency
    patterns = {k: v for k, v in patterns.items() if v['frequency'] >= min_frequency}
    logger.info('Found %d different patterns, after removing uncommon patterns', len(patterns))

    # Remove patterns with undesirable primitives (AlphaD3M doesn't have support to handle some of these primitives)
    blacklist_primitives = {'d3m.primitives.data_transformation.dataframe_to_ndarray.Common',
                            'd3m.primitives.data_transformation.list_to_dataframe.DistilListEncoder',
                            'd3m.primitives.data_transformation.ndarray_to_dataframe.Common',
                            'd3m.primitives.data_transformation.horizontal_concat.DSBOX',
                            'd3m.primitives.data_transformation.horizontal_concat.DataFrameCommon',
                            'd3m.primitives.data_transformation.multi_horizontal_concat.Common',
                            'd3m.primitives.data_transformation.conditioner.Conditioner',
                            'd3m.primitives.data_transformation.remove_semantic_types.Common',
                            'd3m.primitives.data_transformation.replace_semantic_types.Common',
                            'd3m.primitives.data_transformation.remove_columns.Common',
                            'd3m.primitives.operator.dataset_map.DataFrameCommon',
                            'd3m.primitives.data_transformation.i_vector_extractor.IVectorExtractor'}
    patterns = {k: v for k, v in patterns.items() if not blacklist_primitives & v['primitives']}
    logger.info('Found %d different patterns, after blacklisting primitives', len(patterns))

    unique_datasets = set()
    for pattern_id in patterns:
        scores = patterns[pattern_id]['scores']
        adtms = patterns[pattern_id]['adtms']
        patterns[pattern_id]['mean_score'] = np.mean(scores)
        patterns[pattern_id]['mean_adtm'] = np.mean(adtms)
        unique_datasets.update(patterns[pattern_id]['datasets'])
    # Remove patterns with low performances
    patterns = {k: v for k, v in patterns.items() if v['mean_score'] >= mean_score_threshold}
    logger.info('Found %d different patterns, after removing low-performance patterns', len(patterns))

    # Remove patterns with low variability
    patterns = {k: v for k, v in patterns.items() if len(v['datasets']) >= len(unique_datasets) * ratio_datasets}
    logger.info('Found %d different patterns, after removing low-variability patterns', len(patterns))

    if len(patterns) > max_nro_patterns:
        logger.info('Found many patterns, selecting top %d (max_nro_patterns)' % max_nro_patterns)
        sorted_patterns = sorted(patterns.items(), key=lambda x: x[1]['mean_score'], reverse=True)
        patterns = {k: v for k, v in sorted_patterns[:max_nro_patterns]}

    primitive_info = add_correlations(patterns, available_primitives)
    # Make deterministic the order of the patterns
    patterns = sorted(patterns.values(), key=lambda x: x['mean_score'], reverse=True)
    logger.info('Patterns:\n%s', patterns_repr(patterns))
    logger.info('Hierarchy:\n%s', '\n'.join(['%s:\n%s' % (k, ', '.join(v)) for k, v in primitive_info['hierarchy'].items()]))
    patterns = [p['structure'] for p in patterns]

    return patterns, primitive_info


def add_correlations(patterns, available_primitives):
    primitive_hierarchy = {}
    all_pipelines = []
    all_performances = []
    all_primitives = []

    # Add local correlations
    local_probabilities = {}
    for pattern_id, pattern in patterns.items():
        for primitive in pattern['primitives']:
            primitive_type = available_primitives[primitive]['type']
            if primitive_type not in primitive_hierarchy:
                primitive_hierarchy[primitive_type] = set()
            primitive_hierarchy[primitive_type].add(primitive)
        performances = [1 - x for x in pattern['adtms']]  # Use adtms as performances because their are scaled
        all_pipelines += pattern['pipelines']
        all_primitives += pattern['primitives']
        all_performances += performances
        correlations = calculate_correlations(pattern['primitives'], pattern['pipelines'], performances)
        local_probabilities[pattern_id] = {}
        for primitive, correlation in correlations.items():
            primitive_type = available_primitives[primitive]['type']
            if primitive_type not in local_probabilities[pattern_id]:
                local_probabilities[pattern_id][primitive_type] = {}
            local_probabilities[pattern_id][primitive_type][primitive] = correlation

    # Add global correlations
    global_probabilities = {}
    correlations = calculate_correlations(set(all_primitives), all_pipelines, all_performances)

    for primitive, correlation in correlations.items():
        primitive_type = available_primitives[primitive]['type']
        if primitive_type not in global_probabilities:
            global_probabilities[primitive_type] = {}
        global_probabilities[primitive_type][primitive] = correlation

    global_probabilities['S'] = {}
    for pattern, pattern_data in patterns.items():  # Use the mean adtm values as probabilities for the patterns
        global_probabilities['S'][pattern] = 1 - pattern_data['mean_adtm']

    primitive_probabilities = {'global': global_probabilities, 'local': local_probabilities, 'types': available_primitives}
    # Make deterministic the order of the hierarchy
    primitive_hierarchy = OrderedDict({k: sorted(v) for k, v in sorted(primitive_hierarchy.items(), key=lambda x: x[0])})
    primitive_info = {'hierarchy': primitive_hierarchy, 'probabilities': primitive_probabilities}

    return primitive_info


def calculate_correlations(primitives, pipelines, scores, normalize=True):
    correlations = {}

    for primitive in primitives:
        occurrences = [1 if primitive in pipeline else 0 for pipeline in pipelines]
        correlation_coefficient, p_value = stats.pointbiserialr(occurrences, scores)
        if np.isnan(correlation_coefficient):  # Assign a positive correlation (1) to NaN values
            correlation_coefficient = 1
        if normalize:  # Normalize the Pearson values, from [-1, 1] to [0, 1] range
            correlation_coefficient = (correlation_coefficient - (-1)) / 2  # xi − min(x) / max(x) − min(x)
        correlations[primitive] = round(correlation_coefficient, 4)

    return correlations


def calculate_adtm(pipelines):
    dataset_performaces = {}
    pipeline_performances = {}

    for pipeline_data in pipelines:
        # Even the same dataset can be run under different metrics. So, use the metric to create the id of the dataset
        id_dataset = pipeline_data['dataset'] + '_' + pipeline_data['metric']

        if id_dataset not in dataset_performaces:
            dataset_performaces[id_dataset] = {'min': float('inf'), 'max': float('-inf')}
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

        for id_dataset in pipeline_performances[id_pipeline]:  # Iterate over the datasets where the pipeline was used
            minimum = dataset_performaces[id_dataset]['min']
            maximum = dataset_performaces[id_dataset]['max']

            if id_dataset_pipeline == id_dataset:
                score = pipeline_data['score']
            else:
                score = pipeline_performances[id_pipeline][id_dataset]

            if minimum != maximum:
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


def is_available_primitive(pipeline_primitives, available_primitives, verbose=False):
    for primitive in pipeline_primitives:
        if primitive not in available_primitives:
            if verbose:
                logger.warning('Primitive %s is not longer available' % primitive)
            return False
    return True


def patterns_repr(patterns):
    patterns_string = []

    for pattern in patterns:
        pretty_string = ''
        pretty_string += 'structure: [%s]' % ', '.join([i for i in pattern['structure']])
        pretty_string += ', frequency: %d' % pattern['frequency']
        if 'mean_score' in pattern:
            pretty_string += ', mean_score: %.3f' % pattern['mean_score']
        if 'mean_adtm' in pattern:
            pretty_string += ', mean_adtm: %.3f' % pattern['mean_adtm']
        patterns_string.append(pretty_string)

    return '\n'.join(patterns_string)


def test_dataset(dataset_folder_path, task_name='TASK'):
    from os.path import join
    import json
    dataset_path = join(dataset_folder_path, 'TRAIN/dataset_TRAIN/tables/learningData.csv')
    problem_path = join(dataset_folder_path, 'TRAIN/problem_TRAIN/problemDoc.json')

    with open(problem_path) as fin:
        problem_doc = json.load(fin)
        task_keywords = problem_doc['about']['taskKeywords']
        target_column = problem_doc['inputs']['data'][0]['targets'][0]['colName']
    logger.info('Evaluating dataset "%s" with task keywords=%s' % (dataset_folder_path, str(task_keywords)))
    create_metalearningdb_grammar(task_name, dataset_path, target_column, task_keywords)


if __name__ == '__main__':
    dataset_id = '185_baseball_MIN_METADATA'
    dataset_folder_path = '/Users/rlopez/D3M/datasets/seed_datasets_current/%s' % dataset_id
    test_dataset(dataset_folder_path)
