import os
import json
import copy
import logging
from os.path import join
import statistics
from collections import OrderedDict

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


PRIMITIVES_BY_NAME_PATH = os.path.join(os.path.dirname(__file__), '../../resource/primitives_by_name.json')
PRIMITIVES_BY_TYPE_PATH = os.path.join(os.path.dirname(__file__), '../../resource/primitives_by_type.json')
METALEARNINGDB_PATH = os.path.join(os.path.dirname(__file__), '../../resource/metalearningdb.json')

IGNORE_PRIMITIVES = {'d3m.primitives.data_transformation.construct_predictions.Common',
                     'd3m.primitives.data_transformation.extract_columns_by_semantic_types.Common',
                     'd3m.primitives.data_transformation.dataset_to_dataframe.Common',
                     'd3m.primitives.data_transformation.denormalize.Common',
                     'd3m.primitives.data_transformation.flatten.DataFrameCommon',
                     'd3m.primitives.data_transformation.column_parser.Common',
                     'd3m.primitives.schema_discovery.profiler.DSBOX',
                     'd3m.primitives.data_cleaning.column_type_profiler.Simon',
                     'd3m.primitives.data_transformation.text_reader.Common',
                     'd3m.primitives.data_transformation.image_reader.Common',
                     'd3m.primitives.data_transformation.audio_reader.Common',
                     'd3m.primitives.data_transformation.dataframe_to_tensor.DSBOX'
                     }


def merge_pipeline_files(pipelines_file, pipeline_runs_file, problems_file, n=-1, verbose=False):
    logger.info('Adding pipelines to lookup table...')
    pipelines = {}
    with open(pipelines_file, 'r') as f:
        for line in f:
            pipeline = json.loads(line)
            pipelines[pipeline['digest']] = pipeline

    logger.info('Adding problems to lookup table...')
    problems = {}
    with open(problems_file, 'r') as f:
        for line in f:
            problem = json.loads(line)
            problems[problem['digest']] = problem['problem']
            problems[problem['digest']]['id'] = problem['id']

    logger.info('Merging pipeline information with pipeline_runs_file (this might take a while)...')
    merged = []
    with open(pipeline_runs_file, 'r') as f:
        for line in f:
            if len(merged) == n:
                break
            try:
                run = json.loads(line)
                if run['run']['phase'] != 'PRODUCE':
                    continue
                pipeline = pipelines[run['pipeline']['digest']]
                problem = problems[run['problem']['digest']]
                data = {
                    'pipeline_id': pipeline['id'],
                    'pipeline_digest': pipeline['digest'],
                    'pipeline_source': pipeline['source'],
                    'inputs': pipeline['inputs'],
                    'outputs': pipeline['outputs'],
                    'problem': problem,
                    'start': run['start'],
                    'end': run['end'],
                    'steps': pipeline['steps'],
                    'scores': run['run']['results']['scores']
                }
                merged.append(json.dumps(data))
            except Exception as e:
                if (verbose):
                    logger.error(problem['id'], repr(e))
    logger.info('Done.')

    with open(METALEARNINGDB_PATH, 'w') as fout:
        fout.write('\n'.join(merged))


def load_metalearningdb(task_keywords):
    primitives_by_name = load_primitives_by_name()
    primitive_ids = set(primitives_by_name.values())
    ignore_primitives_ids = set()
    all_pipelines = []
    task_pipelines = []

    logger.info('Loading pipelines from metalearning database...')

    with open(METALEARNINGDB_PATH) as fin:
        for line in fin:
            all_pipelines.append(json.loads(line))

    for ignore_primitive in IGNORE_PRIMITIVES:
        if ignore_primitive in primitives_by_name:
            ignore_primitives_ids.add(primitives_by_name[ignore_primitive])

    for pipeline_run in all_pipelines:
        pipeline_primitives = pipeline_run['steps']
        if is_target_task(pipeline_run['problem'], task_keywords) and is_available_primitive(pipeline_primitives, primitive_ids):
            primitives = filter_primitives(pipeline_primitives, ignore_primitives_ids)
            if len(primitives) > 0:
                score = pipeline_run['scores'][0]['value']
                task_pipelines.append((primitives, score))

    logger.info('Found %d pipelines for task %s', len(task_pipelines), '_'.join(task_keywords))

    return task_pipelines


def create_grammar_from_metalearningdb(task_keywords):
    pipelines = load_metalearningdb(task_keywords)
    combine_encoders = 'TEXT' not in task_keywords
    patterns = extract_patterns(pipelines, combine_encoders)
    patterns, empty_elements = merge_patterns(patterns)
    grammar = format_grammar(patterns, empty_elements)

    return grammar


def format_grammar(patterns, empty_elements):
    grammar = 'S -> ' + ' | '.join([' '.join(p) for p in patterns])

    for element in set([item for sublist in patterns for item in sublist]):
        production_rule = element + " -> 'primitive_terminal'"
        if element in empty_elements:
            production_rule += " | 'E'"

        grammar += '\n' + production_rule
    logger.info('Grammar obtained:\n%s', grammar)
    return grammar


def extract_patterns(pipelines, combine_encoders=True, min_frequency=5, min_avg_performance=0.5):
    primitives_by_type = load_primitives_by_type()
    patterns = {}

    for pipeline, score in pipelines:
        primitive_types = [primitives_by_type[p] for p in pipeline]
        if combine_encoders:
            primitive_types = combine_type_encoders(primitive_types)
        pattern_id = ' '.join(primitive_types)
        if pattern_id not in patterns:
            patterns[pattern_id] = {'structure': primitive_types, 'scores': [], 'frequency': 0}
        patterns[pattern_id]['scores'].append(score)
        patterns[pattern_id]['frequency'] += 1

    logger.info('Found %d different patterns', len(patterns))
    # Remove patterns with fewer elements than the minimum frequency
    patterns = {k: v for k, v in patterns.items() if v['frequency'] >= min_frequency}
    logger.info('Found %d different patterns, after removing uncommon patterns', len(patterns))

    for pattern_id in patterns:
        scores = patterns[pattern_id].pop('scores')
        patterns[pattern_id]['average'] = statistics.mean(scores)

    # Remove patterns with low performances
    blacklist_primitive_types = {'OPERATOR', 'ARRAY_CONCATENATION'}
    patterns = {k: v for k, v in patterns.items() if not blacklist_primitive_types & set(v['structure'])}
    logger.info('Found %d different patterns, after blacklisting primitive types', len(patterns))

    # Remove patterns with low performances
    patterns = {k: v for k, v in patterns.items() if v['average'] >= min_avg_performance}
    logger.info('Found %d different patterns, after removing low-performance patterns', len(patterns))
    patterns = sorted(patterns.values(), key=lambda x: x['average'], reverse=True)
    logger.info('Patterns:\n%s', '\n'.join([str(x) for x in patterns]))
    patterns = [p['structure'] for p in patterns]

    return patterns


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





    pass


def combine_type_encoders(primitive_types):
    encoders = ['TEXT_ENCODER', 'DATETIME_ENCODER', 'CATEGORICAL_ENCODER', 'TEXT_FEATURIZER']
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
    primitives_by_type = load_primitives_by_type()
    primitives_by_id = load_primitives_by_id()
    primitive_frequency = {}
    primitive_distribution = {}
    logger.info('Analyzing the distribution of primitives')

    for pipeline, score in pipelines_metalearningdb:
        for primitive_id in pipeline:
            primitive_type = primitives_by_type[primitive_id]
            if primitive_type not in primitive_frequency:
                primitive_frequency[primitive_type] = {'primitives': {}, 'total': 0}
            primitive_name = primitives_by_id[primitive_id]
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
    primitives = OrderedDict()

    for pipeline_step in pipeline_steps:
        if pipeline_step['primitive']['id'] not in ignore_primitives:
                primitives[pipeline_step['primitive']['id']] = pipeline_step['primitive']['python_path']

    return primitives


def load_primitives_by_name():
    primitives_by_name = {}
    available_primitives = set()

    with open(PRIMITIVES_BY_TYPE_PATH) as fin:
        for primitive_type, primitive_names in json.load(fin).items():
            for primitive_name in primitive_names:
                available_primitives.add(primitive_name)

    with open(PRIMITIVES_BY_NAME_PATH) as fin:
        primitives = json.load(fin)

    for primitive in primitives:
        if primitive['python_path'] in available_primitives:
            primitives_by_name[primitive['python_path']] = primitive['id']

    return primitives_by_name


def load_primitives_by_id():
    primitives_by_id = {}
    available_primitives = set()

    with open(PRIMITIVES_BY_TYPE_PATH) as fin:
        for primitive_type, primitive_names in json.load(fin).items():
            for primitive_name in primitive_names:
                available_primitives.add(primitive_name)

    with open(PRIMITIVES_BY_NAME_PATH) as fin:
        primitives = json.load(fin)

    for primitive in primitives:
        if primitive['python_path'] in available_primitives:
            primitives_by_id[primitive['id']] = primitive['python_path']

    return primitives_by_id


def load_primitives_by_type():
    primitives_by_type = {}
    primitives_by_name = load_primitives_by_name()

    with open(PRIMITIVES_BY_TYPE_PATH) as fin:
        primitives = json.load(fin)

    for primitive_type in primitives:
        primitive_names = primitives[primitive_type]
        for primitive_name in primitive_names:
            primitives_by_type[primitives_by_name[primitive_name]] = primitive_type

    return primitives_by_type


if __name__ == '__main__':
    task_keywords = ['CLASSIFICATION', 'TABULAR']
    #pipelines_file = '/Users/rlopez/Downloads/metalearningdb_dump_20200304/pipelines-1583354358.json'
    #pipeline_runs_file = '/Users/rlopez/Downloads/metalearningdb_dump_20200304/pipeline_runs-1583354387.json'
    #problems_file = '/Users/rlopez/Downloads/metalearningdb_dump_20200304/problems-1583354357.json'
    #merge_pipeline_files(pipelines_file, pipeline_runs_file, problems_file)
    create_grammar_from_metalearningdb(task_keywords)
    #analyze_distribution(load_metalearningdb(task_keywords))
    '''non_terminals = {x: i+1 for i, x in enumerate(set(load_primitives_by_type().values()))}
    terminals = {x: len(non_terminals) + i for i, x in enumerate(load_primitives_by_name().keys())}
    terminals['E'] = 0
    rules = {'S -> IMPUTATION ENCODERS FEATURE_SCALING FEATURE_SELECTION CLASSIFICATION': 1, 'ENCODERS -> CATEGORICAL_ENCODER TEXT_ENCODER': 2, 'IMPUTATION -> d3m.primitives.data_cleaning.imputer.SKlearn': 3, 'IMPUTATION -> d3m.primitives.data_cleaning.missing_indicator.SKlearn': 4, 'IMPUTATION -> d3m.primitives.data_cleaning.string_imputer.SKlearn': 5, 'IMPUTATION -> d3m.primitives.data_cleaning.tabular_extractor.Common': 6, 'IMPUTATION -> d3m.primitives.data_preprocessing.greedy_imputation.DSBOX': 7, 'IMPUTATION -> d3m.primitives.data_preprocessing.iterative_regression_imputation.DSBOX': 8, 'IMPUTATION -> d3m.primitives.data_preprocessing.mean_imputation.DSBOX': 9, 'IMPUTATION -> d3m.primitives.data_preprocessing.random_sampling_imputer.BYU': 10, 'IMPUTATION -> d3m.primitives.data_transformation.imputer.DistilCategoricalImputer': 11, 'IMPUTATION -> E': 12, 'FEATURE_SELECTION -> d3m.primitives.feature_selection.generic_univariate_select.SKlearn': 13, 'FEATURE_SELECTION -> d3m.primitives.feature_selection.select_fwe.SKlearn': 14, 'FEATURE_SELECTION -> d3m.primitives.feature_selection.select_percentile.SKlearn': 15, 'FEATURE_SELECTION -> d3m.primitives.feature_selection.variance_threshold.SKlearn': 16, 'FEATURE_SELECTION -> d3m.primitives.feature_selection.joint_mutual_information.AutoRPI': 17, 'FEATURE_SELECTION -> d3m.primitives.feature_selection.pca_features.Pcafeatures': 18, 'FEATURE_SELECTION -> d3m.primitives.feature_selection.rffeatures.Rffeatures': 19, 'FEATURE_SELECTION -> d3m.primitives.feature_selection.score_based_markov_blanket.RPI': 20, 'FEATURE_SELECTION -> d3m.primitives.feature_selection.simultaneous_markov_blanket.AutoRPI': 21, 'FEATURE_SELECTION -> d3m.primitives.feature_selection.skfeature.TAMU': 22, 'FEATURE_SELECTION -> E': 23, 'FEATURE_SCALING -> d3m.primitives.data_preprocessing.binarizer.SKlearn': 24, 'FEATURE_SCALING -> d3m.primitives.data_preprocessing.max_abs_scaler.SKlearn': 25, 'FEATURE_SCALING -> d3m.primitives.data_preprocessing.min_max_scaler.SKlearn': 26, 'FEATURE_SCALING -> d3m.primitives.data_preprocessing.robust_scaler.SKlearn': 27, 'FEATURE_SCALING -> d3m.primitives.data_preprocessing.standard_scaler.SKlearn': 28, 'FEATURE_SCALING -> E': 29, 'CLASSIFICATION -> d3m.primitives.classification.ada_boost.SKlearn': 30, 'CLASSIFICATION -> d3m.primitives.classification.bagging.SKlearn': 31, 'CLASSIFICATION -> d3m.primitives.classification.bernoulli_naive_bayes.SKlearn': 32, 'CLASSIFICATION -> d3m.primitives.classification.decision_tree.SKlearn': 33, 'CLASSIFICATION -> d3m.primitives.classification.dummy.SKlearn': 34, 'CLASSIFICATION -> d3m.primitives.classification.extra_trees.SKlearn': 35, 'CLASSIFICATION -> d3m.primitives.classification.gaussian_naive_bayes.SKlearn': 36, 'CLASSIFICATION -> d3m.primitives.classification.gradient_boosting.SKlearn': 37, 'CLASSIFICATION -> d3m.primitives.classification.k_neighbors.SKlearn': 38, 'CLASSIFICATION -> d3m.primitives.classification.linear_discriminant_analysis.SKlearn': 39, 'CLASSIFICATION -> d3m.primitives.classification.linear_svc.SKlearn': 40, 'CLASSIFICATION -> d3m.primitives.classification.logistic_regression.SKlearn': 41, 'CLASSIFICATION -> d3m.primitives.classification.mlp.SKlearn': 42, 'CLASSIFICATION -> d3m.primitives.classification.multinomial_naive_bayes.SKlearn': 43, 'CLASSIFICATION -> d3m.primitives.classification.nearest_centroid.SKlearn': 44, 'CLASSIFICATION -> d3m.primitives.classification.passive_aggressive.SKlearn': 45, 'CLASSIFICATION -> d3m.primitives.classification.quadratic_discriminant_analysis.SKlearn': 46, 'CLASSIFICATION -> d3m.primitives.classification.random_forest.SKlearn': 47, 'CLASSIFICATION -> d3m.primitives.classification.sgd.SKlearn': 48, 'CLASSIFICATION -> d3m.primitives.classification.svc.SKlearn': 49, 'CLASSIFICATION -> d3m.primitives.classification.bert_classifier.DistilBertPairClassification': 50, 'CLASSIFICATION -> d3m.primitives.classification.cover_tree.Fastlvm': 51, 'CLASSIFICATION -> d3m.primitives.classification.gaussian_classification.JHU': 52, 'CLASSIFICATION -> d3m.primitives.classification.light_gbm.Common': 53, 'CLASSIFICATION -> d3m.primitives.classification.logistic_regression.UBC': 54, 'CLASSIFICATION -> d3m.primitives.classification.lstm.DSBOX': 55, 'CLASSIFICATION -> d3m.primitives.classification.mlp.BBNMLPClassifier': 56, 'CLASSIFICATION -> d3m.primitives.classification.multilayer_perceptron.UBC': 57, 'CLASSIFICATION -> d3m.primitives.classification.random_classifier.Test': 58, 'CLASSIFICATION -> d3m.primitives.classification.random_forest.Common': 59, 'CLASSIFICATION -> d3m.primitives.classification.search.Find_projections': 60, 'CLASSIFICATION -> d3m.primitives.classification.search_hybrid.Find_projections': 61, 'CLASSIFICATION -> d3m.primitives.classification.simple_cnaps.UBC': 62, 'CLASSIFICATION -> d3m.primitives.classification.text_classifier.DistilTextClassifier': 63, 'CLASSIFICATION -> d3m.primitives.classification.xgboost_dart.Common': 64, 'CLASSIFICATION -> d3m.primitives.classification.xgboost_gbtree.Common': 65, 'CATEGORICAL_ENCODER -> d3m.primitives.data_transformation.one_hot_encoder.SKlearn': 66, 'CATEGORICAL_ENCODER -> d3m.primitives.data_preprocessing.encoder.DSBOX': 67, 'CATEGORICAL_ENCODER -> d3m.primitives.data_preprocessing.one_hot_encoder.MakerCommon': 68, 'CATEGORICAL_ENCODER -> d3m.primitives.data_preprocessing.one_hot_encoder.PandasCommon': 69, 'CATEGORICAL_ENCODER -> d3m.primitives.data_preprocessing.unary_encoder.DSBOX': 70, 'CATEGORICAL_ENCODER -> d3m.primitives.data_transformation.one_hot_encoder.DistilOneHotEncoder': 71, 'CATEGORICAL_ENCODER -> d3m.primitives.data_transformation.one_hot_encoder.TPOT': 72, 'TEXT_ENCODER -> d3m.primitives.data_preprocessing.count_vectorizer.SKlearn': 73, 'TEXT_ENCODER -> d3m.primitives.data_preprocessing.tfidf_vectorizer.SKlearn': 74, 'TEXT_ENCODER -> d3m.primitives.data_transformation.encoder.DistilTextEncoder': 75, 'TEXT_ENCODER -> d3m.primitives.feature_construction.corex_text.DSBOX': 76}
    grammar = {'RULES': rules, 'NON_TERMINALS': non_terminals, 'TERMINALS': terminals}
    create_vectors_from_metalearningdb(task, grammar)'''
