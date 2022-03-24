import logging
import json
from os.path import join, dirname, isfile
from d3m import index
from collections import OrderedDict

logger = logging.getLogger(__name__)

PRIMITIVES_LIST_PATH = join(dirname(__file__), 'resource/primitives_list.json')
PRIMITIVES_HIERARCHY_PATH = join(dirname(__file__), 'resource/primitives_hierarchy.json')
INSTALLED_PRIMITIVES = sorted(index.search(), key=lambda x: x.endswith('SKlearn'), reverse=True)

BLACK_LIST = {
     # Not working primitives:
    'd3m.primitives.classification.random_classifier.Test',
    'd3m.primitives.classification.global_causal_discovery.ClassifierRPI',
    'd3m.primitives.classification.tree_augmented_naive_bayes.BayesianInfRPI',
    'd3m.primitives.classification.simple_cnaps.UBC',
    'd3m.primitives.classification.logistic_regression.UBC',
    'd3m.primitives.classification.multilayer_perceptron.UBC',
    'd3m.primitives.classification.canonical_correlation_forests.UBC',
    'd3m.primitives.regression.multilayer_perceptron.UBC',
    'd3m.primitives.regression.canonical_correlation_forests.UBC',
    'd3m.primitives.regression.linear_regression.UBC',
    'd3m.primitives.classification.inceptionV3_image_feature.Gator',
    'd3m.primitives.classification.search.Find_projections',
    'd3m.primitives.classification.search_hybrid.Find_projections',
    'd3m.primitives.regression.search_hybrid_numeric.Find_projections',
    'd3m.primitives.regression.search_numeric.Find_projections',
    'd3m.primitives.data_cleaning.binarizer.SKlearn',
    'd3m.primitives.feature_selection.rffeatures.Rffeatures',
    'd3m.primitives.feature_selection.mutual_info_classif.DistilMIRanking',
    'd3m.primitives.dimensionality_reduction.t_distributed_stochastic_neighbor_embedding.Tsne',
    'd3m.primitives.data_cleaning.string_imputer.SKlearn',
    'd3m.primitives.data_cleaning.tabular_extractor.Common',
    'd3m.primitives.data_cleaning.missing_indicator.SKlearn',
    'd3m.primitives.data_transformation.gaussian_random_projection.SKlearn',
    'd3m.primitives.data_transformation.sparse_random_projection.SKlearn',
    'd3m.primitives.feature_extraction.nk_sent2vec.Sent2Vec',
    'd3m.primitives.classification.mlp.BBNMLPClassifier',
    # Repeated primitives:
    'd3m.primitives.data_transformation.unary_encoder.DSBOX',
    'd3m.primitives.data_transformation.one_hot_encoder.TPOT',
    'd3m.primitives.data_transformation.one_hot_encoder.MakerCommon',
    'd3m.primitives.data_transformation.one_hot_encoder.PandasCommon',
    'd3m.primitives.feature_extraction.tfidf_vectorizer.BBNTfidfTransformer',
    'd3m.primitives.data_transformation.one_hot_encoder.DistilOneHotEncoder',
    'd3m.primitives.feature_selection.pca_features.Pcafeatures',
    'd3m.primitives.classification.random_forest.Common',
    # Poor performance:
    'd3m.primitives.classification.cover_tree.Fastlvm',
    'd3m.primitives.classification.linear_svc.DistilRankedLinearSVC',
    'd3m.primitives.classification.lstm.DSBOX',
    'd3m.primitives.regression.cover_tree.Fastlvm',
    'd3m.primitives.classification.bert_classifier.DistilBertPairClassification',
    'd3m.primitives.regression.global_causal_discovery.RegressorRPI',
    'd3m.primitives.regression.monomial.Test',
    'd3m.primitives.regression.rfm_precondition_ed_gaussian_krr.RFMPreconditionedGaussianKRR',
    'd3m.primitives.regression.rfm_precondition_ed_polynomial_krr.RFMPreconditionedPolynomialKRR',
    'd3m.primitives.regression.tensor_machines_regularized_least_squares.TensorMachinesRegularizedLeastSquares',
    'd3m.primitives.clustering.cluster_curve_fitting_kmeans.ClusterCurveFittingKMeans',
    'd3m.primitives.clustering.kmeans_clustering.UBC',
    'd3m.primitives.clustering.spectral_graph.SpectralClustering',
    'd3m.primitives.data_cleaning.greedy_imputation.DSBOX',
    'd3m.primitives.data_cleaning.iterative_regression_imputation.DSBOX',
    'd3m.primitives.data_cleaning.mean_imputation.DSBOX',
    'd3m.primitives.data_preprocessing.random_sampling_imputer.BYU',
    'd3m.primitives.data_transformation.imputer.DistilCategoricalImputer',
    'd3m.primitives.feature_extraction.feature_agglomeration.SKlearn',
    'd3m.primitives.feature_extraction.boc.UBC',
    'd3m.primitives.feature_extraction.bow.UBC',
    'd3m.primitives.natural_language_processing.glda.Fastlvm',
    'd3m.primitives.natural_language_processing.hdp.Fastlvm',
    'd3m.primitives.natural_language_processing.lda.Fastlvm',
    'd3m.primitives.classification.dummy.SKlearn',
    'd3m.primitives.regression.dummy.SKlearn',
    'd3m.primitives.data_cleaning.normalizer.SKlearn',
    'd3m.primitives.classification.ensemble_voting.DSBOX',
    # Long running times:
    'd3m.primitives.feature_selection.joint_mutual_information.AutoRPI',
    'd3m.primitives.feature_selection.score_based_markov_blanket.RPI',
    'd3m.primitives.feature_selection.simultaneous_markov_blanket.AutoRPI'
}


def get_primitive_class(primitive_name):
    return index.get_primitive(primitive_name)


def get_primitive_family(primitive_name):
    return get_primitive_class(primitive_name).metadata.to_json_structure()['primitive_family']


def get_primitive_algorithms(primitive_name):
    return get_primitive_class(primitive_name).metadata.to_json_structure()['algorithm_types']


def get_primitive_info(primitive_name):
    primitive_dict = get_primitive_class(primitive_name).metadata.to_json_structure()

    return {
            'id': primitive_dict['id'],
            'name': primitive_dict['name'],
            'version': primitive_dict['version'],
            'python_path': primitive_dict['python_path'],
            'digest': primitive_dict['digest'],
            'type': get_primitive_type(primitive_name)
    }


def get_primitive_type(primitive_name):
    primitive_type = get_primitive_family(primitive_name)
    #  Use the algorithm types as families because they are more descriptive
    if primitive_type in {'DATA_TRANSFORMATION', 'DATA_PREPROCESSING', 'DATA_CLEANING'}:
        algorithm_types = get_primitive_algorithms(primitive_name)
        primitive_type = algorithm_types[0]

    # Changing the primitive families using some predefined rules
    if primitive_name in {'d3m.primitives.data_cleaning.quantile_transformer.SKlearn',
                          'd3m.primitives.data_cleaning.normalizer.SKlearn',
                          'd3m.primitives.normalization.iqr_scaler.DSBOX'}:
        primitive_type = 'FEATURE_SCALING'

    elif primitive_name in {'d3m.primitives.feature_extraction.feature_agglomeration.SKlearn',
                            'd3m.primitives.feature_selection.mutual_info_classif.DistilMIRanking'}:
        primitive_type = 'FEATURE_SELECTION'

    elif primitive_name in {'d3m.primitives.feature_extraction.pca.SKlearn',
                            'd3m.primitives.feature_selection.pca_features.Pcafeatures',
                            'd3m.primitives.feature_extraction.truncated_svd.SKlearn',
                            'd3m.primitives.feature_extraction.pca_features.RandomizedPolyPCA',
                            'd3m.primitives.data_transformation.gaussian_random_projection.SKlearn',
                            'd3m.primitives.data_transformation.sparse_random_projection.SKlearn',
                            'd3m.primitives.data_transformation.fast_ica.SKlearn'}:
        primitive_type = 'DIMENSIONALITY_REDUCTION'  # Or should it be FEATURE_SELECTION ?

    elif primitive_name in {'d3m.primitives.classification.bert_classifier.DistilBertTextClassification',
                            'd3m.primitives.classification.text_classifier.DistilTextClassifier'}:
        primitive_type = 'TEXT_CLASSIFIER'

    elif primitive_name in {'d3m.primitives.data_transformation.enrich_dates.DistilEnrichDates'}:
        primitive_type = 'DATETIME_ENCODER'

    elif primitive_name in {'d3m.primitives.vertex_nomination.seeded_graph_matching.DistilVertexNomination',
                            'd3m.primitives.classification.gaussian_classification.JHU'}:
        primitive_type = 'VERTEX_CLASSIFICATION'

    elif primitive_name in {'d3m.primitives.graph_clustering.gaussian_clustering.JHU'}:
        primitive_type = 'COMMUNITY_DETECTION'

    elif primitive_name in {'d3m.primitives.data_transformation.load_single_graph.DistilSingleGraphLoader',
                            'd3m.primitives.data_transformation.load_graphs.DistilGraphLoader',
                            'd3m.primitives.data_transformation.load_graphs.JHU',
                            'd3m.primitives.data_transformation.load_edgelist.DistilEdgeListLoader'}:
        primitive_type = 'GRAPH_LOADER'

    elif primitive_name in {'d3m.primitives.feature_extraction.yolo.DSBOX'}:
        primitive_type = 'OBJECT_DETECTION'

    elif primitive_name in {'d3m.primitives.feature_construction.corex_text.DSBOX',
                            'd3m.primitives.data_transformation.encoder.DistilTextEncoder',
                            'd3m.primitives.feature_extraction.tfidf_vectorizer.SKlearn',
                            'd3m.primitives.feature_extraction.count_vectorizer.SKlearn',
                            'd3m.primitives.feature_extraction.boc.UBC',
                            'd3m.primitives.feature_extraction.bow.UBC',
                            'd3m.primitives.feature_extraction.nk_sent2vec.Sent2Vec',
                            'd3m.primitives.feature_extraction.tfidf_vectorizer.BBNTfidfTransformer'}:
        primitive_type = 'TEXT_FEATURIZER'

    elif primitive_name in {'d3m.primitives.feature_extraction.image_transfer.DistilImageTransfer',
                            'd3m.primitives.feature_extraction.resnet50_image_feature.DSBOX'}:
        primitive_type = 'IMAGE_FEATURIZER'

    elif primitive_name in {'d3m.primitives.feature_extraction.audio_transfer.DistilAudioTransfer'}:
        primitive_type = 'AUDIO_FEATURIZER'

    elif primitive_name in {'d3m.primitives.data_transformation.audio_reader.DistilAudioDatasetLoader'}:
        primitive_type = 'AUDIO_READER'

    elif primitive_name in {'d3m.primitives.feature_extraction.resnext101_kinetics_video_features.VideoFeaturizer'}:
        primitive_type = 'VIDEO_FEATURIZER'

    elif primitive_name in {'d3m.primitives.feature_extraction.random_projection_timeseries_featurization.DSBOX'}:
        primitive_type = 'TIME_SERIES_FEATURIZER'

    elif primitive_name in {'d3m.primitives.data_transformation.time_series_to_list.DSBOX'}:
        primitive_type = 'TIME_SERIES_READER'

    elif primitive_name in {'d3m.primitives.data_transformation.grouping_field_compose.Common'}:
        primitive_type = 'TIME_SERIES_GROUPER'

    elif primitive_name in {'d3m.primitives.data_transformation.dataframe_to_tensor.DSBOX'}:
        primitive_type = 'TO_TENSOR'

    if primitive_type == 'ENCODE_ONE_HOT':
        primitive_type = 'CATEGORICAL_ENCODER'

    return primitive_type


def load_primitives_hierarchy():
    if isfile(PRIMITIVES_HIERARCHY_PATH):
        with open(PRIMITIVES_HIERARCHY_PATH) as fin:
            primitives = json.load(fin)
        # Verify if the loaded primitives from file are installed
        installed_primitives = {}
        for primitive_type in primitives.keys():
            installed_primitives[primitive_type] = [x for x in primitives[primitive_type] if x in INSTALLED_PRIMITIVES]

        logger.info('Loading primitives info from file')
        return installed_primitives

    primitives = {}
    for primitive_name in INSTALLED_PRIMITIVES:
        if primitive_name not in BLACK_LIST:
            try:
                primitive_type = get_primitive_type(primitive_name)
            except Exception as e:
                logger.error('Loading metadata about primitive %s', primitive_name, exc_info=e)
                continue

            if primitive_type not in primitives:
                primitives[primitive_type] = []
            primitives[primitive_type].append(primitive_name)

    with open(PRIMITIVES_HIERARCHY_PATH, 'w') as fout:
        json.dump(OrderedDict(sorted(primitives.items())), fout, indent=4)
    logger.info('Loading primitives info from D3M index')

    return primitives


def load_primitives_list():
    if isfile(PRIMITIVES_LIST_PATH):
        with open(PRIMITIVES_LIST_PATH) as fin:
            primitives = json.load(fin)
        # Verify if the loaded primitives from file are installed
        installed_primitives = []
        for primitive_data in primitives:
            if primitive_data['python_path'] in INSTALLED_PRIMITIVES:
                installed_primitives.append(primitive_data)

        logger.info('Loading primitives info from file')
        return installed_primitives

    primitives = []
    for primitive_name in INSTALLED_PRIMITIVES:
            try:
                primitive_info = get_primitive_info(primitive_name)
            except Exception as e:
                logger.error('Loading metadata about primitive %s', primitive_name, exc_info=e)
                continue
            primitives.append(primitive_info)

    with open(PRIMITIVES_LIST_PATH, 'w') as fout:
        json.dump(primitives, fout, indent=4)
    logger.info('Loading primitives info from D3M index')

    return primitives
