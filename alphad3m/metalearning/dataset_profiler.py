import logging
import pandas as pd
import datamart_profiler
from metalearn import Metafeatures

logger = logging.getLogger(__name__)

DEFAULT_METAFEATURES = ['NumberOfInstances', 'NumberOfFeatures', 'NumberOfNumericFeatures', 'NumberOfCategoricalFeatures',
                        'RatioOfNumericFeatures', 'RatioOfCategoricalFeatures', 'NumberOfClasses', 'MeanClassProbability',
                        'StdevClassProbability', 'SkewClassProbability', 'KurtosisClassProbability', 'MinClassProbability',
                        'Quartile1ClassProbability', 'Quartile2ClassProbability', 'Quartile3ClassProbability', 'MaxClassProbability',
                        'MinorityClassSize', 'MajorityClassSize', 'Dimensionality', 'NumberOfMissingValues', 'RatioOfMissingValues',
                        'NumberOfInstancesWithMissingValues', 'RatioOfInstancesWithMissingValues', 'NumberOfFeaturesWithMissingValues',
                        'RatioOfFeaturesWithMissingValues', 'MeanCardinalityOfCategoricalFeatures', 'StdevCardinalityOfCategoricalFeatures',
                        'SkewCardinalityOfCategoricalFeatures', 'KurtosisCardinalityOfCategoricalFeatures', 'MinCardinalityOfCategoricalFeatures',
                        'Quartile1CardinalityOfCategoricalFeatures', 'Quartile2CardinalityOfCategoricalFeatures',
                        'Quartile3CardinalityOfCategoricalFeatures', 'MaxCardinalityOfCategoricalFeatures', 'MeanCardinalityOfNumericFeatures',
                        'StdevCardinalityOfNumericFeatures', 'SkewCardinalityOfNumericFeatures', 'KurtosisCardinalityOfNumericFeatures',
                        'MinCardinalityOfNumericFeatures', 'Quartile1CardinalityOfNumericFeatures', 'Quartile2CardinalityOfNumericFeatures',
                        'Quartile3CardinalityOfNumericFeatures', 'MaxCardinalityOfNumericFeatures']


def extract_metafeatures(dataset_path, target_column):
    data = pd.read_csv(dataset_path)
    Y = data[target_column]
    Y = pd.Series([str(i) for i in Y], name=target_column)  # Cast to string to get metalearn lib working correctly
    X = data.drop(columns=[target_column])
    metafeatures = Metafeatures()
    mfs = metafeatures.compute(X, Y, metafeature_ids=DEFAULT_METAFEATURES, seed=0, timeout=300)

    return mfs


def extract_dataprofiles(dataset_path, target_column, ignore_target_column=False):
    metadata = datamart_profiler.process_dataset(dataset_path, coverage=False)
    feature_types = set()
    missing_values = False
    for item in metadata['columns']:
        if ignore_target_column and item['name'] == target_column:
            continue
        identified_types = item['semantic_types'] if len(item['semantic_types']) > 0 else [item['structural_type']]
        for feature_type in identified_types:
            feature_types.add(feature_type)

        if 'missing_values_ratio' in item and item['name'] != target_column:
            missing_values = True

    dps = {'feature_types': sorted(feature_types), 'missing_values': missing_values}

    return dps
