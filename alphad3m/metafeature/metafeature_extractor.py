import os
import logging
import pickle
import frozendict
from alphad3m.schema import database
from d3m.container import Dataset
from d3m.metadata import base as metadata_base
from alphad3m.pipeline_operations.pipeline_execute import execute
logger = logging.getLogger(__name__)

METAFEATURES_DEFAULT = [
                        'dimensionality',
                        'number_distinct_values_of_categorical_attributes.max',
                        'number_distinct_values_of_numeric_attributes.max',
                        'entropy_of_categorical_attributes.max',
                        'kurtosis_of_attributes.max',
                        'mean_of_attributes.max',
                        'entropy_of_numeric_attributes.max',
                        'skew_of_attributes.max',
                        'standard_deviation_of_attributes.max',
                        'number_distinct_values_of_categorical_attributes.mean',
                        'number_distinct_values_of_numeric_attributes.mean',
                        'entropy_of_categorical_attributes.mean',
                        'kurtosis_of_attributes.mean',
                        'mean_of_attributes.mean',
                        'entropy_of_numeric_attributes.mean',
                        'skew_of_attributes.mean',
                        'standard_deviation_of_attributes.mean',
                        'number_distinct_values_of_categorical_attributes.min',
                        'number_distinct_values_of_numeric_attributes.min',
                        'entropy_of_categorical_attributes.min',
                        'kurtosis_of_attributes.min',
                        'mean_of_attributes.min',
                        'entropy_of_numeric_attributes.min',
                        'skew_of_attributes.min',
                        'standard_deviation_of_attributes.min',
                        'number_of_categorical_attributes',
                        'number_of_attributes',
                        'number_of_features_with_missing_values',
                        'number_of_instances',
                        'number_of_instances_with_missing_values',
                        'number_of_missing_values',
                        'number_of_numeric_attributes',
                        'pca.eigenvalue_component_1',
                        'pca.eigenvalue_component_2',
                        'pca.eigenvalue_component_3',
                        'pca.explained_variance_ratio_component_1',
                        'pca.explained_variance_ratio_component_2',
                        'pca.explained_variance_ratio_component_3',
                        'entropy_of_categorical_attributes.quartile_1',
                        'kurtosis_of_attributes.quartile_1',
                        'mean_of_attributes.quartile_1',
                        'entropy_of_numeric_attributes.quartile_1',
                        'skew_of_attributes.quartile_1',
                        'standard_deviation_of_attributes.quartile_1',
                        'entropy_of_categorical_attributes.median',
                        'kurtosis_of_attributes.median',
                        'mean_of_attributes.median',
                        'entropy_of_numeric_attributes.median',
                        'skew_of_attributes.median',
                        'standard_deviation_of_attributes.median',
                        'entropy_of_categorical_attributes.quartile_3',
                        'kurtosis_of_attributes.quartile_3',
                        'mean_of_attributes.quartile_3',
                        'entropy_of_numeric_attributes.quartile_3',
                        'skew_of_attributes.quartile_3',
                        'standard_deviation_of_attributes.quartile_3',
                        'ratio_of_categorical_attributes',
                        'ratio_of_features_with_missing_values',
                        'ratio_of_instances_with_missing_values',
                        'ratio_of_missing_values',
                        'ratio_of_numeric_attributes',
                        'number_distinct_values_of_categorical_attributes.std',
                        'number_distinct_values_of_numeric_attributes.std',
                        'kurtosis_of_attributes.std',
                        'mean_of_attributes.std',
                        'skew_of_attributes.std',
                        'standard_deviation_of_attributes.std'
                        ]


class ComputeMetafeatures():

    def __init__(self, dataset, targets=None, features=None, DBSession=None):
        self.dataset = Dataset.load(dataset)
        self.dataset_uri = dataset
        self.DBSession = DBSession
        self.targets = targets
        self.features = features

    def _add_target_columns_metadata(self):
        for target in self.targets:
            resource_id = target[0]
            target_name = target[1]
            for column_index in range(self.dataset.metadata.query((resource_id, metadata_base.ALL_ELEMENTS))['dimension']['length']):
                if self.dataset.metadata.query((resource_id, metadata_base.ALL_ELEMENTS, column_index)).get('name',
                                                                                                None) == target_name:
                    semantic_types = list(self.dataset.metadata.query(
                        (resource_id, metadata_base.ALL_ELEMENTS, column_index)).get('semantic_types', []))

                    if 'https://metadata.datadrivendiscovery.org/types/Target' not in semantic_types:
                        semantic_types.append('https://metadata.datadrivendiscovery.org/types/Target')
                        self.dataset.metadata = self.dataset.metadata.update(
                            (resource_id, metadata_base.ALL_ELEMENTS, column_index),
                            {'semantic_types': semantic_types})

                    if 'https://metadata.datadrivendiscovery.org/types/TrueTarget' not in semantic_types:
                        semantic_types.append('https://metadata.datadrivendiscovery.org/types/TrueTarget')
                        self.dataset.metadata = self.dataset.metadata.update(
                            (resource_id, metadata_base.ALL_ELEMENTS, column_index),
                            {'semantic_types': semantic_types})

    def _create_metafeatures_pipeline(self, db, origin):

        pipeline = database.Pipeline(
            origin=origin,
            dataset=self.dataset_uri)

        def make_module(package, version, name):
            pipeline_module = database.PipelineModule(
                pipeline=pipeline,
                package=package, version=version, name=name)
            db.add(pipeline_module)
            return pipeline_module

        def make_data_module(name):
            return make_module('data', '0.0', name)

        def make_primitive_module(name):
            if name[0] == '.':
                name = 'd3m.primitives' + name
            return make_module('d3m', '2019.4.4', name)

        def connect(from_module, to_module,
                    from_output='produce', to_input='inputs'):
            db.add(database.PipelineConnection(pipeline=pipeline,
                                               from_module=from_module,
                                               to_module=to_module,
                                               from_output_name=from_output,
                                               to_input_name=to_input))

        input_data = make_data_module('dataset')
        db.add(database.PipelineParameter(
            pipeline=pipeline, module=input_data,
            name='targets', value=pickle.dumps(self.targets),
        ))
        db.add(database.PipelineParameter(
            pipeline=pipeline, module=input_data,
            name='features', value=pickle.dumps(self.features),
        ))

        # FIXME: Denormalize?
        #step0 = make_primitive_module('d3m.primitives.data_transformation.denormalize.Common')
        #connect(input_data, step0, from_output='dataset')

        step1 = make_primitive_module('d3m.primitives.data_transformation.dataset_to_dataframe.Common')
        connect(input_data, step1, from_output='dataset')
        #connect(step0, step1)

        step2 = make_primitive_module('d3m.primitives.data_transformation.column_parser.Common')
        connect(step1, step2)

        step3 = make_primitive_module('d3m.primitives.metalearning.metafeature_extractor.BYU')
        connect(step2, step3)

        db.add(pipeline)
        db.commit()
        logger.info(origin + ' PIPELINE ID: %s', pipeline.id)

        return pipeline.id

    def compute_metafeatures(self, origin):
        db = self.DBSession()
        # Add true and suggested targets
        self._add_target_columns_metadata()
        # Create the metafeatures computing pipeline
        pipeline_id = self._create_metafeatures_pipeline(db, origin)
        # Run the pipeline
        logger.info('Computing Metafeatures')
        metafeature_vector = {x: 0 for x in METAFEATURES_DEFAULT}
        try:
            # TODO Improve the sending of parameters
            outputs = execute(pipeline_id, self.dataset_uri, None, None, None,
                              db_filename=os.path.join(os.environ.get('D3MOUTPUTDIR'), 'temp', 'db.sqlite3'))
                             # TODO: Change this static string path
            for key, value in outputs.items():
                metafeature_results = value.metadata.query(())['data_metafeatures']
                for metafeature_key, metafeature_value in metafeature_results.items():
                    if isinstance(metafeature_value, frozendict.FrozenOrderedDict):
                        for k, v in metafeature_value.items():
                            if 'primitive' not in k:
                                metafeature_name = metafeature_key + '_' + k
                                if metafeature_name in metafeature_vector:
                                    metafeature_vector[metafeature_name] = v
                    else:
                        if metafeature_key in metafeature_vector:
                            metafeature_vector[metafeature_key] = metafeature_value

            return list(metafeature_vector.values())

        except Exception:
            logger.exception('Error running Metafeatures')
            # FIXME: This is a default to address metafeatures not generating features for datasets with numeric targets
            return list(metafeature_vector.values())
        finally:
            db.rollback()
            db.close()
