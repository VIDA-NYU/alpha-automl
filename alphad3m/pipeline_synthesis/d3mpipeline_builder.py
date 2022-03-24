import pickle
import logging
from alphad3m.schema import database
from d3m import index
from d3m.container import Dataset, DataFrame, ndarray, List
from alphad3m.utils import is_collection, get_collection_type

logger = logging.getLogger(__name__)

CONTAINER_CAST = {
    Dataset: {
        DataFrame: 'd3m.primitives.data_transformation.dataset_to_dataframe.Common',
        ndarray: ('d3m.primitives.data_transformation.dataset_to_dataframe.Common'
                  '|d3m.primitives.data_transformation.dataframe_to_ndarray.Common'),
        List: ('d3m.primitives.data_transformation.dataset_to_dataframe.Common'
               '|d3m.primitives.data_transformation.dataframe_to_list.Common')
    },
    DataFrame: {
        Dataset: '',
        ndarray: 'd3m.primitives.data_transformation.dataframe_to_ndarray.Common',
        List: 'd3m.primitives.data_transformation.dataframe_to_list.Common'
    },
    ndarray: {
        Dataset: '',
        DataFrame: 'd3m.primitives.data_transformation.ndarray_to_dataframe.Common',
        List: 'd3m.primitives.data_transformation.ndarray_to_list.Common'
    },
    List: {
        Dataset: '',
        DataFrame: 'd3m.primitives.data_transformation.list_to_dataframe.Common',
        ndarray: 'd3m.primitives.data_transformation.list_to_ndarray.Common',
    }
}


def make_pipeline_module(db, pipeline, name, package='d3m', version='2019.10.10'):
    pipeline_module = database.PipelineModule(pipeline=pipeline, package=package, version=version, name=name)
    db.add(pipeline_module)
    return pipeline_module


def make_data_module(db, pipeline, targets, features):
    input_data = make_pipeline_module(db, pipeline, 'dataset', 'data', '0.0')
    db.add(database.PipelineParameter(
        pipeline=pipeline, module=input_data,
        name='targets', value=pickle.dumps(targets),
    ))
    db.add(database.PipelineParameter(
        pipeline=pipeline, module=input_data,
        name='features', value=pickle.dumps(features),
    ))
    return input_data


def connect(db, pipeline, from_module, to_module, from_output='produce', to_input='inputs'):
    if 'index' not in from_output:
        if not from_module.name.startswith('dataset'):
            from_module_prim = index.get_primitive(from_module.name)
            from_module_output = from_module_prim.metadata.query()['primitive_code']['class_type_arguments']['Outputs']
        else:
            from_module_output = Dataset

        to_module_prim = index.get_primitive(to_module.name)
        to_module_input = to_module_prim.metadata.query()['primitive_code']['class_type_arguments']['Inputs']
        arguments = to_module_prim.metadata.query()['primitive_code']['arguments']

        if to_input not in arguments:
             raise NameError('Argument %s not found in %s' % (to_input, to_module.name))

        if to_module.name == 'd3m.primitives.feature_extraction.audio_transfer.DistilAudioTransfer':
            from_output = 'produce_collection'

        if from_module_output != to_module_input and \
                from_module.name != 'd3m.primitives.data_transformation.audio_reader.DistilAudioDatasetLoader':
            # FIXME: DistilAudioDatasetLoader has a bug https://github.com/uncharted-distil/distil-primitives/issues/294
            cast_module_steps = CONTAINER_CAST[from_module_output][to_module_input]
            if cast_module_steps:
                for cast_step in cast_module_steps.split('|'):
                    cast_module = make_pipeline_module(db, pipeline, cast_step)
                    db.add(database.PipelineConnection(pipeline=pipeline,
                                                       from_module=from_module,
                                                       to_module=cast_module,
                                                       from_output_name=from_output,
                                                       to_input_name='inputs'))
                    from_module = cast_module
            else:
                raise TypeError('Incompatible connection types: %s and %s for primitives %s and %s' % (
                    str(from_module_output), str(to_module_input), from_module.name, to_module.name))

    db.add(database.PipelineConnection(pipeline=pipeline,
                                       from_module=from_module,
                                       to_module=to_module,
                                       from_output_name=from_output,
                                       to_input_name=to_input))


def set_hyperparams(db, pipeline, module, **hyperparams):
    db.add(database.PipelineParameter(
        pipeline=pipeline, module=module,
        name='hyperparams', value=pickle.dumps(hyperparams),
    ))


def change_default_hyperparams(db, pipeline, primitive_name, primitive, learner_index=None):
    if primitive_name == 'd3m.primitives.data_transformation.one_hot_encoder.SKlearn':
        set_hyperparams(db, pipeline, primitive, use_semantic_types=True, return_result='replace', handle_unknown='ignore')
    elif primitive_name == 'd3m.primitives.data_cleaning.imputer.SKlearn':
        set_hyperparams(db, pipeline, primitive, use_semantic_types=True, return_result='replace', strategy='most_frequent', error_on_no_input=False)
    elif primitive_name.endswith('.SKlearn') and not (primitive_name.startswith('d3m.primitives.classification.') or
                                                      primitive_name.startswith('d3m.primitives.regression.')):
        set_hyperparams(db, pipeline, primitive, use_semantic_types=True, return_result='replace')
    elif primitive_name == 'd3m.primitives.data_transformation.enrich_dates.DistilEnrichDates':
        set_hyperparams(db, pipeline, primitive, replace=True)
    elif primitive_name == 'd3m.primitives.data_transformation.encoder.DSBOX':
        set_hyperparams(db, pipeline, primitive, n_limit=50)
    elif primitive_name == 'd3m.primitives.data_transformation.splitter.DSBOX':
        set_hyperparams(db, pipeline, primitive, threshold_row_length=2000)
    elif primitive_name == 'd3m.primitives.data_transformation.encoder.DistilTextEncoder':
        set_hyperparams(db, pipeline, primitive, encoder_type='tfidf')
    elif primitive_name == 'd3m.primitives.classification.text_classifier.DistilTextClassifier':
        set_hyperparams(db, pipeline, primitive, metric='accuracy')
    elif primitive_name == 'd3m.primitives.data_transformation.satellite_image_loader.DistilSatelliteImageLoader':
        set_hyperparams(db, pipeline, primitive, return_result='replace')
    elif primitive_name == 'd3m.primitives.clustering.k_means.DistilKMeans':
        set_hyperparams(db, pipeline, primitive, cluster_col_name='Class')
    elif primitive_name == 'd3m.primitives.data_transformation.adjacency_spectral_embedding.JHU':
        set_hyperparams(db, pipeline, primitive, use_attributes=True, max_dimension=5)
    elif primitive_name == 'd3m.primitives.graph_clustering.gaussian_clustering.JHU':
        set_hyperparams(db, pipeline, primitive, max_clusters=10)
    elif primitive_name == 'd3m.primitives.time_series_classification.convolutional_neural_net.LSTM_FCN':
        set_hyperparams(db, pipeline, primitive, epochs=1)
    elif primitive_name == 'd3m.primitives.time_series_forecasting.nbeats.DeepNeuralNetwork':
        set_hyperparams(db, pipeline, primitive, window_sampling_limit_multiplier=200, batch_size=10)
    elif primitive_name == 'd3m.primitives.time_series_forecasting.esrnn.RNN':
        set_hyperparams(db, pipeline, primitive, auto_tune=True)
    elif primitive_name == 'd3m.primitives.semisupervised_classification.iterative_labeling.AutonBox':
        if learner_index is not None:
            set_hyperparams(db, pipeline, primitive,  blackbox={'type': 'PRIMITIVE', 'data': learner_index})


def need_entire_dataframe(primitives):
    for primitive in primitives:
        if primitive in {'d3m.primitives.data_transformation.time_series_to_list.DSBOX',
                         'd3m.primitives.feature_extraction.random_projection_timeseries_featurization.DSBOX',
                         'd3m.primitives.data_transformation.dataframe_to_tensor.DSBOX',
                         'd3m.primitives.feature_extraction.resnet50_image_feature.DSBOX'}:
            return True
    return False


def need_parsed_targets(primitives):
    for primitive in primitives:
        if primitive in {'d3m.primitives.collaborative_filtering.link_prediction.DistilCollaborativeFiltering',
                         'd3m.primitives.time_series_forecasting.esrnn.RNN',
                         'd3m.primitives.time_series_forecasting.nbeats.DeepNeuralNetwork',
                         'd3m.primitives.time_series_forecasting.vector_autoregression.VAR'}:
            return True
    return False


def is_learner(primitive):
    if primitive.startswith('d3m.primitives.classification.'):
        return True
    return False


def is_linear_pipeline(primitives):
    for primitive in primitives:
        if primitive in {'d3m.primitives.data_transformation.load_single_graph.DistilSingleGraphLoader',
                         'd3m.primitives.data_transformation.load_graphs.DistilGraphLoader',
                         'd3m.primitives.data_transformation.load_graphs.JHU',
                         'd3m.primitives.data_transformation.load_edgelist.DistilEdgeListLoader',
                         'd3m.primitives.graph_matching.seeded_graph_matching.JHU'}:
            return True

    return False


def is_template_pipeline(primitives):
    for primitive in primitives:
        if primitive in {'d3m.primitives.graph_matching.euclidean_nomination.JHU',
                         'd3m.primitives.feature_extraction.yolo.DSBOX',
                         'd3m.primitives.object_detection.retina_net.ObjectDetectionRN'}:
            return True

    return False


def process_template(db, input_data, pipeline, pipeline_template, count_template_steps=0, prev_step=None):
    prev_steps = {}
    for pipeline_step in pipeline_template['steps']:
        if pipeline_step['type'] == 'PRIMITIVE':
            step = make_pipeline_module(db, pipeline, pipeline_step['primitive']['python_path'])
            if 'outputs' in pipeline_step:
                for output in pipeline_step['outputs']:
                    prev_steps['steps.%d.%s' % (count_template_steps, output['id'])] = step

            count_template_steps += 1
            if 'hyperparams' in pipeline_step:
                hyperparams = {}
                for hyper, desc in pipeline_step['hyperparams'].items():
                    hyperparams[hyper] = {'type': desc['type'], 'data': desc['data']}
                set_hyperparams(db, pipeline, step, **hyperparams)
        else:
            # TODO In the future we should be able to handle subpipelines
            break
        if prev_step:
            if 'arguments' in pipeline_step:
                for argument, desc in pipeline_step['arguments'].items():
                    connect(db, pipeline, prev_steps[desc['data']], step,
                            from_output=desc['data'].split('.')[-1], to_input=argument)
            # index is a special connection to keep the order of steps in fixed pipeline templates
            connect(db, pipeline, prev_step, step, from_output='index', to_input='index')
        else:
            connect(db, pipeline, input_data, step, from_output='dataset')
        prev_step = step

    return prev_step, count_template_steps


def add_semantic_types(db, metadata, pipeline, pipeline_template, prev_step):
    count_steps = 0
    if pipeline_template is None:
        for semantic_type, columns in metadata['semantictypes_indices'].items():
            step_add_type = make_pipeline_module(db, pipeline, 'd3m.primitives.data_transformation.'
                                                               'add_semantic_types.Common')
            count_steps += 1
            set_hyperparams(db, pipeline, step_add_type, columns=columns, semantic_types=[semantic_type])
            connect(db, pipeline, prev_step, step_add_type)
            prev_step = step_add_type
    else:
        step_add_type = make_pipeline_module(db, pipeline, 'd3m.primitives.schema_discovery.profiler.Common')
        count_steps += 1
        connect(db, pipeline, prev_step, step_add_type)
        prev_step = step_add_type
    return prev_step, count_steps


def add_file_readers(db, pipeline, prev_step, dataset_path):
    last_step = prev_step
    count_steps = 0
    if get_collection_type(dataset_path) == 'text':
        text_reader = make_pipeline_module(db, pipeline, 'd3m.primitives.data_transformation.text_reader.Common')
        count_steps += 1
        set_hyperparams(db, pipeline, text_reader, return_result='replace')
        connect(db, pipeline, prev_step, text_reader)
        last_step = text_reader

    elif get_collection_type(dataset_path) == 'image':
        image_reader = make_pipeline_module(db, pipeline, 'd3m.primitives.data_transformation.image_reader.Common')
        count_steps += 1
        set_hyperparams(db, pipeline, image_reader, return_result='replace')
        connect(db, pipeline, prev_step, image_reader)
        last_step = image_reader

    return last_step, count_steps


def add_rocauc_primitives(pipeline, current_step, prev_step, target_step, dataframe_step, learner_index, db):
    horizontal_concat = make_pipeline_module(db, pipeline, 'd3m.primitives.data_transformation.horizontal_concat.DataFrameCommon')
    # 'index'  is an artificial connection, just to guarantee the order of the steps
    connect(db, pipeline, current_step, horizontal_concat, from_output='index', to_input='index')
    connect(db, pipeline, prev_step, horizontal_concat, to_input='left')
    connect(db, pipeline, target_step, horizontal_concat, to_input='right')

    compute_values = make_pipeline_module(db, pipeline, 'd3m.primitives.operator.compute_unique_values.Common')
    connect(db, pipeline, horizontal_concat, compute_values)

    construct_confidence = make_pipeline_module(db, pipeline, 'd3m.primitives.data_transformation.construct_confidence.Common')
    set_hyperparams(db, pipeline, construct_confidence, primitive_learner={"type": "PRIMITIVE", "data": learner_index})
    connect(db, pipeline, compute_values, construct_confidence)
    connect(db, pipeline, dataframe_step, construct_confidence, to_input='reference')


def add_previous_primitives(db, pipeline, primitives, prev_step):
    remaining_primitives = []
    count_steps = 0
    for primitive in primitives:
        if need_entire_dataframe([primitive]):
            step_add_type = make_pipeline_module(db, pipeline, primitive)
            count_steps += 1
            connect(db, pipeline, prev_step, step_add_type)
            prev_step = step_add_type
        else:
            remaining_primitives.append(primitive)

    return prev_step, remaining_primitives, count_steps


def select_parsed_semantic_types(primitives, pipeline, step, db):
    for primitive in primitives:
        if primitive in {'d3m.primitives.data_transformation.grouping_field_compose.Common'}:
            set_hyperparams(db, pipeline, step,
                            parse_semantic_types=['http://schema.org/Boolean', 'http://schema.org/Integer',
                                                  'http://schema.org/Float', 'http://schema.org/DateTime',
                                                  'https://metadata.datadrivendiscovery.org/types/FloatVector']
                            )
        elif primitive in {'d3m.primitives.time_series_forecasting.arima.DSBOX'}:
            set_hyperparams(db, pipeline, step,
                            parse_semantic_types=['http://schema.org/Boolean', 'http://schema.org/Integer',
                                                  'http://schema.org/Float']
                            )


class BaseBuilder:

    def make_d3mpipeline(self, primitives, origin, dataset, pipeline_template, targets, features,
                         metadata, metrics=[], DBSession=None):
        # TODO: Remove parameters 'features' and 'targets' are not needed
        db = DBSession()
        dataset_path = dataset[7:]
        origin_name = '%s (%s)' % (origin, ', '.join([p.replace('d3m.primitives.', '') for p in primitives]))
        pipeline = database.Pipeline(origin=origin_name, dataset=dataset)
        count_steps = 0
        learner_index = None

        if is_linear_pipeline(primitives):
            return self.make_linear_pipeline(primitives, dataset, targets, features, origin, db)

        if is_template_pipeline(primitives):
            return self.make_template_pipeline(primitives, dataset, targets, features, origin, db)

        try:
            input_data = make_data_module(db, pipeline, targets, features)

            step0 = make_pipeline_module(db, pipeline, 'd3m.primitives.data_transformation.denormalize.Common')

            if metadata['large_rows']:
                step_sample = make_pipeline_module(db, pipeline, 'd3m.primitives.data_transformation.dataset_sample.Common')
                set_hyperparams(db, pipeline, step_sample, sample_size=metadata['sample_size'])
                connect(db, pipeline, step0, step_sample)
                count_steps += 1
                step0 = step_sample

            if not pipeline_template:
                connect(db, pipeline, input_data, step0, from_output='dataset')
            else:
                template_step, template_count = process_template(db, input_data, pipeline, pipeline_template)
                connect(db, pipeline, template_step, step0)
                count_steps += template_count

            step1 = make_pipeline_module(db, pipeline, 'd3m.primitives.data_transformation.dataset_to_dataframe.Common')
            connect(db, pipeline, step0, step1)
            count_steps += 1
            prev_step = step1

            if is_collection(dataset_path) and not need_entire_dataframe(primitives):
                prev_step, reader_steps = add_file_readers(db, pipeline, prev_step, dataset_path)
                count_steps += reader_steps

            if len(metadata['semantictypes_indices']) > 0:
                prev_step, semantic_steps = add_semantic_types(db, metadata, pipeline, pipeline_template, prev_step)
                count_steps += semantic_steps

            dataframe_step = prev_step
            if need_entire_dataframe(primitives):
                prev_step, primitives, primitive_steps = add_previous_primitives(db, pipeline, primitives, prev_step)
                count_steps += primitive_steps

            if metadata['large_columns']:
                # TODO: Remove this when https://gitlab.com/datadrivendiscovery/common-primitives/-/issues/149 is fixed
                step2 = make_pipeline_module(db, pipeline, 'd3m.primitives.data_transformation.column_parser.DistilColumnParser')
            else:
                step2 = make_pipeline_module(db, pipeline, 'd3m.primitives.data_transformation.column_parser.Common')
                select_parsed_semantic_types(primitives, pipeline, step2, db)

            connect(db, pipeline, prev_step, step2)
            count_steps += 1

            step3 = make_pipeline_module(db, pipeline, 'd3m.primitives.data_transformation.extract_columns_by_semantic_types.Common')
            set_hyperparams(db, pipeline, step3,
                            semantic_types=['https://metadata.datadrivendiscovery.org/types/Attribute'],
                            exclude_columns=metadata['exclude_columns'])
            connect(db, pipeline, step2, step3)
            count_steps += 1

            step4 = make_pipeline_module(db, pipeline, 'd3m.primitives.data_transformation.extract_columns_by_semantic_types.Common')
            set_hyperparams(db, pipeline, step4, semantic_types=['https://metadata.datadrivendiscovery.org/types/TrueTarget'])
            if need_parsed_targets(primitives):
                connect(db, pipeline, step2, step4)
            else:
                connect(db, pipeline, dataframe_step, step4)
            count_steps += 1

            current_step = prev_step = preprev_step = step3
            for primitive in primitives:
                current_step = make_pipeline_module(db, pipeline, primitive)
                change_default_hyperparams(db, pipeline, primitive, current_step, learner_index)

                if 'semisupervised_classification' in primitive:
                    connect(db, pipeline, preprev_step, current_step)
                    connect(db, pipeline, prev_step, current_step, from_output='index', to_input='index')
                else:
                    connect(db, pipeline, prev_step, current_step)

                if 'outputs' in index.get_primitive(primitive).metadata.query()['primitive_code']['arguments']:
                    connect(db, pipeline, step4, current_step, to_input='outputs')

                preprev_step = prev_step
                prev_step = current_step
                count_steps += 1
                if is_learner(primitive):
                    learner_index = count_steps

            if 'ROC_AUC' in metrics[0]['metric'].name:
                add_rocauc_primitives(pipeline, current_step, preprev_step, step4, dataframe_step, learner_index, db)
            else:
                step5 = make_pipeline_module(db, pipeline, 'd3m.primitives.data_transformation.construct_predictions.Common')
                connect(db, pipeline, current_step, step5)
                connect(db, pipeline, dataframe_step, step5, to_input='reference')

            db.add(pipeline)
            db.commit()
            logger.info('%s PIPELINE ID: %s', origin, pipeline.id)
            return pipeline.id
        except Exception as e:
            logger.exception('Error creating pipeline id=%s, primitives=%s', pipeline.id, str(primitives), exc_info=e)
            return None
        finally:
            db.close()

    def make_linear_pipeline(self, primitives, dataset, targets, features, origin, db):
        origin_name = '%s (%s)' % (origin, ', '.join([p.replace('d3m.primitives.', '') for p in primitives]))
        pipeline = database.Pipeline(origin=origin_name, dataset=dataset)

        try:
            input_data = make_data_module(db, pipeline, targets, features)
            step0 = make_pipeline_module(db, pipeline, primitives[0])
            connect(db, pipeline, input_data, step0, from_output='dataset')
            primitives = primitives[1:]
            prev_step = step0

            for primitive in primitives:
                current_step = make_pipeline_module(db, pipeline, primitive)
                change_default_hyperparams(db, pipeline, primitive, current_step)
                connect(db, pipeline, prev_step, current_step)
                if 'outputs' in index.get_primitive(primitive).metadata.query()['primitive_code']['arguments']:
                    connect(db, pipeline, prev_step, current_step, to_input='outputs', from_output='produce_target')
                prev_step = current_step

            db.add(pipeline)
            db.commit()
            logger.info('%s PIPELINE ID: %s', origin, pipeline.id)
            return pipeline.id

        except:
            logger.exception('Error creating pipeline id=%s, primitives=%s', pipeline.id, str(primitives))
            return None
        finally:
            db.close()

    def make_template_pipeline(self, primitives, dataset, targets, features, origin, db):
        # There are some primitives hard to be connected. So, use them in a pipeline template
        origin_name = '%s (%s)' % (origin, ', '.join([p.replace('d3m.primitives.', '') for p in primitives]))
        pipeline = database.Pipeline(origin=origin_name, dataset=dataset)

        try:
            input_data = make_data_module(db, pipeline, targets, features)
            if primitives[0] == 'd3m.primitives.graph_matching.euclidean_nomination.JHU':
                step0 = make_pipeline_module(db, pipeline,
                                             'd3m.primitives.data_transformation.dataset_to_dataframe.Common')
                connect(db, pipeline, input_data, step0)

                step1 = make_pipeline_module(db, pipeline,
                                             'd3m.primitives.data_transformation.dataset_to_dataframe.Common')
                set_hyperparams(db, pipeline, step1, dataframe_resource='1')
                connect(db, pipeline, input_data, step1)

                step2 = make_pipeline_module(db, pipeline,
                                             'd3m.primitives.data_transformation.dataset_to_dataframe.Common')
                set_hyperparams(db, pipeline, step2, dataframe_resource='2')
                connect(db, pipeline, input_data, step2)

                step3 = make_pipeline_module(db, pipeline, primitives[0])
                connect(db, pipeline, step1, step3, to_input='inputs_1')
                connect(db, pipeline, step2, step3, to_input='inputs_2')
                connect(db, pipeline, step0, step3, to_input='reference')

            elif primitives[0] == 'd3m.primitives.feature_extraction.yolo.DSBOX':
                # FIXME: It's simple to generalize and use in make_d3mpipeline
                step0 = make_pipeline_module(db, pipeline, 'd3m.primitives.data_transformation.denormalize.Common')
                connect(db, pipeline, input_data, step0, from_output='dataset')
                step1 = make_pipeline_module(db, pipeline,
                                             'd3m.primitives.data_transformation.dataset_to_dataframe.Common')
                connect(db, pipeline, step0, step1)

                step2 = make_pipeline_module(db, pipeline,
                                             'd3m.primitives.data_transformation.extract_columns_by_semantic_types.Common')
                set_hyperparams(db, pipeline, step2,
                                semantic_types=['https://metadata.datadrivendiscovery.org/types/PrimaryMultiKey',
                                                'https://metadata.datadrivendiscovery.org/types/FileName']
                                )
                connect(db, pipeline, step1, step2)

                step3 = make_pipeline_module(db, pipeline, 'd3m.primitives.data_transformation.'
                                                           'extract_columns_by_semantic_types.Common')
                set_hyperparams(db, pipeline, step3,
                                semantic_types=['https://metadata.datadrivendiscovery.org/types/TrueTarget'],
                                )
                connect(db, pipeline, step1, step3)

                step4 = make_pipeline_module(db, pipeline, primitives[0])
                connect(db, pipeline, step2, step4)
                connect(db, pipeline, step3, step4, to_input='outputs')

                step5 = make_pipeline_module(db, pipeline,
                                             'd3m.primitives.data_transformation.construct_predictions.Common')
                connect(db, pipeline, step4, step5)
                connect(db, pipeline, step2, step5, to_input='reference')

            elif primitives[0] == 'd3m.primitives.object_detection.retina_net.ObjectDetectionRN':
                # FIXME: It's simple to generalize and use in make_linear_pipeline
                step0 = make_pipeline_module(db, pipeline, 'd3m.primitives.data_transformation.denormalize.Common')
                connect(db, pipeline, input_data, step0, from_output='dataset')
                step1 = make_pipeline_module(db, pipeline, 'd3m.primitives.data_transformation.'
                                                           'dataset_to_dataframe.Common')
                set_hyperparams(db, pipeline, step1)
                connect(db, pipeline, step0, step1)

                step2 = make_pipeline_module(db, pipeline, primitives[0])
                connect(db, pipeline, step1, step2)
                connect(db, pipeline, step1, step2, to_input='outputs')

            db.add(pipeline)
            db.commit()
            logger.info('%s PIPELINE ID: %s', origin, pipeline.id)
            return pipeline.id

        except:
            logger.exception('Error creating pipeline id=%s, primitives=%s', pipeline.id, str(primitives))
            return None
        finally:
            db.close()


class AudioBuilder(BaseBuilder):

    def make_d3mpipeline(self, primitives, origin, dataset, pipeline_template, targets, features,
                         metadata, metrics=[], DBSession=None):
        db = DBSession()
        origin_name = '%s (%s)' % (origin, ', '.join([p.replace('d3m.primitives.', '') for p in primitives]))
        pipeline = database.Pipeline(origin=origin_name, dataset=dataset)
        count_steps = 0

        try:
            input_data = make_data_module(db, pipeline, targets, features)
            prev_step = input_data
            while True:
                # Use primitives that need Dataset object as inputs
                if index.get_primitive(primitives[0]).metadata.query()['primitive_code']['arguments']['inputs']['type'] \
                        != Dataset:
                    break
                current_step = make_pipeline_module(db, pipeline, primitives[0])
                connect(db, pipeline, prev_step, current_step)
                prev_step = current_step
                count_steps += 1
                primitives = primitives[1:]

            dataframe_step = current_step
            step1 = make_pipeline_module(db, pipeline, 'd3m.primitives.data_transformation.column_parser.Common')
            set_hyperparams(db, pipeline, step1, parse_semantic_types=[
                        "http://schema.org/Boolean",
                        "http://schema.org/Integer",
                        "http://schema.org/Float",
                        "https://metadata.datadrivendiscovery.org/types/FloatVector"
                    ]
            )
            connect(db, pipeline, prev_step, step1)
            count_steps += 1

            step2 = make_pipeline_module(db, pipeline, 'd3m.primitives.data_transformation.extract_columns_by_semantic_types.Common')
            set_hyperparams(db, pipeline, step2, semantic_types=['https://metadata.datadrivendiscovery.org/types/TrueTarget'])
            connect(db, pipeline, step1, step2)
            count_steps += 1

            current_step = prev_step
            for primitive in primitives:
                current_step = make_pipeline_module(db, pipeline, primitive)
                change_default_hyperparams(db, pipeline, primitive, current_step)
                connect(db, pipeline, prev_step, current_step)
                count_steps += 1

                if primitive != primitives[-1]:  # Not update when it is the last primitive
                    prev_step = current_step

                to_module_primitive = index.get_primitive(primitive)
                if 'outputs' in to_module_primitive.metadata.query()['primitive_code']['arguments']:
                    connect(db, pipeline, step2, current_step, to_input='outputs')

            if 'ROC_AUC' in metrics[0]['metric'].name:
                add_rocauc_primitives(pipeline, current_step, prev_step, step2, dataframe_step, count_steps, db)
            else:
                step6 = make_pipeline_module(db, pipeline, 'd3m.primitives.data_transformation.'
                                                           'construct_predictions.Common')
                connect(db, pipeline, current_step, step6)
                connect(db, pipeline, step1, step6, to_input='reference')

            db.add(pipeline)
            db.commit()
            logger.info('%s PIPELINE ID: %s', origin, pipeline.id)
            return pipeline.id
        except:
            logger.exception('Error creating pipeline id=%s, primitives=%s', pipeline.id, str(primitives))
            return None
        finally:
            db.close()
