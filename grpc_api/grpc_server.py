"""GRPC server code, exposing AlphaD3M over the TA3-TA2 protocol.

Those adapters wrap the D3mTa2 object and handle all the GRPC and protobuf
logic, converting to/from protobuf messages. No GRPC or protobuf objects should
leave this module.
"""


import grpc
import json
import calendar
import datetime
import logging
from uuid import UUID
from os.path import join
from alphad3m import __version__
import d3m_automl_rpc.core_pb2 as pb_core
import d3m_automl_rpc.core_pb2_grpc as pb_core_grpc
import d3m_automl_rpc.problem_pb2 as pb_problem
import d3m_automl_rpc.value_pb2 as pb_value
import d3m_automl_rpc.pipeline_pb2 as pb_pipeline
import d3m_automl_rpc.primitive_pb2 as pb_primitive
from d3m_automl_rpc.utils import decode_pipeline_description, decode_problem_description, decode_performance_metric, \
    encode_pipeline_description, decode_value
from d3m.metadata import pipeline as pipeline_module
from d3m.metadata.problem import Problem
from google.protobuf.timestamp_pb2 import Timestamp
from alphad3m.grpc_api.grpc_logger import log_service
from alphad3m.primitive_loader import load_primitives_list
from alphad3m.utils import PersistentQueue

logger = logging.getLogger(__name__)


def to_timestamp(dt):
    """Converts a UTC datetime object into a gRPC Timestamp.

    :param dt: Time to convert, or None for now.
    :type dt: datetime.datetime | None
    """
    if dt is None:
        dt = datetime.datetime.utcnow()
    return Timestamp(seconds=calendar.timegm(dt.timetuple()),
                     nanos=dt.microsecond * 1000)


def error(context, code, format, *args):
    message = format % args
    context.set_code(code)
    context.set_details(message)
    if code == grpc.StatusCode.NOT_FOUND:
        return KeyError(message)
    else:
        return ValueError(message)


@log_service(logger)
class CoreService(pb_core_grpc.CoreServicer):

    def __init__(self, ta2):
        self._ta2 = ta2
        self._ta2.add_observer(self._ta2_event)
        self._requests = {}

    def _ta2_event(self, event, **kwargs):
        if 'job_id' in kwargs and kwargs['job_id'] in self._requests:
            job_id = kwargs['job_id']
            self._requests[job_id].put((event, kwargs))
            if event in ('scoring_success', 'scoring_error',
                         'training_success', 'training_error',
                         'test_success', 'test_error'):
                self._requests[job_id].close()

    def Hello(self, request, context):
        version = pb_core.DESCRIPTOR.GetOptions().Extensions[pb_core.protocol_version]
        user_agent = "alphad3m %s" % __version__

        return pb_core.HelloResponse(
            user_agent=user_agent,
            version=version,
            allowed_value_types=['RAW', 'DATASET_URI', 'CSV_URI'],
            supported_extensions=[],
            supported_task_keywords=[],  # TODO: Add supported_task_keywords using core package enums
            supported_performance_metrics=[],  # TODO: Add supported_performance_metrics using core package enums
            supported_evaluation_methods=['K_FOLD', 'HOLDOUT', 'RANKING'],
            supported_search_features=[]
        )

    def SearchSolutions(self, request, context):
        """Create a `Session` and start generating & scoring pipelines.
        """
        if len(request.inputs) > 1:
            raise error(context, grpc.StatusCode.UNIMPLEMENTED,
                        "Search with more than 1 input is not supported")
        expected_version = pb_core.DESCRIPTOR.GetOptions().Extensions[
            pb_core.protocol_version]

        if request.version != expected_version:
            logger.error("TA3 is using a different protocol version: %r "
                         "(us: %r)", request.version, expected_version)

        template = request.template

        if template is not None and len(template.steps) > 0:  # isinstance(template, pb_pipeline.PipelineDescription)
            pipeline = decode_pipeline_description(template, pipeline_module.Resolver())
            if pipeline.has_placeholder():
                template = pipeline.to_json_structure()
            else:  # Pipeline template fully defined
                problem = None
                if request.problem:
                    problem = decode_problem_description(request.problem)
                search_id = self._ta2.new_session(problem)
                dataset = request.inputs[0].dataset_uri
                if not dataset.startswith('file://'):
                    dataset = 'file://' + dataset

                self._ta2.build_fixed_pipeline(search_id, pipeline.to_json_structure(), dataset)
                return pb_core.SearchSolutionsResponse(search_id=str(search_id),)
        else:
            template = None

        dataset = request.inputs[0].dataset_uri
        if not dataset.endswith('datasetDoc.json'):
            raise error(context, grpc.StatusCode.INVALID_ARGUMENT,
                        "Dataset is not in D3M format: %s", dataset)
        if not dataset.startswith('file://'):
            dataset = 'file://' + dataset

        problem = decode_problem_description(request.problem)
        timeout_search = request.time_bound_search
        timeout_run = request.time_bound_run
        report_rank = True if request.rank_solutions_limit > 0 else False

        if timeout_search <= 0.0: timeout_search = None
        if timeout_run <= 0.0: timeout_run = None

        search_id = self._ta2.new_session(problem)
        session = self._ta2.sessions[search_id]
        task_keywords = session.problem['problem']['task_keywords']
        metrics = session.metrics

        automl_hyperparameters = {}
        if request.automl_hyperparameters:
            for hp_name, hp_value in request.automl_hyperparameters.items():
                automl_hyperparameters[hp_name] = decode_value(hp_value)['value']

        self._ta2.build_pipelines(search_id, dataset, task_keywords, metrics, timeout_search, timeout_run,
                                  automl_hyperparameters, template, report_rank=report_rank)

        return pb_core.SearchSolutionsResponse(
            search_id=str(search_id),
        )

    def GetSearchSolutionsResults(self, request, context):
        """Get the created pipelines and scores.
        """
        session_id = UUID(hex=request.search_id)
        if session_id not in self._ta2.sessions:
            raise error(context, grpc.StatusCode.NOT_FOUND,
                        "Unknown search ID %r", session_id)

        session = self._ta2.sessions[session_id]

        def msg_solution(pipeline_id):
            scores = self._ta2.get_pipeline_scores(pipeline_id)
            progress = session.progress

            if scores:
                if session.metrics and session.metrics[0]['metric'].name in scores:
                    metric = session.metrics[0]['metric']
                    try:
                        internal_score = metric.normalize(scores[metric.name])
                    except:
                        internal_score = scores[metric.name]
                        logger.warning('Problems normalizing metric, using the raw value: %.2f' % scores[metric.name])
                else:
                    internal_score = float('nan')
                scores = [
                    pb_core.Score(
                        metric=pb_problem.ProblemPerformanceMetric(
                            metric=m,
                            k=0,
                            pos_label=''),
                        value=pb_value.Value(
                            raw=pb_value.ValueRaw(double=s)
                        ),
                    )
                    for m, s in scores.items()
                ]
                scores = [pb_core.SolutionSearchScore(scores=scores)]
                return pb_core.GetSearchSolutionsResultsResponse(
                    done_ticks=progress.current,
                    all_ticks=progress.total,
                    progress=pb_core.Progress(
                        state=pb_core.RUNNING,
                        status="Solution scored",
                        start=to_timestamp(session.start),
                    ),
                    solution_id=str(pipeline_id),
                    internal_score=internal_score,
                    scores=scores,
                )

        def msg_progress(status, state=pb_core.RUNNING):
            progress = session.progress

            return pb_core.GetSearchSolutionsResultsResponse(
                done_ticks=progress.current,
                all_ticks=progress.total,
                progress=pb_core.Progress(
                    state=state,
                    status=status,
                    start=to_timestamp(session.start),
                ),
                internal_score=float('nan'),
            )

        def msg_fixed_solution(pipeline_id, state=pb_core.RUNNING):
            progress = session.progress

            return pb_core.GetSearchSolutionsResultsResponse(
                done_ticks=progress.current,
                all_ticks=progress.total,
                progress=pb_core.Progress(
                    state=state,
                    status="Solution Created",
                    start=to_timestamp(session.start),
                ),
                solution_id=str(pipeline_id),
                internal_score=float('nan'),
            )

        with session.with_observer_queue() as queue:
            # Send the solutions that already exist
            for pipeline_id in session.pipelines:
                msg = msg_solution(pipeline_id)
                if msg is not None:
                    yield msg

            # Send updates by listening to notifications on session
            while session.working or not queue.empty():
                if not context.is_active():
                    logger.info(
                        "Client closed GetSearchSolutionsResults stream")
                    break
                event, kwargs = queue.get()
                if event == 'finish_session' or event == 'done_searching':
                    break
                elif event == 'new_pipeline':
                    yield msg_progress("Trying new solution")
                elif event == 'new_fixed_pipeline':
                    pipeline_id = kwargs['pipeline_id']
                    yield msg_fixed_solution(pipeline_id)
                elif event == 'scoring_success':
                    pipeline_id = kwargs['pipeline_id']
                    msg = msg_solution(pipeline_id)
                    if msg is not None:
                        yield msg
                    else:
                        yield msg_progress("No appropriate score")
                elif event == 'scoring_error':
                    yield msg_progress("Solution doesn't work")

            yield msg_progress("End of search", pb_core.COMPLETED)

    def ScoreSolution(self, request, context):
        """Request scores for a pipeline.
        """
        pipeline_id = UUID(hex=request.solution_id)

        dataset = request.inputs[0].dataset_uri
        if not dataset.endswith('datasetDoc.json'):
            raise error(context, grpc.StatusCode.INVALID_ARGUMENT,
                        "Dataset is not in D3M format: %s", dataset)

        if dataset.startswith('/'):
            logger.warning("Dataset is a path, turning it into a file:// URL")
            dataset = 'file://' + dataset

        metrics = []
        for metric in request.performance_metrics:
                metrics.append(decode_performance_metric(metric))

        logger.info("Got ScoreSolution request, dataset=%s, "
                    "metrics=%s",
                    dataset, metrics)

        problem = None
        timeout_run = None
        for session_id in self._ta2.sessions.keys():
            if pipeline_id in self._ta2.sessions[session_id].pipelines:
                problem = self._ta2.sessions[session_id].problem
                timeout_run = self._ta2.sessions[session_id].timeout_run
                break

        scoring_config = {
                        'method': request.configuration.method,
                        'train_score_ratio': str(request.configuration.train_test_ratio),
                        'random_seed': request.configuration.random_seed,
                        'shuffle': str(request.configuration.shuffle).lower(),
                        'stratified': str(request.configuration.stratified).lower()
                        }
        if scoring_config['method'] == 'K_FOLD':
            scoring_config['number_of_folds'] = str(request.configuration.folds)

        job_id = self._ta2.score_pipeline(pipeline_id, metrics, dataset, problem, scoring_config, timeout_run)
        self._requests[job_id] = PersistentQueue()

        return pb_core.ScoreSolutionResponse(
            request_id='%x' % job_id,
        )

    def GetScoreSolutionResults(self, request, context):
        """Wait for a scoring job to be done.
        """
        try:
            job_id = int(request.request_id, 16)
            queue = self._requests[job_id]
        except (ValueError, KeyError):
            raise error(context, grpc.StatusCode.NOT_FOUND,
                        "Unknown ID %r", request.request_id)

        for event, kwargs in queue.read():
            if not context.is_active():
                logger.info("Client closed GetScoreSolutionResults stream")
                break

            if event == 'scoring_start':
                yield pb_core.GetScoreSolutionResultsResponse(
                    progress=pb_core.Progress(
                        state=pb_core.RUNNING,
                        status="Scoring in progress",
                    ),
                )
            elif event == 'scoring_success':
                pipeline_id = kwargs['pipeline_id']
                scores = self._ta2.get_pipeline_scores(pipeline_id)
                scores = [
                    pb_core.Score(
                        metric=pb_problem.ProblemPerformanceMetric(
                            metric=m,
                            k=0,
                            pos_label=''),
                        value=pb_value.Value(
                            raw=pb_value.ValueRaw(double=s)
                        ),
                    )
                    for m, s in scores.items()
                ]
                yield pb_core.GetScoreSolutionResultsResponse(
                    progress=pb_core.Progress(
                        state=pb_core.COMPLETED,
                        status="Scoring completed",
                    ),
                    scores=scores,
                )
                break
            elif event == 'scoring_error':
                status = kwargs['error_msg']
                yield pb_core.GetScoreSolutionResultsResponse(
                    progress=pb_core.Progress(
                        state=pb_core.ERRORED,
                        status=status,
                    ),
                )
                break

    def FitSolution(self, request, context):
        """Train a pipeline on a dataset.

        This will make it available for testing and exporting.
        """
        pipeline_id = UUID(hex=request.solution_id)
        dataset = request.inputs[0].dataset_uri
        steps_to_expose = list(request.expose_outputs)

        if dataset.startswith('/'):
            logger.warning("Dataset is a path, turning it into a file:// URL")
            dataset = 'file://' + dataset

        problem = None
        for session_id in self._ta2.sessions.keys():
            if pipeline_id in self._ta2.sessions[session_id].pipelines:
                problem = self._ta2.sessions[session_id].problem
                break

        job_id = self._ta2.train_pipeline(pipeline_id, dataset, problem, steps_to_expose)
        self._requests[job_id] = PersistentQueue()

        return pb_core.FitSolutionResponse(
            request_id='%x' % job_id,
        )

    def GetFitSolutionResults(self, request, context):
        """Wait for a training job to be done.
        """
        try:
            job_id = int(request.request_id, 16)
            queue = self._requests[job_id]
        except (ValueError, KeyError):
            raise error(context, grpc.StatusCode.NOT_FOUND,
                        "Unknown ID %r", request.request_id)

        for event, kwargs in queue.read():
            if not context.is_active():
                logger.info("Client closed GetFitSolutionsResults stream")
                break

            if event == 'training_start':
                yield pb_core.GetFitSolutionResultsResponse(
                    progress=pb_core.Progress(
                        state=pb_core.RUNNING,
                        status="Training in progress",
                    ),
                )
            elif event == 'training_success':
                pipeline_id = kwargs['pipeline_id']
                storage_dir = kwargs['storage_dir']
                steps_to_expose = kwargs['steps_to_expose']
                yield pb_core.GetFitSolutionResultsResponse(
                    progress=pb_core.Progress(
                        state=pb_core.COMPLETED,
                        status="Training completed",
                    ),
                    exposed_outputs={step_id: pb_value.Value(csv_uri='file://%s/fit_%s_%s.csv' %
                                                                     (storage_dir, pipeline_id, step_id))
                                     for step_id in steps_to_expose},
                    fitted_solution_id=str(pipeline_id),
                )
                break
            elif event == 'training_error':
                status = kwargs['error_msg']
                yield pb_core.GetFitSolutionResultsResponse(
                    progress=pb_core.Progress(
                        state=pb_core.ERRORED,
                        status=status,
                    ),
                )
                break
            elif event == 'done_searching':
                break

    def ProduceSolution(self, request, context):
        """Run testing from a trained pipeline.
        """
        pipeline_id = UUID(hex=request.fitted_solution_id)
        dataset = request.inputs[0].dataset_uri
        steps_to_expose = list(request.expose_outputs)

        if dataset.startswith('/'):
            logger.warning("Dataset is a path, turning it into a file:// URL")
            dataset = 'file://' + dataset

        job_id = self._ta2.test_pipeline(pipeline_id, dataset, steps_to_expose)
        self._requests[job_id] = PersistentQueue()

        return pb_core.ProduceSolutionResponse(
            request_id='%x' % job_id,
        )

    def GetProduceSolutionResults(self, request, context):
        """Wait for the requested test run to be done.
        """
        try:
            job_id = int(request.request_id, 16)
            queue = self._requests[job_id]
        except (ValueError, KeyError):
            raise error(context, grpc.StatusCode.NOT_FOUND,
                        "Unknown ID %r", request.request_id)

        for event, kwargs in queue.read():
            if not context.is_active():
                logger.info("Client closed GetProduceSolutionResults "
                            "stream")
                break
            if event == 'testing_success':
                pipeline_id = kwargs['pipeline_id']
                storage_dir = kwargs['storage_dir']
                steps_to_expose = kwargs['steps_to_expose']
                yield pb_core.GetProduceSolutionResultsResponse(
                    progress=pb_core.Progress(
                        state=pb_core.COMPLETED,
                        status="Execution completed",
                    ),
                    exposed_outputs={step_id: pb_value.Value(csv_uri='file://%s/produce_%s_%s.csv' %
                                                                     (storage_dir, pipeline_id, step_id))
                                     for step_id in steps_to_expose},
                )
                break
            elif event == 'testing_error':
                status = kwargs['error_msg']
                yield pb_core.GetProduceSolutionResultsResponse(
                    progress=pb_core.Progress(
                        state=pb_core.ERRORED,
                        status=status,
                    ),
                )
                break

    def SaveSolution(self, request, context):
        """Save a pipeline.
        """
        pipeline_id = UUID(hex=request.solution_id)
        session_id = None
        problem = None

        for session_id in self._ta2.sessions.keys():
            if pipeline_id in self._ta2.sessions[session_id].pipelines:
                problem = self._ta2.sessions[session_id].problem
                break

        solution_uri = self._ta2.save_pipeline(session_id, pipeline_id, problem)

        return pb_core.SaveSolutionResponse(
            solution_uri='%s' % solution_uri,
        )

    def LoadSolution(self, request, context):
        """Load a pipeline into a search session.
        """
        solution_path = request.solution_uri.replace('file://', '')

        with open(join(solution_path, 'pipeline.json')) as fin:
            pipeline = json.load(fin)

        with open(join(solution_path, 'problem.json')) as fin:
            problem_json = json.load(fin)
            problem = Problem.from_json_structure(problem_json)

        search_id = self._ta2.new_session(problem)
        solution_id = self._ta2.load_pipeline(search_id, pipeline)

        return pb_core.LoadSolutionResponse(
            solution_id='%s' % solution_id,
        )

    def SaveFittedSolution(self, request, context):
        """Save a trained pipeline.
        """
        pipeline_id = UUID(hex=request.fitted_solution_id)
        fitted_solution_uri = self._ta2.save_fitted_pipeline(str(pipeline_id))

        return pb_core.SaveFittedSolutionResponse(
            fitted_solution_uri='%s' % fitted_solution_uri,
        )

    def DescribeSolution(self, request, context):
        """Describe a pipeline as a grpc message.
        """
        pipeline_id = UUID(hex=request.solution_id)
        pipeline = self._ta2.get_pipeline(pipeline_id)

        if not pipeline:
            raise error(context, grpc.StatusCode.NOT_FOUND,
                        "Unknown solution ID %r", request.solution_id)

        grpc_pipeline = encode_pipeline_description(pipeline, ['RAW', 'DATASET_URI', 'CSV_URI'], self._ta2.output_folder)

        return pb_core.DescribeSolutionResponse(
            pipeline=grpc_pipeline,
            steps=None
        )

    def SolutionExport(self, request, context):
        """Export a pipeline.
        """
        pipeline_id = UUID(hex=request.solution_id)
        session_id = None
        for session_key in self._ta2.sessions:
            session = self._ta2.sessions[session_key]
            with session.lock:
                if pipeline_id in session.pipelines:
                    session_id = session.id
                    break
        if session_id is None:
            raise error(context, grpc.StatusCode.NOT_FOUND,
                        "Unknown solution ID %r", request.solution_id)
        session = self._ta2.sessions[session_id]
        rank = request.rank
        if rank < 0.0:
            rank = None

        session.write_exported_pipeline(pipeline_id, rank)
        return pb_core.SolutionExportResponse()

    def ListPrimitives(self, request, context):
        primitives = []
        installed_primitives = load_primitives_list()

        for primitive in installed_primitives:
            primitives.append(pb_primitive.Primitive(id=primitive['id'], version=primitive['version'],
                                                     python_path=primitive['python_path'], name=primitive['name'],
                                                     digest=primitive['digest']))

        return pb_core.ListPrimitivesResponse(primitives=primitives)

    def StopSearchSolutions(self, request, context):
        """Stop the search without deleting the `Session`.
        """
        session_id = UUID(hex=request.search_id)
        if session_id in self._ta2.sessions:
            self._ta2.stop_session(session_id)
            logger.info("Search stopped: %s", session_id)
        return pb_core.StopSearchSolutionsResponse()

    def EndSearchSolutions(self, request, context):
        """Stop the search and delete the `Session`.
        """
        session_id = UUID(hex=request.search_id)
        if session_id in self._ta2.sessions:
            self._ta2.finish_session(session_id)
            logger.info("Search terminated: %s", session_id)
        return pb_core.EndSearchSolutionsResponse()
