"""The AutoML class, that creates pipelines, train, and run them.

We use multiprocessing to run training in separate processes, sending messages
back to this process via a Queue.
"""

from concurrent import futures
import datetime
import grpc
import json
import logging
import os
import pickle
from queue import Empty, Queue
from sqlalchemy import select
from sqlalchemy.orm import aliased, joinedload, lazyload
from sqlalchemy.sql import func
from os.path import join, exists
import shutil
import multiprocessing
import subprocess
import threading
import time
import d3m_automl_rpc.core_pb2_grpc as pb_core_grpc
from uuid import uuid4, UUID
from alphad3m import __version__
from alphad3m.multiprocessing import Receiver, run_process
from alphad3m.grpc_api import grpc_server
from alphad3m.data_ingestion.data_profiler import profile_data
from alphad3m.pipeline_synthesis.templates import generate_pipelines
from alphad3m.utils import Observable, ProgressStatus, is_collection, get_dataset_sample, create_outputfolders, \
    read_streams, get_internal_scoring_config
from alphad3m.schema import database
from alphad3m.schema.convert import to_d3m_json
from d3m.container import Dataset
from d3m.metadata.problem import TaskKeyword, parse_problem_description
from d3m.metadata import pipeline as pipeline_module


#  #### These are hyperparameters of the search ####
PIPELINES_TO_TUNE = 0  # Number of pipelines (top k) to be tuned.
TIME_TO_TUNE = 0.15  # The ratio of the time to be used for the tuning phase.
TIME_TO_SCORE = 5  # In minutes. Internal time to score a pipeline during the searching phase.
MAX_RUNNING_TIME = 43800  # In minutes. If time is not provided for the searching either scoring, run it for 1 month.
MAX_RUNNING_PROCESSES = int(os.environ.get('D3MCPU', multiprocessing.cpu_count()))  # Number of processes to be used
USE_AUTOMATIC_GRAMMAR = False
PRIORITIZE_PRIMITIVES = True
EXCLUDE_PRIMITIVES = []
INCLUDE_PRIMITIVES = []
SEARCH_STRATEGIES = ['TEMPLATES', 'ALPHA_AUTOML']

logger = logging.getLogger(__name__)


class AutoML(Observable):
    def __init__(self, output_folder):
        Observable.__init__(self)
        self.output_folder = output_folder
        #  Internal folders
        self.internal_folder = os.path.join(self.output_folder, 'temp')
        self.runtime_folder = os.path.join(self.output_folder, 'temp', 'runtime_output')
        # TODO: Remove the runtime_folder. Use internal_folder instead
        create_outputfolders(self.internal_folder)
        create_outputfolders(self.runtime_folder)
        self.db_filename = os.path.join(self.internal_folder, 'db.sqlite3')

        logger.info("output_folder=%r", self.output_folder)

        self.dbengine, self.DBSession = database.connect(self.db_filename)

        self.sessions = {}
        self.executor = ThreadPoolExecutor(max_workers=16)
        self._run_queue = Queue()
        self._run_thread = threading.Thread(target=self._pipeline_running_thread)
        self._run_thread.setDaemon(True)
        self._run_thread.start()

        logger.warning("AlphaD3M started, version=%s", __version__)

    def run_search(self, dataset_path, problem_path=None, problem_config=None, timeout=None):
        if dataset_path[0] == '/':
            dataset_path = 'file://' + dataset_path

        problem = parse_problem_description(problem_path)
        task_keywords = problem['problem']['task_keywords']
        # Create session
        session = Session(self, problem, self.output_folder, self.DBSession)
        logger.info('Dataset: %s, task: %s, metrics: %s', dataset_path, '_'.join([x.name for x in task_keywords]),
                    session.metrics)
        self.sessions[session.id] = session

        # Create pipelines, NO TUNING
        with session.with_observer_queue() as queue:
            self.build_pipelines(session.id, dataset_path, task_keywords, session.metrics, timeout, None, None,
                                 tune=PIPELINES_TO_TUNE)

            while queue.get(True)[0] != 'done_searching':
                pass

        '''logger.info('Tuning pipelines...')

        # Now do tuning, when we already have written out some solutions
        with session.with_observer_queue() as queue:
            session.tune_when_ready()
            while queue.get(True)[0] != 'done_searching':
                pass'''

    def run_server(self, port=None):
        """Spin up the gRPC server to receive requests from a TA3 system.

        This is called by the ``ta2_serve`` executable. It is part of the
        TA2+TA3 evaluation.
        """

        if not port:
            port = 45042
        core_rpc = grpc_server.CoreService(self)
        server = grpc.server(self.executor, options=(("grpc.so_reuseport", 0),))
        pb_core_grpc.add_CoreServicer_to_server(
            core_rpc, server)
        server.add_insecure_port('[::]:%d' % port)
        logger.info("Started gRPC server on port %d", port)
        server.start()
        while True:
            time.sleep(60)

    def new_session(self, problem):
        session = Session(self, problem, self.output_folder, self.DBSession)
        self.sessions[session.id] = session

        return session.id

    def finish_session(self, session_id):
        session = self.sessions.pop(session_id)
        session.close()

    def stop_session(self, session_id):
        session = self.sessions[session_id]
        session.stop_requested = True

    def get_pipeline(self, pipeline_id):
        db = self.DBSession()
        try:
            pipeline = (
                db.query(database.Pipeline)
                    .filter(database.Pipeline.id == pipeline_id)
                    .options(joinedload(database.Pipeline.modules),
                             joinedload(database.Pipeline.connections))
            ).one()

            json_pipeline = to_d3m_json(pipeline)
            d3m_pipeline = pipeline_module.Pipeline.from_json_structure(json_pipeline, )
            # FIXME: from_json_structure is returning a different digest

            return d3m_pipeline

        finally:
            db.close()

    def get_pipeline_scores(self, pipeline_id):
        db = self.DBSession()
        try:
            # Find most recent evaluation
            evaluation_id = (
                select([database.Evaluation.id])
                .where(database.Evaluation.pipeline_id == pipeline_id)
                .order_by(database.Evaluation.date.desc())
            ).as_scalar()
            # Get scores from that evaluation
            scores = db.query(
                select([func.avg(database.EvaluationScore.value),
                        database.EvaluationScore.metric])
                .where(
                    database.EvaluationScore.evaluation_id ==
                    evaluation_id
                )
                .group_by(database.EvaluationScore.metric)
            ).all()
            return {metric: value for value, metric in scores}
        finally:
            db.close()

    def score_pipeline(self, pipeline_id, metrics, dataset_uri, problem, scoring_config, timeout_run):
        if timeout_run is None:
            timeout_run = MAX_RUNNING_TIME

        timeout_run = timeout_run * 60  # Minutes to seconds

        job = ScoreJob(self, pipeline_id, dataset_uri, metrics, problem, scoring_config, timeout_run)
        self._run_queue.put(job)
        return id(job)

    def train_pipeline(self, pipeline_id, dataset, problem, steps_to_expose):
        job = TrainJob(self, pipeline_id, dataset, problem, steps_to_expose)
        self._run_queue.put(job)
        return id(job)

    def test_pipeline(self, pipeline_id, dataset, steps_to_expose):
        job = TestJob(self, pipeline_id, dataset, steps_to_expose)
        self._run_queue.put(job)
        return id(job)

    def save_pipeline(self, session_id, pipeline_id, problem):
        pipeline_id = str(pipeline_id)
        session_id = str(session_id)
        pipeline_path = join(self.output_folder, session_id, 'pipelines_scored', '%s.json' % pipeline_id)

        if exists(pipeline_path):
            pipeline_folder_path = join(self.output_folder, 'temp', 'saved_pipelines', pipeline_id)
            create_outputfolders(pipeline_folder_path)
            shutil.copyfile(pipeline_path, join(pipeline_folder_path, 'pipeline.json'))
            problem_json = problem.to_json_structure()

            with open(join(pipeline_folder_path, 'problem.json'), 'w') as fout:
                json.dump(problem_json, fout, indent=2)

            return 'file://' + pipeline_folder_path

        return None

    def load_pipeline(self, session_id, pipeline):
        session = self.sessions[session_id]
        db = self.DBSession()
        dataset = 'N/A'

        pipeline_database = database.Pipeline(origin='Loaded pipeline', dataset=dataset)

        # TODO: Do it on d3mpipeline_generator.py
        def make_module(package, version, name):
            pipeline_module = database.PipelineModule(
                pipeline=pipeline_database,
                package=package, version=version, name=name)
            db.add(pipeline_module)
            return pipeline_module

        def make_data_module(name):
            return make_module('data', '0.0', name)

        def make_primitive_module(name):
            if name[0] == '.':
                name = 'd3m.primitives' + name
            return make_module('d3m', '2018.7.10', name)

        def connect(from_module, to_module,
                    from_output='produce', to_input='inputs'):
            db.add(database.PipelineConnection(pipeline=pipeline_database,
                                               from_module=from_module,
                                               to_module=to_module,
                                               from_output_name=from_output,
                                               to_input_name=to_input))

        def set_hyperparams(module, **hyperparams):
            db.add(database.PipelineParameter(
                pipeline=pipeline_database, module=module,
                name='hyperparams', value=pickle.dumps(hyperparams),
            ))

        try:
            # TODO: Use pipeline input for this
            if dataset:
                input_data = make_data_module('dataset')
                db.add(database.PipelineParameter(
                    pipeline=pipeline_database, module=input_data,
                    name='targets', value=pickle.dumps('targets'),
                ))
                db.add(database.PipelineParameter(
                    pipeline=pipeline_database, module=input_data,
                    name='features', value=pickle.dumps('features'),
                ))
            prev_step = None
            prev_steps = {}
            count_template_steps = 0
            for pipeline_step in pipeline['steps']:
                if pipeline_step['type'] == 'PRIMITIVE':
                    step = make_primitive_module(pipeline_step['primitive']['python_path'])
                    if 'outputs' in pipeline_step:
                        for output in pipeline_step['outputs']:
                            prev_steps['steps.%d.%s' % (count_template_steps, output['id'])] = step

                    count_template_steps += 1
                    if 'hyperparams' in pipeline_step:
                        hyperparams = {}
                        for hyper, desc in pipeline_step['hyperparams'].items():
                            hyperparams[hyper] = {'type': desc['type'], 'data': desc['data']}
                        set_hyperparams(step, **hyperparams)
                else:
                    # TODO In the future we should be able to handle subpipelines
                    break
                if prev_step:
                    if 'arguments' in pipeline_step:
                        for argument, desc in pipeline_step['arguments'].items():
                            connect(prev_steps[desc['data']], step, from_output=desc['data'].split('.')[-1],
                                    to_input=argument)
                    connect(prev_step, step, from_output='index', to_input='index')
                else:
                    connect(input_data, step, from_output='dataset')
                prev_step = step
            db.add(pipeline_database)
            db.commit()
            pipeline_id = pipeline_database.id
            logger.info("Created fixed pipeline %s", pipeline_id)
            session.write_searched_pipeline(pipeline_id)

            session.notify('new_fixed_pipeline', pipeline_id=pipeline_id)
            with session.lock:
                session.pipelines.add(pipeline_id)
                # Force working=True so we get 'done_searching' even if no pipeline
                # gets created
                session.working = False
        finally:
            db.close()

        return pipeline_id

    def save_fitted_pipeline(self, pipeline_id):
        fitted_pipeline_path = join(self.runtime_folder, 'fitted_solution_%s.pkl' % pipeline_id)
        if exists(fitted_pipeline_path):
            pipeline_path_dst = join(self.output_folder, 'temp', 'saved_fitted_pipelines', pipeline_id)
            create_outputfolders(pipeline_path_dst)
            shutil.copyfile(fitted_pipeline_path, join(pipeline_path_dst, 'fitted_solution_%s.pkl' % pipeline_id))
            return 'file://' + pipeline_path_dst

        return None

    def build_pipelines(self, session_id, dataset, task_keywords, metrics, timeout_search, timeout_run, hyperparameters,
                        template=None, targets=None, features=None, tune=None, report_rank=False):
        self.executor.submit(self._build_pipelines, session_id, dataset, task_keywords, metrics, timeout_search,
                             timeout_run, hyperparameters, template, targets, features, tune, report_rank)

    def build_fixed_pipeline(self, session_id, pipeline, dataset, targets=None, features=None):
        self.executor.submit(self._build_fixed_pipeline, session_id, pipeline, dataset, targets, features)

    # Runs in a worker thread from executor
    def _build_fixed_pipeline(self, session_id, pipeline_template, dataset, targets, features):

        session = self.sessions[session_id]
        with session.lock:
            # Force working=True so we get 'done_searching' even if no pipeline
            # gets created
            session.working = True

        db = self.DBSession()

        if dataset:
            dataset_uri = dataset
        else:
            dataset_uri = 'NA'

        pipeline_database = database.Pipeline(origin='Fixed pipeline template', dataset=dataset_uri)

        # TODO: Do it on d3mpipeline_generator.py
        def make_module(package, version, name):
            pipeline_module = database.PipelineModule(
                pipeline=pipeline_database,
                package=package, version=version, name=name)
            db.add(pipeline_module)
            return pipeline_module

        def make_data_module(name):
            return make_module('data', '0.0', name)

        def make_primitive_module(name):
            if name[0] == '.':
                name = 'd3m.primitives' + name
            return make_module('d3m', '2018.7.10', name)

        def connect(from_module, to_module,
                    from_output='produce', to_input='inputs'):
            db.add(database.PipelineConnection(pipeline=pipeline_database,
                                               from_module=from_module,
                                               to_module=to_module,
                                               from_output_name=from_output,
                                               to_input_name=to_input))

        def set_hyperparams(module, **hyperparams):
            db.add(database.PipelineParameter(
                pipeline=pipeline_database, module=module,
                name='hyperparams', value=pickle.dumps(hyperparams),
            ))

        try:
            # TODO: Use pipeline input for this
            if dataset:
                input_data = make_data_module('dataset')
                db.add(database.PipelineParameter(
                    pipeline=pipeline_database, module=input_data,
                    name='targets', value=pickle.dumps(targets),
                ))
                db.add(database.PipelineParameter(
                    pipeline=pipeline_database, module=input_data,
                    name='features', value=pickle.dumps(features),
                ))
            prev_step = None
            prev_steps = {}
            count_template_steps = 0
            for pipeline_step in pipeline_template['steps']:
                if pipeline_step['type'] == 'PRIMITIVE':
                    step = make_primitive_module(pipeline_step['primitive']['python_path'])
                    if 'outputs' in pipeline_step:
                        for output in pipeline_step['outputs']:
                            prev_steps['steps.%d.%s' % (count_template_steps,output['id'])] = step

                    count_template_steps += 1
                    if 'hyperparams' in pipeline_step:
                        hyperparams = {}
                        for hyper, desc in pipeline_step['hyperparams'].items():
                            hyperparams[hyper] = {'type':desc['type'] ,'data':desc['data']}
                        set_hyperparams(step, **hyperparams)
                else:
                    # TODO In the future we should be able to handle subpipelines
                    break
                if prev_step:
                    if 'arguments' in pipeline_step:
                        for argument, desc in pipeline_step['arguments'].items():
                            connect(prev_steps[desc['data']], step, from_output=desc['data'].split('.')[-1], to_input=argument)
                    connect(prev_step, step, from_output='index', to_input='index')
                else:
                    connect(input_data, step, from_output='dataset')
                prev_step = step
            db.add(pipeline_database)
            db.commit()
            pipeline_id = pipeline_database.id
            logger.info("Created fixed pipeline %s", pipeline_id)
            session.write_searched_pipeline(pipeline_id)

            session.notify('new_fixed_pipeline', pipeline_id=pipeline_id)
            with session.lock:
                session.pipelines.add(pipeline_id)
                # Force working=True so we get 'done_searching' even if no pipeline
                # gets created
                session.working = False
        finally:
            db.close()

    # Runs in a worker thread from executor
    def _build_pipelines(self, session_id, dataset_uri, task_keywords, metrics, timeout_search, timeout_run,
                         hyperparameters, pipeline_template, targets, features, tune, report_rank):
        """Generates pipelines for the session.
        """
        session = self.sessions[session_id]
        with session.lock:
            session.targets = targets
            session.features = features
            if session.metrics != metrics:
                if session.metrics:
                    old = 'from %s ' % ', '.join([m['metric'] for m in session.metrics])
                else:
                    old = ''
                session.metrics = metrics
                logger.info("Set metrics to %s %s(for session %s)", metrics, old, session_id)

            # Force working=True so we get 'done_searching' even if no pipeline
            # gets created
            session.working = True

        if timeout_search is None:
            timeout_search = MAX_RUNNING_TIME

        if 'use_automatic_grammar' not in hyperparameters:
            hyperparameters['use_automatic_grammar'] = USE_AUTOMATIC_GRAMMAR

        if 'prioritize_primitives' not in hyperparameters:
            hyperparameters['prioritize_primitives'] = PRIORITIZE_PRIMITIVES

        if 'include_primitives' not in hyperparameters or hyperparameters['include_primitives'] is None:
            hyperparameters['include_primitives'] = INCLUDE_PRIMITIVES

        if 'exclude_primitives' not in hyperparameters or hyperparameters['exclude_primitives'] is None:
            hyperparameters['exclude_primitives'] = EXCLUDE_PRIMITIVES

        timeout_search = timeout_search * 60  # Minutes to seconds
        timeout_search_internal = timeout_search

        if PIPELINES_TO_TUNE > 0:
            timeout_search_internal = timeout_search * (1 - TIME_TO_TUNE)

        now = time.time()
        expected_search_end = now + timeout_search
        sample_dataset_uri = self._get_sample_uri(dataset_uri, session.problem)
        metadata = profile_data(dataset_uri, session.targets)
        time_already_used = time.time() - now  # Time already used for sampling and profiling
        timeout_search_internal = timeout_search_internal - time_already_used
        session.dataset_uri = dataset_uri
        session.sample_dataset_uri = sample_dataset_uri
        session.report_rank = report_rank
        session.timeout_run = timeout_run
        session.expected_search_end = expected_search_end

        if 'TEMPLATES' in SEARCH_STRATEGIES:
            self._build_pipeline_from_template(session, task_keywords, dataset_uri, sample_dataset_uri, metrics,
                                               hyperparameters, metadata, timeout_search_internal)
        if 'ALPHA_AUTOML' in SEARCH_STRATEGIES:
            self._build_pipelines_from_generator(session, task_keywords, dataset_uri, sample_dataset_uri, metrics,
                                                 hyperparameters, metadata, timeout_search_internal, pipeline_template)

        session.tune_when_ready(tune)

    def _build_pipelines_from_generator(self, session, task_keywords, dataset_uri, sample_dataset_uri, metrics,
                                        hyperparameters, metadata, timeout_search, pipeline_template):
        logger.info("Starting AlphaD3M process, timeout is %s", timeout_search)
        msg_queue = Receiver()
        proc = run_process(
            'alphad3m.pipeline_synthesis.setup_search.generate_pipelines',
            'generate',
            msg_queue,
            task_keywords=task_keywords,
            dataset=dataset_uri,
            metrics=metrics,
            problem=session.problem,
            targets=session.targets,
            features=session.features,
            hyperparameters=hyperparameters,
            metadata=metadata,
            pipeline_template=pipeline_template,
            time_bound=timeout_search,
            db_filename=self.db_filename,
        )

        start = time.time()
        stop = False

        # Now we wait for pipelines to be sent over the pipe
        while proc.poll() is None:
            if not stop:
                if session.stop_requested:
                    logger.error("Session stop requested, sending SIGTERM to generator process")
                    stop = True

                if time.time() > start + timeout_search:
                    logger.error("Reached search timeout (%d > %d seconds), sending SIGTERM to generator process",
                                 time.time() - start, timeout_search)
                    stop = True

                if stop:
                    proc.terminate()
                    try:
                        proc.wait(30)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
            try:
                msg, *args = msg_queue.recv(3)
            except Empty:
                continue

            if msg == 'eval':
                pipeline_id, = args
                logger.info("Got pipeline %s from generator process", pipeline_id)
                score = self.run_pipeline(session, dataset_uri, sample_dataset_uri, task_keywords, pipeline_id)
                logger.info("Sending score to generator process")
                if not stop:
                    msg_queue.send(score)
            else:
                raise RuntimeError("Got unknown message from generator process: %r" % msg)

        logger.warning("Generator process exited with %r", proc.returncode)

    def _build_pipeline_from_template(self, session, task_keywords, dataset_uri, sample_dataset_uri, metrics,
                                      hyperparameters, metadata, timeout_search):

        pipeline_ids = generate_pipelines(task_keywords, dataset_uri, session.problem, session.targets, session.features,
                                          hyperparameters, metadata,  metrics, self.DBSession)
        for pipeline_id in pipeline_ids:
            if pipeline_id is not None:
                try:
                    # Add it to the session
                    session.add_scoring_pipeline(pipeline_id)
                    logger.info('Created pipeline %s', pipeline_id)
                    scoring_config = get_internal_scoring_config(task_keywords)

                    self._run_queue.put(ScoreJob(self, pipeline_id, dataset_uri, session.metrics, session.problem,
                                                 scoring_config, timeout_search, session.report_rank,
                                                 sample_dataset_uri))

                    session.notify('new_pipeline', pipeline_id=pipeline_id)
                except Exception:
                    logger.exception('Error building pipeline from template')

    def run_pipeline(self, session, dataset_uri, sample_dataset_uri, task_keywords, pipeline_id):

        """Score a single pipeline.

        This is used by the pipeline synthesis code.
        """
        scoring_config = get_internal_scoring_config(task_keywords)
        timeout_run = session.timeout_run

        if timeout_run is None:
            timeout_run = TIME_TO_SCORE
            if any(tk in [TaskKeyword.VIDEO, TaskKeyword.IMAGE, TaskKeyword.AUDIO] for tk in task_keywords):
                timeout_run = timeout_run * 2  # Increase the score timeout for video, image and audio datasets

        timeout_run = timeout_run * 60  # Minutes to seconds

        # Add the pipeline to the session, score it
        with session.with_observer_queue() as queue:
            session.add_scoring_pipeline(pipeline_id)
            logger.info("Created pipeline %s", pipeline_id)
            self._run_queue.put(ScoreJob(self, pipeline_id, dataset_uri, session.metrics, session.problem,
                                         scoring_config, timeout_run, session.report_rank, sample_dataset_uri))
            session.notify('new_pipeline', pipeline_id=pipeline_id)

            while True:
                event, kwargs = queue.get(True)
                if event == 'done_searching':
                    raise RuntimeError("Never got pipeline results")
                elif (event == 'scoring_error' and
                      kwargs['pipeline_id'] == pipeline_id):
                    return None
                elif (event == 'scoring_success' and
                      kwargs['pipeline_id'] == pipeline_id):
                    break

        db = self.DBSession()
        try:
            # Find most recent evaluation
            evaluation_id = (
                select([database.Evaluation.id])
                    .where(database.Evaluation.pipeline_id == pipeline_id)
                    .order_by(database.Evaluation.date.desc())
            ).as_scalar()
            # Get scores from that evaluation
            scores = db.query(
                select([func.avg(database.EvaluationScore.value),
                        database.EvaluationScore.metric])
                    .where(
                    database.EvaluationScore.evaluation_id ==
                    evaluation_id
                )
                    .group_by(database.EvaluationScore.metric)
            ).all()

            first_metric = session.metrics[0]['metric'].name
            for value, metric in scores:
                if metric == first_metric:
                    logger.info("Evaluation result: %s -> %r", metric, value)
                    return value
            logger.info("Didn't get the requested metric from evaluation")
            return None
        finally:
            db.close()

    def _get_sample_uri(self, dataset_uri, problem):
        logger.info('About to sample dataset %s', dataset_uri)
        task_keywords = problem['problem']['task_keywords']

        if any(tk in [TaskKeyword.OBJECT_DETECTION, TaskKeyword.FORECASTING, TaskKeyword.GRAPH] for tk in task_keywords):
            logger.info('Not doing sampling for task %s', '_'.join([x.name for x in task_keywords]))
            return None

        dataset = Dataset.load(dataset_uri)

        if is_collection(dataset_uri[7:]):
            logger.info('Not doing sampling for collections')
            return None

        dataset_sample_folder = 'file://%s/temp/dataset_sample/' % os.environ.get('D3MOUTPUTDIR')
        dataset_sample_uri = None

        if os.path.exists(dataset_sample_folder[6:]):
            shutil.rmtree(dataset_sample_folder[6:])

        dataset_sample = get_dataset_sample(dataset, problem, dataset_sample_folder)

        if isinstance(dataset_sample, str):  # Was the dataset sampled?
            dataset_sample_uri = dataset_sample

        return dataset_sample_uri

    # Runs in a background thread
    def _pipeline_running_thread(self):
        running_jobs = {}
        while True:
            # Poll jobs, remove finished ones
            remove = []
            for job in running_jobs.values():
                if job.check():
                    remove.append(id(job))
            for job_id in remove:
                del running_jobs[job_id]

            # Start new jobs if we are under the maximum
            if len(running_jobs) < MAX_RUNNING_PROCESSES:
                try:
                    job = self._run_queue.get(False)
                except Empty:
                    pass
                else:
                    job.start(db_filename=self.db_filename,
                              predictions_root=self.runtime_folder)
                    running_jobs[id(job)] = job

            time.sleep(3)


class Session(Observable):
    """A session, in the gRPC meaning.

    This corresponds to a search in which pipelines are created.
    """
    def __init__(self, ta2, problem, output_folder, DBSession):
        Observable.__init__(self)
        self.id = uuid4()
        self._ta2 = ta2
        self.problem = problem
        self.DBSession = DBSession

        # D3M folders
        str_id = str(self.id)
        self._searched_pipelines_dir = os.path.join(output_folder, str_id, 'pipelines_searched')
        self._scored_pipelines_dir = os.path.join(output_folder, str_id, 'pipelines_scored')
        self._ranked_pipelines_dir = os.path.join(output_folder, str_id, 'pipelines_ranked')
        self._run_pipelines_dir = os.path.join(output_folder, str_id, 'pipeline_runs')
        create_outputfolders(self._searched_pipelines_dir)
        create_outputfolders(self._scored_pipelines_dir)
        create_outputfolders(self._ranked_pipelines_dir)
        create_outputfolders(self._run_pipelines_dir)

        self.metrics = []
        self.report_rank = False

        self._observer = self._ta2.add_observer(self._ta2_event)

        self.start = datetime.datetime.utcnow()

        # Should tuning be triggered when we are done with current pipelines?
        self._tune_when_ready = None

        # All the pipelines that belong to this session
        self.pipelines = set()
        # The pipelines currently in the queue for scoring
        self.pipelines_scoring = set()
        # The pipelines in the queue for hyperparameter tuning
        self.pipelines_tuning = set()
        # Pipelines already tuned, and pipelines created through tuning
        self.tuned_pipelines = set()
        # Flag indicating we started scoring & tuning, and a
        # 'done_searching' signal should be sent once no pipeline is pending
        self.working = False
        # Flag allowing TA3 to stop the search early
        self.stop_requested = False

        # Read metrics from problem
        if self.problem is not None:
            self.metrics = problem['problem']['performance_metrics']

        self._targets = None
        self._features = None
        self.dataset_uri = None
        self.sample_dataset_uri = None
        self.timeout_run = None
        self.expected_search_end = None

    @property
    def problem_id(self):
        return self.problem['id']

    @property
    def targets(self):
        if self._targets is not None:
            return set(self._targets)
        else:
            # Read targets from problem
            targets = set()
            #assert len(self.problem['inputs']) == 1
            for target in self.problem['inputs'][0]['targets']:
                targets.add((target['resource_id'], target['column_name']))
            return targets

    @targets.setter
    def targets(self, value):
        if value is None:
            self._targets = None
        elif isinstance(value, (set, list)):
            if not value:
                raise ValueError("Can't set targets to empty set")
            self._targets = set(value)
        else:
            raise TypeError("targets should be a set or None")

    @property
    def features(self):
        return self._features

    @features.setter
    def features(self, value):
        if value is None:
            self._features = None
        elif isinstance(value, (set, list)):
            if not value:
                raise ValueError("Can't set features to empty set")
            self._features = set(value)
        else:
            raise TypeError("features should be a set or None")

    def _ta2_event(self, event, **kwargs):
        if event == 'scoring_start':
            if kwargs['pipeline_id'] in self.pipelines_scoring:
                logger.info("Scoring pipeline for %s (session %s has %d "
                            "pipelines left to score)",
                            kwargs['pipeline_id'], self.id,
                            len(self.pipelines_scoring))
                self.notify(event, **kwargs)
        elif event == 'scoring_success' or event == 'scoring_error':
            if kwargs['pipeline_id'] in self.pipelines_scoring:
                self.notify(event, **kwargs)
                self.pipeline_scoring_done(kwargs['pipeline_id'], event)

    def tune_when_ready(self, tune=None):
        if tune is None:
            tune = PIPELINES_TO_TUNE
        self._tune_when_ready = tune
        self.working = True
        self.check_status()

    def add_scoring_pipeline(self, pipeline_id):
        with self.lock:
            self.working = True
            self.pipelines.add(pipeline_id)
            self.pipelines_scoring.add(pipeline_id)
            self.write_searched_pipeline(pipeline_id)

    def pipeline_scoring_done(self, pipeline_id, event=None):
        with self.lock:
            self.pipelines_scoring.discard(pipeline_id)
            self.check_status()
            if event == 'scoring_success':
                self.write_scored_pipeline(pipeline_id)

    def pipeline_tuning_done(self, old_pipeline_id, new_pipeline_id=None):
        with self.lock:
            self.pipelines_tuning.discard(old_pipeline_id)
            self.tuned_pipelines.add(old_pipeline_id)
            if new_pipeline_id is not None:
                self.pipelines.add(new_pipeline_id)
                self.tuned_pipelines.add(new_pipeline_id)
                self.write_searched_pipeline(new_pipeline_id)
                self.write_scored_pipeline(new_pipeline_id)
            self.check_status()

    @property
    def progress(self):
        if self._tune_when_ready is not None:
            to_tune = self._tune_when_ready - len(self.tuned_pipelines) / 2
        else:
            to_tune = 0
        return ProgressStatus(
            current=len(self.pipelines) - len(self.pipelines_scoring),
            total=len(self.pipelines) + to_tune,
        )

    def get_top_pipelines(self, db, metric, limit=None):
        metric_name = metric.name
        pipeline = aliased(database.Pipeline)
        evaluation_score = (
            select([func.avg(database.EvaluationScore.value)])
            .where(database.EvaluationScore.evaluation_id ==
                   database.Evaluation.id)
            .where(database.EvaluationScore.metric == metric_name)
            .where(database.Evaluation.pipeline_id == pipeline.id)
            .as_scalar()
        )
        if metric.best_value() == 1:
            eval_score_order = evaluation_score.desc()
        else:
            eval_score_order = evaluation_score.asc()  # Error based metrics
        q = (
            db.query(pipeline, evaluation_score)
            .filter(pipeline.id.in_(self.pipelines))
            .filter(evaluation_score != None)
            # FIXME: Using a joined load here results in duplicated results
            .options(lazyload(pipeline.parameters))
            .order_by(eval_score_order)
        )
        if limit is not None:
            q = q.limit(limit)
        return q.all()

    def check_status(self):
        with self.lock:
            # Session is not to be finished automatically
            if self._tune_when_ready is None:
                return
            # We are already done
            if not self.working:
                return
            # If pipelines are still in the queue
            if self.pipelines_scoring or self.pipelines_tuning:
                return

            db = self.DBSession()
            try:
                # If we are out of pipelines to score, maybe submit pipelines for
                # tuning
                logger.info("Session %s: scoring done", self.id)

                tune = []
                if self._tune_when_ready and self.stop_requested:
                    logger.info("Session stop requested, skipping tuning")
                elif self._tune_when_ready:
                    top_pipelines = self.get_top_pipelines(
                        db, self.metrics[0]['metric'],
                        self._tune_when_ready)
                    for pipeline, _ in top_pipelines:
                        if pipeline.id not in self.tuned_pipelines:
                            tune.append(pipeline.id)

                    if len(tune) > 0:
                        # Found some pipelines to tune, do that
                        logger.warning("Found %d pipelines to tune", len(tune))
                        timeout_tuning = self.expected_search_end - time.time()
                        for pipeline_id in tune:
                            logger.info("    %s", pipeline_id)
                            self._ta2._run_queue.put(TuneHyperparamsJob(self._ta2, self, pipeline_id, self.problem, timeout_tuning))
                            self.pipelines_tuning.add(pipeline_id)
                        return
                    logger.info("Found no pipeline to tune")
                else:
                    logger.info("No tuning requested")

                # Session is done (but new pipelines might be added later)
                self.working = False
                self.notify('done_searching')

                logger.warning("Search done")
                if self.metrics:
                    metric = self.metrics[0]['metric']
                    top_pipelines = self.get_top_pipelines(db, metric)
                    logger.warning("Found %d pipelines", len(top_pipelines))

                    for i, (pipeline, score) in enumerate(top_pipelines):
                        created = pipeline.created_date - self.start
                        logger.info("    %d) %s %s=%s origin=%s time=%.2fs",
                                    i + 1, pipeline.id, metric, score,
                                    pipeline.origin, created.total_seconds())
            finally:
                db.close()

    def write_searched_pipeline(self, pipeline_id):
        if not self._searched_pipelines_dir:
            logger.info("Not writing log file")
            return

        db = self.DBSession()
        try:
            # Get pipeline
            pipeline = db.query(database.Pipeline).get(pipeline_id)

            logger.warning("Writing searched_pipeline JSON for pipeline %s "
                           "origin=%s",
                           pipeline_id, pipeline.origin)

            filename = os.path.join(self._searched_pipelines_dir,
                                    '%s.json' % pipeline_id)
            obj = to_d3m_json(pipeline)
            with open(filename, 'w') as fp:
                json.dump(obj, fp, indent=2)
        except Exception:
            logger.exception("Error writing searched_pipeline for %s",
                             pipeline_id)
        finally:
            db.close()

    def write_scored_pipeline(self, pipeline_id):
        if not self._scored_pipelines_dir:
            logger.info("Not writing log file")
            return

        db = self.DBSession()
        try:
            # Get pipeline
            pipeline = db.query(database.Pipeline).get(pipeline_id)

            logger.warning("Writing scored_pipeline JSON for pipeline %s "
                           "origin=%s",
                           pipeline_id, pipeline.origin)

            filename = os.path.join(self._scored_pipelines_dir,
                                    '%s.json' % pipeline_id)
            obj = to_d3m_json(pipeline)
            with open(filename, 'w') as fp:
                json.dump(obj, fp, indent=2)
        except Exception:
            logger.exception("Error writing scored_pipeline for %s",
                             pipeline_id)
        finally:
            db.close()

    def write_exported_pipeline(self, pipeline_id, rank=None):
        metric = self.metrics[0]['metric'].name

        db = self.DBSession()
        try:
            # Get pipeline
            pipeline = db.query(database.Pipeline).get(pipeline_id)

            if rank is None:
                # Find most recent evaluation
                evaluation_id = (
                    select([database.Evaluation.id])
                    .where(database.Evaluation.pipeline_id == pipeline_id)
                    .order_by(database.Evaluation.date.desc())
                ).as_scalar()
                # Get score from that evaluation
                score = db.query(
                    select([func.avg(database.EvaluationScore.value)])
                    .where(
                        database.EvaluationScore.evaluation_id ==
                        evaluation_id
                    )
                    .where(database.EvaluationScore.metric == metric)
                    .as_scalar()
                )
                if score is None:
                    rank = 1000.0
                    logger.error("Writing pipeline JSON for pipeline %s, but "
                                 "it is not scored for %s. Rank set to %s. "
                                 "origin=%s",
                                 pipeline_id, metric, rank, pipeline.origin)
                else:
                    logger.warning("Writing pipeline JSON for pipeline %s "
                                   "%s=%s origin=%s",
                                   pipeline_id, metric, score.value,
                                   pipeline.origin)
                    rank = 1.0 - self.metrics[0]['metric'].normalize(score.value)
            else:
                logger.warning("Writing pipeline JSON for pipeline %s with "
                               "provided rank %s. origin=%s",
                               pipeline_id, rank, pipeline.origin)

            obj = to_d3m_json(pipeline)

            with open(os.path.join(self._ranked_pipelines_dir, '%s.json' % pipeline_id), 'w') as fout:
                json.dump(obj, fout, indent=2)
            with open(os.path.join(self._ranked_pipelines_dir, '%s.rank' % pipeline_id), 'w') as fout:
                fout.write(str(rank))

        finally:
            db.close()

    def close(self):
        self._ta2.remove_observer(self._observer)
        self._observer = None
        self.stop_requested = True
        self.notify('finish_session')


class Job(object):
    def __init__(self):
        self.msg = None

    def start(self, **kwargs):
        raise NotImplementedError

    def check(self):
        if self.msg is not None:
            try:
                while True:
                    self.message(*self.msg.recv(0))
            except Empty:
                pass

        return self.poll()

    def poll(self):
        raise NotImplementedError

    def message(self, *args):
        raise NotImplementedError


class ScoreJob(Job):

    def __init__(self, ta2, pipeline_id, dataset_uri, metrics, problem, scoring_config, timeout_run, report_rank=False,
                 sample_dataset_uri=None):
        Job.__init__(self)
        self.ta2 = ta2
        self.pipeline_id = pipeline_id
        self.dataset_uri = dataset_uri
        self.sample_dataset_uri = sample_dataset_uri
        self.metrics = metrics
        self.problem = problem
        self.scoring_config = scoring_config
        self.timeout_run = timeout_run
        self.expected_end = time.time() + timeout_run
        self.report_rank = report_rank

    def start(self, db_filename, **kwargs):
        self.msg = Receiver()
        self.proc = run_process('alphad3m.pipeline_operations.pipeline_score.score', 'score', self.msg,
                                pipeline_id=self.pipeline_id,
                                dataset_uri=self.dataset_uri,
                                sample_dataset_uri=self.sample_dataset_uri,
                                metrics=self.metrics,
                                problem=self.problem,
                                scoring_config=self.scoring_config,
                                report_rank=self.report_rank,
                                db_filename=db_filename)

        self.ta2.notify('scoring_start',
                        pipeline_id=self.pipeline_id,
                        job_id=id(self))

    def poll(self):
        timeout_reached = False
        if time.time() > self.expected_end:
            timeout_reached = True
            logger.error('Reached timeout (%d seconds) to score a pipeline' % self.timeout_run)
            self.proc.terminate()
            try:
                self.proc.wait(30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()

        if self.proc.poll() is None:
            return False

        error_msg = read_streams(self.proc)
        if timeout_reached: error_msg = 'Reached timeout (%d seconds) to score a pipeline' % self.timeout_run
        log = logger.info if self.proc.returncode == 0 else logger.error
        log('Pipeline scoring process done, returned %d (pipeline: %s)', self.proc.returncode, self.pipeline_id)

        if self.proc.returncode == 0:
            self.ta2.notify('scoring_success',
                            pipeline_id=self.pipeline_id,
                            job_id=id(self))
        else:
            self.ta2.notify('scoring_error',
                            pipeline_id=self.pipeline_id,
                            job_id=id(self),
                            error_msg=error_msg)
        return True


class TrainJob(Job):
    def __init__(self, ta2, pipeline_id, dataset, problem, steps_to_expose):
        Job.__init__(self)
        self.ta2 = ta2
        self.pipeline_id = pipeline_id
        self.dataset = dataset
        self.problem = problem
        self.steps_to_expose = steps_to_expose

    def start(self, db_filename, **kwargs):
        logger.info('Training pipeline for %s', self.pipeline_id)
        self.msg = Receiver()
        self.proc = run_process('alphad3m.pipeline_operations.pipeline_train.train', 'train', self.msg,
                                pipeline_id=self.pipeline_id,
                                dataset=self.dataset,
                                problem=self.problem,
                                storage_dir=self.ta2.runtime_folder,
                                steps_to_expose=self.steps_to_expose,
                                db_filename=db_filename)
        self.ta2.notify('training_start',
                        pipeline_id=self.pipeline_id,
                        job_id=id(self))

    def poll(self):
        if self.proc.poll() is None:
            return False

        error_msg = read_streams(self.proc)
        log = logger.info if self.proc.returncode == 0 else logger.error
        log('Pipeline training process done, returned %d (pipeline: %s)', self.proc.returncode, self.pipeline_id)

        if self.proc.returncode == 0:
            steps_to_expose = []
            for step_id in self.steps_to_expose:
                if exists(join(self.ta2.runtime_folder, 'fit_%s_%s.csv' % (self.pipeline_id, step_id))):
                    steps_to_expose.append(step_id)
            self.ta2.notify('training_success',
                            pipeline_id=self.pipeline_id,
                            storage_dir=self.ta2.runtime_folder,
                            steps_to_expose=steps_to_expose,
                            job_id=id(self))
        else:
            self.ta2.notify('training_error',
                            pipeline_id=self.pipeline_id,
                            job_id=id(self),
                            error_msg=error_msg)
        return True


class TestJob(Job):
    def __init__(self, ta2, pipeline_id, dataset, steps_to_expose):
        Job.__init__(self)
        self.ta2 = ta2
        self.pipeline_id = pipeline_id
        self.dataset = dataset
        self.steps_to_expose = steps_to_expose

    def start(self, db_filename, **kwargs):
        logger.info("Testing pipeline for %s", self.pipeline_id)
        self.msg = Receiver()
        self.proc = run_process('alphad3m.pipeline_operations.pipeline_test.test', 'test', self.msg,
                                pipeline_id=self.pipeline_id,
                                dataset=self.dataset,
                                storage_dir=self.ta2.runtime_folder,
                                steps_to_expose=self.steps_to_expose,
                                db_filename=db_filename)
        self.ta2.notify('testing_start',
                        pipeline_id=self.pipeline_id,
                        job_id=id(self))

    def poll(self):
        if self.proc.poll() is None:
            return False

        error_msg = read_streams(self.proc)
        log = logger.info if self.proc.returncode == 0 else logger.error
        log('Pipeline testing process done, returned %d (pipeline: %s)', self.proc.returncode, self.pipeline_id)

        if self.proc.returncode == 0:
            steps_to_expose = []
            for step_id in self.steps_to_expose:
                if exists(join(self.ta2.runtime_folder, 'produce_%s_%s.csv' % (self.pipeline_id, step_id))):
                    steps_to_expose.append(step_id)
            self.ta2.notify('testing_success',
                            pipeline_id=self.pipeline_id,
                            storage_dir=self.ta2.runtime_folder,
                            steps_to_expose=steps_to_expose,
                            job_id=id(self))
        else:
            self.ta2.notify('testing_error',
                            pipeline_id=self.pipeline_id,
                            job_id=id(self),
                            error_msg=error_msg)
        return True


class TuneHyperparamsJob(Job):
    def __init__(self, ta2, session, pipeline_id, problem, timeout_tuning):
        Job.__init__(self)
        self.ta2 = ta2
        self.session = session
        self.pipeline_id = pipeline_id
        self.problem = problem
        self.timeout_tuning = timeout_tuning
        self.expected_end = time.time() + timeout_tuning

    def start(self, db_filename, **kwargs):
        logger.info("Running tuning for %s (session %s has %d pipelines left to tune)",
                    self.pipeline_id, self.session.id,
                    len(self.session.pipelines_tuning))

        self.msg = Receiver()

        self.proc = run_process('alphad3m.pipeline_operations.pipeline_tune.tune',
                                'tune', self.msg,
                                pipeline_id=self.pipeline_id,
                                storage_dir=self.ta2.internal_folder,
                                dataset_uri=self.session.dataset_uri,
                                sample_dataset_uri=self.session.sample_dataset_uri,
                                metrics=self.session.metrics,
                                problem=self.problem,
                                scoring_config=get_internal_scoring_config(self.problem['problem']['task_keywords']),
                                report_rank=self.session.report_rank,
                                timeout_tuning=self.timeout_tuning,
                                db_filename=db_filename)

        self.session.notify('tuning_start',
                            pipeline_id=self.pipeline_id,
                            job_id=id(self))

    def poll(self):
        if time.time() > self.expected_end:
            logger.error('Reached timeout (%d seconds) to tune a pipeline' % self.timeout_tuning)
            self.proc.terminate()
            try:
                self.proc.wait(30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()

        if self.proc.poll() is None:
            return False

        error_msg = read_streams(self.proc)
        log = logger.info if self.proc.returncode == 0 else logger.error
        log("Pipeline tuning process done, returned %d (pipeline: %s)",
            self.proc.returncode, self.pipeline_id)

        if self.proc.returncode == 0:
            logger.info("New pipeline: %s)", self.tuned_pipeline_id)
            self.session.notify('tuning_success',
                                old_pipeline_id=self.pipeline_id,
                                new_pipeline_id=self.tuned_pipeline_id,
                                job_id=id(self))
            self.session.notify('scoring_success',
                                pipeline_id=self.tuned_pipeline_id,
                                job_id=id(self))
            self.session.pipeline_tuning_done(self.pipeline_id,
                                              self.tuned_pipeline_id)
        else:
            self.session.notify('tuning_error',
                                pipeline_id=self.pipeline_id,
                                job_id=id(self),
                                error_msg=error_msg)
            self.session.pipeline_tuning_done(self.pipeline_id)
        return True

    def message(self, msg, arg):
        if msg == 'progress':
            # TODO: Report progress
            logger.info("Tuning pipeline %s: %.0f%%",
                        self.pipeline_id, arg * 100)
        elif msg == 'tuned_pipeline_id':
            self.tuned_pipeline_id = arg
        else:
            logger.error("Unexpected message from tuning process %s",
                         msg)


class ThreadPoolExecutor(futures.ThreadPoolExecutor):
    def submit(self, fn, *args, **kwargs):
        def wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Exception:
                logger.exception("Exception in worker thread")
                raise
        return futures.ThreadPoolExecutor.submit(self, wrapper,
                                                 *args, **kwargs)
