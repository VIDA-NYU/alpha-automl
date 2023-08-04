import logging
import time
import multiprocessing
from alpha_automl.data_profiler import profile_data
from alpha_automl.scorer import make_splitter, score_pipeline
from alpha_automl.utils import sample_dataset, is_equal_splitting
from alpha_automl.pipeline_synthesis.setup_search import search_pipelines as search_pipelines_proc


USE_AUTOMATIC_GRAMMAR = False
PRIORITIZE_PRIMITIVES = False
EXCLUDE_PRIMITIVES = []
INCLUDE_PRIMITIVES = []
NEW_PRIMITIVES = {}
SPLITTING_STRATEGY = 'holdout'
SAMPLE_SIZE = 2000

logger = logging.getLogger(__name__)


class AutoMLManager():

    def __init__(self, output_folder, time_bound, time_bound_run, task, verbose):
        self.output_folder = output_folder
        self.time_bound = time_bound * 60
        self.time_bound_run = time_bound_run * 60
        self.task = task
        self.X = None
        self.y = None
        self.scoring = None
        self.splitting_strategy = None
        self.verbose = verbose

    def search_pipelines(self, X, y, scoring, splitting_strategy, automl_hyperparams=None):
        if automl_hyperparams is None:
            automl_hyperparams = {}

        self.X = X
        self.y = y
        self.scoring = scoring
        self.splitting_strategy = splitting_strategy

        for pipeline_data in self._search_pipelines(automl_hyperparams):
            yield pipeline_data

    def _search_pipelines(self, automl_hyperparams):
        search_start_time = time.time()
        automl_hyperparams = self.check_automl_hyperparams(automl_hyperparams)

        metadata = profile_data(self.X)
        X, y, is_sample = sample_dataset(self.X, self.y, SAMPLE_SIZE, self.task)
        internal_splitting_strategy = make_splitter(SPLITTING_STRATEGY)
        need_rescoring = True

        if not is_sample and is_equal_splitting(internal_splitting_strategy, self.splitting_strategy):
            need_rescoring = False

        queue = multiprocessing.Queue()
        search_process = multiprocessing.Process(target=search_pipelines_proc,
                                                 args=(X, y, self.scoring, internal_splitting_strategy, self.task,
                                                       self.time_bound, automl_hyperparams, metadata,
                                                       self.output_folder, self.verbose, queue
                                                       )
                                                 )

        search_process.start()
        found_pipelines = 0

        while True:
            try:
                result = queue.get(timeout=30)
            except Exception:
                logger.info('Queue Timeout after 30s')
                result = None

            if result == 'DONE':
                search_process.terminate()
                search_process.join(30)
                logger.info(f'Found {found_pipelines} pipelines')
                logger.info('Search done')
                break
            elif result is not None:
                pipeline = result
                score = pipeline.get_score()
                logger.info('Found new pipeline')
                yield {'pipeline': pipeline, 'message': 'FOUND'}

                if need_rescoring:
                    score, start_time, end_time = score_pipeline(pipeline.get_pipeline(), self.X, self.y, self.scoring,
                                                                self.splitting_strategy)
                    pipeline.set_score(score)
                    pipeline.set_start_time(start_time)
                    pipeline.set_end_time(end_time)

                if score is not None:
                    logger.info(f'Pipeline scored successfully, score={score}')
                    found_pipelines += 1
                    yield {'pipeline': pipeline, 'message': 'SCORED'}

            if time.time() > search_start_time + self.time_bound:
                search_process.terminate()
                search_process.join(30)
                logger.info(f'Found {found_pipelines} pipelines')
                logger.info('Reached search timeout')
                break

    def check_automl_hyperparams(self, automl_hyperparams):
        if 'use_automatic_grammar' not in automl_hyperparams:
            automl_hyperparams['use_automatic_grammar'] = USE_AUTOMATIC_GRAMMAR

        if 'prioritize_primitives' not in automl_hyperparams:
            automl_hyperparams['prioritize_primitives'] = PRIORITIZE_PRIMITIVES

        if 'include_primitives' not in automl_hyperparams or automl_hyperparams['include_primitives'] is None:
            automl_hyperparams['include_primitives'] = INCLUDE_PRIMITIVES

        if 'exclude_primitives' not in automl_hyperparams or automl_hyperparams['exclude_primitives'] is None:
            automl_hyperparams['exclude_primitives'] = EXCLUDE_PRIMITIVES

        if 'new_primitives' not in automl_hyperparams or automl_hyperparams['new_primitives'] is None:
            automl_hyperparams['new_primitives'] = NEW_PRIMITIVES

        return automl_hyperparams
