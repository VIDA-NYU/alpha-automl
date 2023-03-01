import logging
import time
import multiprocessing
from alpha_automl.pipeline_synthesis.setup_search import search_pipelines as search_pipelines_proc
from alpha_automl.utils import *

USE_AUTOMATIC_GRAMMAR = False
PRIORITIZE_PRIMITIVES = False
EXCLUDE_PRIMITIVES = []
INCLUDE_PRIMITIVES = []
SPLITTING_STRATEGY = 'holdout'
SAMPLE_SIZE = 2000

logger = logging.getLogger(__name__)


class AutoMLManager():

    def __init__(self, output_folder, time_bound, time_bound_run, task):
        self.output_folder = output_folder
        self.time_bound = time_bound * 60
        self.time_bound_run = time_bound_run * 60
        self.task = task
        self.X = None
        self.y = None
        self.scoring = None
        self.splitting_strategy = None

    def search_pipelines(self, X, y, scoring, splitting_strategy, hyperparameters=None):
        if hyperparameters is None:
            hyperparameters = {}

        self.X = X
        self.y = y
        self.scoring = scoring
        self.splitting_strategy = splitting_strategy

        for pipeline_data in self._search_pipelines( hyperparameters):
            yield pipeline_data

    def _search_pipelines(self, hyperparameters):
        start_time = time.time()
        if 'use_automatic_grammar' not in hyperparameters:
            hyperparameters['use_automatic_grammar'] = USE_AUTOMATIC_GRAMMAR

        if 'prioritize_primitives' not in hyperparameters:
            hyperparameters['prioritize_primitives'] = PRIORITIZE_PRIMITIVES

        if 'include_primitives' not in hyperparameters or hyperparameters['include_primitives'] is None:
            hyperparameters['include_primitives'] = INCLUDE_PRIMITIVES

        if 'exclude_primitives' not in hyperparameters or hyperparameters['exclude_primitives'] is None:
            hyperparameters['exclude_primitives'] = EXCLUDE_PRIMITIVES

        X, y, is_sample = sample_dataset(self.X, self.y, SAMPLE_SIZE)
        splitting_strategy = make_splitter(SPLITTING_STRATEGY, y)

        queue = multiprocessing.Queue()
        search_process = multiprocessing.Process(target=search_pipelines_proc,
                                                 args=(X, y, self.scoring, splitting_strategy, self.task,
                                                       self.time_bound, hyperparameters, self.output_folder, queue
                                                       )
                                                 )
        search_process.start()
        found_pipelines = 0

        while True:
            result = queue.get()

            if result == 'DONE':
                search_process.terminate()
                search_process.join(30)
                logger.info(f'Found {found_pipelines} pipelines')
                logger.info('Search done')
                break

            pipeline_object = result[0]
            pipeline_score = result[1]
            logger.info('Found new pipeline')
            yield {'pipeline_object': pipeline_object, 'pipeline_score': pipeline_score, 'message': 'FOUND'}

            if is_sample:
                pipeline_score = score_pipeline(pipeline_object, self.X, self.y, self.scoring, self.splitting_strategy)

            if pipeline_score is not None:
                logger.info(f'Pipeline scored successfully, score={pipeline_score}')
                found_pipelines += 1
                yield {'pipeline_object': pipeline_object, 'pipeline_score': pipeline_score, 'message': 'SCORED'}

            if time.time() > start_time + self.time_bound:
                search_process.terminate()
                search_process.join(30)
                logger.info(f'Found {found_pipelines} pipelines')
                logger.info('Reached search timeout')
                break

    def search_pipelines_fake(self, X, y, scoring, splitting_strategy):
        from alpha_automl.utils import score_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.svm import SVC

        pipelines = []
        pipeline1 = Pipeline(steps=[("preprocessor", StandardScaler()), ("classifier", LogisticRegression())])

        score1 = score_pipeline(pipeline1, X, y, scoring, splitting_strategy)
        pipelines.append({'pipeline_object': pipeline1, 'pipeline_score': score1})

        pipeline2 = Pipeline(steps=[("preprocessor", StandardScaler()), ("classifier", SVC())])
        score2 = score_pipeline(pipeline2, X, y, scoring, splitting_strategy)
        pipelines.append({'pipeline_object': pipeline2, 'pipeline_score': score2})

        for pipeline in pipelines:
            yield pipeline
