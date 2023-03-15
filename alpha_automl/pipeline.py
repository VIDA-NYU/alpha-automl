import logging
from sklearn.compose import ColumnTransformer
from alpha_automl.utils import COLUMN_SELECTOR_ID


logger = logging.getLogger(__name__)


class Pipeline():

    def __init__(self, pipeline_sk, score, start_time, end_time):
        self.pipeline = pipeline_sk
        self.score = score
        self.start_time = start_time
        self.end_time = end_time
        self.summary = None
        self._make_summary()

    def get_pipeline(self):
        return self.pipeline

    def get_score(self):
        return self.score

    def get_start_time(self):
        return self.start_time

    def get_end_time(self):
        return self.end_time

    def get_summary(self):
        return self.summary

    def set_pipeline(self, pipeline):
        self.pipeline = pipeline
        self._make_summary()

    def set_score(self, score):
        self.score = score

    def set_start_time(self, start_time):
        self.start_time = start_time

    def set_end_time(self, end_time):
        self.end_time = end_time

    def _make_summary(self):
        step_names = []
        for step_name, step_object in self.pipeline.named_steps.items():
            step_name = step_name.split('.')[-1]
            step_names.append(step_name)
            if isinstance(step_object, ColumnTransformer):
                for transformer_name, _, _ in step_object.transformers:
                    if transformer_name == COLUMN_SELECTOR_ID:
                        continue  # Don't show column selector
                    step_name = transformer_name.split('-')[0].split('.')[-1]
                    if step_name not in step_names:
                        step_names.append(step_name)

        self.summary = ', '.join(step_names)
