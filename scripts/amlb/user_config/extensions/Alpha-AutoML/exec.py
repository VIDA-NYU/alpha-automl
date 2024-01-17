import os
import logging
import numpy as np
import pandas as pd
import tempfile as tmp
from alpha_automl import AutoMLClassifier
from frameworks.shared.callee import call_run, result
from frameworks.shared.utils import Timer


os.environ['JOBLIB_TEMP_FOLDER'] = tmp.gettempdir()
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
log = logging.getLogger(__name__)


def run(dataset, config):
    log.info(f'\n**** Running Alpha-AutoML ****\n')

    metrics_mapping = {
        'acc': 'accuracy_score',
        #'auc': metrics.roc_auc,
        'f1': 'f1_score',
        #'logloss': metrics.log_loss,
        'mae': 'mean_absolute_error',
        'mse': 'mean_squared_error',
        'r2': 'r2_score'
    }

    metric = config.metric

    if metric is None:
        log.warning(f'Performance metric {metric} not supported, defaulting to accuracy')
        metric = 'acc'

    train_dataset_path = dataset.train.path
    test_dataset_path = dataset.test.path
    target_name = dataset.target.name
    output_path = config.output_dir
    time_bound = int(config.max_runtime_seconds/60)

    log.info(f'Received parameters:\n'
             f'train_dataset: {train_dataset_path}\n'
             f'test_dataset: {test_dataset_path}\n'
             f'target_name: {target_name}\n'
             f'time_bound: {time_bound}\n'
             f'metric: {metric}\n'
             )

    automl = AutoMLClassifier(time_bound=time_bound, metric=metrics_mapping[metric], time_bound_run=15,
                              output_folder=output_path, start_mode='spawn', verbose=logging.DEBUG)

    train_dataset = pd.read_csv(train_dataset_path)
    test_dataset = pd.read_csv(test_dataset_path)
    X_train = train_dataset.drop(columns=[target_name])
    y_train = train_dataset[[target_name]]
    X_test = test_dataset.drop(columns=[target_name])
    y_test = test_dataset[[target_name]]

    with Timer() as training:
        automl.fit(X_train, y_train)
        automl.plot_leaderboard(use_print=True)
        predictions = automl.predict(X_test)

    classes = pd.read_csv(train_dataset)[target_name].unique()
    probabilities = pd.DataFrame(0, index=np.arange(len(predictions)), columns=classes)

    return result(dataset=dataset,
                  output_file=config.output_predictions_file,
                  probabilities=probabilities,
                  predictions=predictions,
                  training_duration=training.duration)


if __name__ == '__main__':
    call_run(run)
