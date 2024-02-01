import os
import logging
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
                              output_folder=output_path, verbose=logging.DEBUG)

    train_dataset = pd.read_csv(train_dataset_path)
    test_dataset = pd.read_csv(test_dataset_path)
    X_train = train_dataset.drop(columns=[target_name])
    y_train = train_dataset[[target_name]]
    X_test = test_dataset.drop(columns=[target_name])
    y_test = test_dataset[[target_name]]

    with Timer() as train_time:
        automl.fit(X_train, y_train)

    automl.plot_leaderboard(use_print=True)
    best_pipeline = automl.get_pipeline()
    classes = automl.label_encoder.inverse_transform(best_pipeline.classes_)

    with Timer() as test_time:
        log.info('Testing pipeline')
        predictions = best_pipeline.predict(X_test)
        predictions = automl.label_encoder.inverse_transform(predictions)

    try:
        probabilities = pd.DataFrame(best_pipeline.predict_proba(X_test), columns=classes)
    except:  # Some primitives don't implement predict_proba method
        log.warning(f'The method predict_proba is not supported, using fallback')
        probabilities = pd.DataFrame(0, index=range(len(predictions)), columns=classes)  # Dataframe of zeros
        for index, prediction in enumerate(predictions):
            probabilities.at[index, prediction] = 1.0

    return result(
                  output_file=config.output_predictions_file,
                  predictions=predictions,
                  truth=y_test,
                  probabilities=probabilities,
                  probabilities_labels=probabilities.columns.values.astype(str).tolist(),
                  training_duration=train_time.duration,
                  predict_duration=test_time.duration,
                  target_is_encoded=False)


if __name__ == '__main__':
    call_run(run)
