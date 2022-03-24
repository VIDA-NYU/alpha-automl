"""Various utilities that are not specific to D3M.
"""

import os
import contextlib
import logging
import json
from queue import Empty, Queue
import threading
import subprocess
from d3m.metadata.problem import TaskKeyword
from sklearn.model_selection import train_test_split
from alphad3m.primitive_loader import load_primitives_list

SAMPLE_SIZE = 2000
RANDOM_SEED = 0

logger = logging.getLogger(__name__)


class Observable(object):
    """Allow adding callbacks on an object, to be called on notifications.
    """
    def __init__(self):
        self.__observers = {}
        self.__next_key = 0
        self.lock = threading.RLock()

    def add_observer(self, observer):
        with self.lock:
            key = self.__next_key
            self.__next_key += 1
            self.__observers[key] = observer
            return key

    def remove_observer(self, key):
        with self.lock:
            del self.__observers[key]

    @contextlib.contextmanager
    def with_observer(self, observer):
        key = self.add_observer(observer)
        try:
            yield
        finally:
            self.remove_observer(key)

    @contextlib.contextmanager
    def with_observer_queue(self):
        queue = Queue()
        with self.with_observer(lambda e, **kw: queue.put((e, kw))):
            yield queue

    def notify(self, event, **kwargs):
        with self.lock:
            for observer in self.__observers.values():
                try:
                    observer(event, **kwargs)
                except Exception:
                    logging.exception("Error in observer")


class _PQ_Reader(object):
    def __init__(self, pq):
        self._pq = pq
        self._pos = 0
        self.finished = False

    def get(self, timeout=None):
        if self.finished:
            return None
        with self._pq.lock:
            # There are unread items
            if (len(self._pq.list) > self._pos or
                    # Or get woken up
                    self._pq.change.wait(timeout)):
                self._pos += 1
                item = self._pq.list[self._pos - 1]
                if item is None:
                    self.finished = True
                return item
            # Timeout
            else:
                raise Empty


class PersistentQueue(object):
    """A Queue object that will always yield items inserted from the start.
    """
    def __init__(self):
        self.list = []
        self.lock = threading.RLock()
        self.change = threading.Condition(self.lock)

    def put(self, item):
        """Put an item in the queue, waking up readers.
        """
        if item is None:
            raise TypeError("Can't put None in PersistentQueue")
        with self.lock:
            self.list.append(item)
            self.change.notify_all()

    def close(self):
        """End the queue, readers will terminate.
        """
        with self.lock:
            self.list.append(None)
            self.change.notify_all()

    def read(self):
        """Get an iterator on all items from the queue.
        """
        reader = self.reader()
        while True:
            item = reader.get()
            if item is None:
                return
            yield item

    def reader(self):
        """Get a reader object you can use to read with a timeout.
        """
        return _PQ_Reader(self)


class ProgressStatus(object):
    def __init__(self, current, total=1.0):
        self.current = max(0.0, min(current, total))
        if total <= 0.0:
            self.total = 1.0
        else:
            self.total = total

    @property
    def ratio(self):
        return self.current / self.total

    @property
    def percentage(self):
        return '%d%%' % int(self.current / self.total)


def is_collection(dataset_path):
    with open(dataset_path) as fin:
        dataset_doc = json.load(fin)
        for data_resource in dataset_doc['dataResources']:
            if data_resource.get('isCollection', False):
                return True

    return False


def get_collection_type(dataset_path):
    with open(dataset_path) as fin:
        dataset_doc = json.load(fin)
        for data_resource in dataset_doc['dataResources']:
            if data_resource.get('isCollection', False):
                return data_resource['resType']

    return None


def need_denormalize(dataset_path):
    with open(dataset_path) as fin:
        dataset_doc = json.load(fin)
        if len(dataset_doc['dataResources']) > 1:
            return True
        return False


def get_dataset_sample(dataset, problem, dataset_sample_path=None):
    task_keywords = problem['problem']['task_keywords']
    sample_size = SAMPLE_SIZE

    if any(tk in [TaskKeyword.OBJECT_DETECTION, TaskKeyword.FORECASTING] for tk in task_keywords):
        logger.info('Not doing sampling for task %s', '_'.join([x.name for x in task_keywords]))
        return dataset

    try:
        target_name = problem['inputs'][0]['targets'][0]['column_name']
        for res_id in dataset:
            if ('https://metadata.datadrivendiscovery.org/types/DatasetEntryPoint'
                    in dataset.metadata.query([res_id])['semantic_types']):
                break
        else:
            res_id = next(iter(dataset))

        if any(tk in [TaskKeyword.VIDEO, TaskKeyword.IMAGE, TaskKeyword.AUDIO] for tk in task_keywords):
            sample_size = 100
        original_size = len(dataset[res_id])

        if hasattr(dataset[res_id], 'columns') and len(dataset[res_id]) > sample_size:
            labels = dataset[res_id].get(target_name)
            ratio = sample_size / original_size
            try:
                x_train, x_test, y_train, y_test = train_test_split(dataset[res_id], labels, random_state=RANDOM_SEED,
                                                                    test_size=ratio, stratify=labels)
            except:
                # Not using stratified sampling when the minority class has few instances, not enough for all the folds
                x_train, x_test, y_train, y_test = train_test_split(dataset[res_id], labels, random_state=RANDOM_SEED,
                                                                    test_size=ratio)
            dataset[res_id] = x_test
            logger.info('Sampling down data from %d to %d', original_size, len(dataset[res_id]))
            if dataset_sample_path:
                dataset.save(dataset_sample_path + 'datasetDoc.json')
                dataset_sample_uri = dataset_sample_path + 'datasetDoc.json'
                return dataset_sample_uri

        else:
            logger.info('Not doing sampling for small dataset (size = %d)', original_size)
    except:
        logger.error('Error sampling in dataset %s')

    return dataset


def get_internal_scoring_config(task_keywords):
    scoring_config = {'shuffle': 'true', 'method': 'K_FOLD',  'number_of_folds': '2',
                      'stratified': 'true' if TaskKeyword.CLASSIFICATION in task_keywords else 'false'
                      }

    return scoring_config


def create_outputfolders(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def read_streams(process):
    error_msg = None
    try:
        _, stderr = process.communicate(timeout=5)
        error_msg = stderr.decode()
    except subprocess.TimeoutExpired:
        logger.error('Timeout error reading child process logs')

    return error_msg


def load_primitives_types():
    primitives_types = {}
    primitives = load_primitives_list()

    for primitive in primitives:
        primitives_types[primitive['python_path']] = primitive['type']

    return primitives_types
