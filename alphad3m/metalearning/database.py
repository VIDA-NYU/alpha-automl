import json
import pickle
import logging
from os.path import join, dirname

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

METALEARNINGDB_PATH = join(dirname(__file__), '../resource/metalearningdb.pkl')


def merge_pipeline_files(pipelines_file, pipeline_runs_file, problems_file, n=-1, verbose=False):
    logger.info('Adding pipelines to lookup table...')
    pipelines = {}
    with open(pipelines_file, 'r') as f:
        for line in f:
            pipeline = json.loads(line)
            pipelines[pipeline['digest']] = pipeline

    logger.info('Adding problems to lookup table...')
    problems = {}
    with open(problems_file, 'r') as f:
        for line in f:
            problem = json.loads(line)
            problems[problem['digest']] = problem['problem']
            problems[problem['digest']]['id'] = problem['id']

    logger.info('Merging pipeline information with pipeline_runs_file (this might take a while)...')
    merged = []
    with open(pipeline_runs_file, 'r') as f:
        for line in f:
            if len(merged) == n:
                break
            try:
                run = json.loads(line)
                if run['run']['phase'] != 'PRODUCE':
                    continue
                pipeline = pipelines[run['pipeline']['digest']]
                problem = problems[run['problem']['digest']]
                data = {
                    'pipeline_id': pipeline['id'],
                    'pipeline_digest': pipeline['digest'],
                    'pipeline_source': pipeline['source'],
                    'inputs': pipeline['inputs'],
                    'outputs': pipeline['outputs'],
                    'problem': problem,
                    'start': run['start'],
                    'end': run['end'],
                    'steps': pipeline['steps'],
                    'scores': run['run']['results']['scores']
                }
                merged.append(json.dumps(data))
            except Exception as e:
                if (verbose):
                    logger.error(problem['id'], repr(e))
    logger.info('Done.')

    with open(METALEARNINGDB_PATH, 'wb') as f:
        pickle.dump(merged, f)


def load_metalearningdb():
    all_pipelines = []
    logger.info('Loading pipelines from metalearning database...')
    with open(METALEARNINGDB_PATH, 'rb') as fin:
        # Use pickle instead of json because it was faster in our experiments
        all_pipelines = pickle.load(fin)

    logger.info('Found %d pipelines in metalearning database' % len(all_pipelines))

    return all_pipelines


if __name__ == '__main__':
    pipelines_file = '/Users/rlopez/Downloads/metalearningdb_dump_20200304/pipelines-1583354358.json'
    pipeline_runs_file = '/Users/rlopez/Downloads/metalearningdb_dump_20200304/pipeline_runs-1583354387.json'
    problems_file = '/Users/rlopez/Downloads/metalearningdb_dump_20200304/problems-1583354357.json'
    merge_pipeline_files(pipelines_file, pipeline_runs_file, problems_file)
