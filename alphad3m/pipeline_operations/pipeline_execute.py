import logging
import d3m.runtime
import d3m.metadata.base
from sqlalchemy.orm import joinedload
from d3m.container import Dataset
from d3m.metadata import base as metadata_base
from alphad3m.schema import database, convert
from multiprocessing import Manager, Process


logger = logging.getLogger(__name__)

@database.with_db
def execute(pipeline_id, dataset, problem, results_path, msg_queue, db):
    # Get pipeline from database

    pipeline = (
        db.query(database.Pipeline)
            .filter(database.Pipeline.id == pipeline_id)
            .options(joinedload(database.Pipeline.modules),
                     joinedload(database.Pipeline.connections))
    ).one()

    logger.info('About to execute pipeline, id=%s, dataset=%r',
                pipeline_id, dataset)

    # Load data
    dataset = Dataset.load(dataset)
    logger.info('Loaded dataset')

    json_pipeline = convert.to_d3m_json(pipeline)
    logger.info('Pipeline to be executed:\n%s',
                '\n'.join([x['primitive']['python_path'] for x in json_pipeline['steps']]))

    d3m_pipeline = d3m.metadata.pipeline.Pipeline.from_json_structure(json_pipeline, )

    runtime = d3m.runtime.Runtime(pipeline=d3m_pipeline, problem_description=problem,
                                  context=metadata_base.Context.TESTING)

    manager = Manager()
    return_dict = manager.dict()
    p = Process(target=worker, args=(runtime, dataset, return_dict))
    p.start()
    p.join(180)  # Maximum 3 minutes
    fit_results = return_dict['fit_results']
    fit_results.check_success()

    if results_path is not None:
        logger.info('Storing fit results at %s', results_path)
        fit_results.values['outputs.0'].to_csv(results_path)
    else:
        logger.info('NOT storing fit results')

    return fit_results.values


def worker(runtime, dataset, return_dict):
    return_dict['fit_results'] = runtime.fit(inputs=[dataset])
