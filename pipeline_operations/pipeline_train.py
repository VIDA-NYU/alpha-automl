import os
import logging
import pickle
import d3m.runtime
import d3m.metadata.base
from os.path import join
from sqlalchemy.orm import joinedload
from d3m.container import Dataset, DataFrame
from d3m.metadata import base as metadata_base
from alphad3m.schema import database, convert


logger = logging.getLogger(__name__)


@database.with_db
def train(pipeline_id, dataset, problem, storage_dir, steps_to_expose, msg_queue, db):
    # Get pipeline from database
    pipeline = (
        db.query(database.Pipeline)
            .filter(database.Pipeline.id == pipeline_id)
            .options(joinedload(database.Pipeline.modules),
                     joinedload(database.Pipeline.connections))
    ).one()

    logger.info('About to train pipeline, id=%s, dataset=%r',
                pipeline_id, dataset)

    # Load data
    dataset = Dataset.load(dataset)
    logger.info('Loaded dataset')

    # Training step - fit pipeline on training data
    logger.info('Running training')

    d3m_pipeline = d3m.metadata.pipeline.Pipeline.from_json_structure(
        convert.to_d3m_json(pipeline),
    )

    expose_outputs = True if len(steps_to_expose) > 0 else False

    fitted_pipeline, predictions, results = d3m.runtime.fit(d3m_pipeline, [dataset], problem_description=problem,
                                                            context=metadata_base.Context.TESTING,
                                                            volumes_dir=os.environ.get('D3MSTATICDIR', None),
                                                            random_seed=0,
                                                            expose_produced_outputs=expose_outputs)

    results.check_success()

    logger.info('Storing fit results at %s', storage_dir)
    for step_id in results.values:
        if step_id in steps_to_expose and isinstance(results.values[step_id], DataFrame):
            results.values[step_id].to_csv(join(storage_dir, 'fit_%s_%s.csv' % (pipeline_id, step_id)))

    with open(join(storage_dir, 'fitted_solution_%s.pkl' % pipeline_id), 'wb') as fout:
        pickle.dump(fitted_pipeline, fout)
