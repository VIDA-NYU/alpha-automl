"""The SQLAlchemy models we use to persist data.
"""

import enum
import functools
import logging
from sqlalchemy import Column, ForeignKey, create_engine, func, not_, select, \
    inspect
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import column_property, relationship, sessionmaker
from sqlalchemy.sql import functions
from sqlalchemy.types import Binary, Boolean, DateTime, Enum, Float, Integer, \
    String

from alphad3m.schema.sql_uuid import UUID, UuidMixin


logger = logging.getLogger(__name__)


Base = declarative_base()


class Pipeline(UuidMixin, Base):
    __tablename__ = 'pipelines'

    origin = Column(String, nullable=False)
    dataset = Column(String, nullable=False)
    created_date = Column(DateTime, nullable=False,
                          server_default=functions.now())
    task = Column(String, nullable=True)
    parameters = relationship('PipelineParameter', lazy='joined')
    modules = relationship('PipelineModule')
    connections = relationship('PipelineConnection')
    runs = relationship('Run')

    def __eq__(self, other):
        return type(other) is Pipeline and other.id == self.id

    def __repr__(self):
        return '<Pipeline %r>' % self.id

    __str__ = __repr__


class PipelineModule(UuidMixin, Base):
    __tablename__ = 'pipeline_modules'

    pipeline_id = Column(UUID, ForeignKey('pipelines.id'), nullable=False)
    pipeline = relationship(Pipeline)
    package = Column(String, nullable=False)
    version = Column(String, nullable=False)
    name = Column(String, nullable=False)
    # connections_from = relationship('PipelineConnection')
    # connections_to = relationship('PipelineConnection')


class PipelineConnection(Base):
    __tablename__ = 'pipeline_connections'

    pipeline_id = Column(UUID, ForeignKey('pipelines.id'), nullable=False)
    pipeline = relationship(Pipeline)

    from_module_id = Column(UUID, ForeignKey('pipeline_modules.id'),
                            primary_key=True)
    from_module = relationship('PipelineModule',
                               foreign_keys=[from_module_id],
                               backref='connections_from')
    from_output_name = Column(String, primary_key=True)

    to_module_id = Column(UUID, ForeignKey('pipeline_modules.id'),
                          primary_key=True)
    to_module = relationship('PipelineModule',
                             foreign_keys=[to_module_id],
                             backref='connections_to')
    to_input_name = Column(String, primary_key=True)


class PipelineParameter(Base):
    __tablename__ = 'pipeline_parameters'

    pipeline_id = Column(UUID, ForeignKey('pipelines.id'), nullable=False)
    pipeline = relationship(Pipeline)

    module_id = Column(UUID, ForeignKey('pipeline_modules.id'),
                       primary_key=True)
    module = relationship('PipelineModule')
    name = Column(String, primary_key=True)
    value = Column(String, nullable=True)


class Evaluation(UuidMixin, Base):
    __tablename__ = 'evaluations'

    pipeline_id = Column(UUID, ForeignKey('pipelines.id'), nullable=False)
    pipeline = relationship('Pipeline')
    date = Column(DateTime, nullable=False,
                  server_default=functions.now())
    scores = relationship('EvaluationScore', lazy='joined')


class EvaluationScore(Base):
    __tablename__ = 'evaluation_scores'

    evaluation_id = Column(UUID, ForeignKey('evaluations.id'),
                                 primary_key=True)
    evaluation = relationship('Evaluation')
    fold = Column(Integer, primary_key=True, nullable=True)
    metric = Column(String, primary_key=True)
    value = Column(Float, nullable=False)


class RunType(enum.Enum):
    TRAIN = 1
    TEST = 2


class Run(UuidMixin, Base):
    __tablename__ = 'runs'

    pipeline_id = Column(UUID, ForeignKey('pipelines.id'), nullable=False)
    pipeline = relationship('Pipeline')
    date = Column(DateTime, nullable=False,
                  server_default=functions.now())
    reason = Column(String, nullable=False)
    type = Column(Enum(RunType), nullable=False)
    special = Column(Boolean, nullable=False, default=False)
    inputs = relationship('Input')
    outputs = relationship('Output')


class Input(Base):
    __tablename__ = 'inputs'

    run_id = Column(UUID, ForeignKey('runs.id'), primary_key=True)
    run = relationship('Run')
    module_id = Column(UUID, ForeignKey('pipeline_modules.id'),
                       primary_key=True)
    module = relationship('PipelineModule')
    input_name = Column(String, primary_key=True, nullable=True)
    value = Column(Binary, nullable=False)


class Output(Base):
    __tablename__ = 'outputs'

    run_id = Column(UUID, ForeignKey('runs.id'), primary_key=True)
    run = relationship('Run')
    module_id = Column(UUID, ForeignKey('pipeline_modules.id'),
                       primary_key=True)
    module = relationship('PipelineModule')
    output_name = Column(String, primary_key=True, nullable=True)
    value = Column(Binary, nullable=False)


# Trained true iff there's a Run with special=False and type=TRAIN
Pipeline.trained = column_property(
    select(
        [func.count(Run.id)]
    ).where((Run.pipeline_id == Pipeline.id) &
            (not_(Run.special)) &
            (Run.type == RunType.TRAIN))
    .as_scalar() != 0
)


def connect(filename):
    """Connect to the database using an environment variable.
    """
    logger.info("Connecting to SQL database")
    url = 'sqlite:///{0}'.format(filename)
    engine = create_engine(url, echo=False)

    if not engine.dialect.has_table(engine.connect(), 'pipelines'):
        logger.warning("The tables don't seem to exist; creating")
        Base.metadata.create_all(bind=engine)

    return engine, sessionmaker(bind=engine,
                                autocommit=False,
                                autoflush=False)


def with_db(wrapped):
    @functools.wraps(wrapped)
    def wrapper(*args, db_filename=None, **kwargs):
        engine, DBSession = connect(db_filename)
        db = DBSession()
        try:
            return wrapped(*args, **kwargs, db=db)
        finally:
            db.close()
    return wrapper


def with_sessionmaker(wrapped):
    @functools.wraps(wrapped)
    def wrapper(*args, db_filename=None, **kwargs):
        engine, DBSession = connect(db_filename)
        return wrapped(*args, **kwargs, DBSession=DBSession)
    return wrapper


def _duplicate(db, obj, replace={}):
    new_obj = type(obj)()
    mapper = inspect(type(obj))
    for k, col in mapper.columns.items():
        if k in replace:
            v = replace[k]
            if isinstance(v, dict):
                old_v = getattr(obj, k)
                v = v[old_v]
        elif k != 'id':
            v = getattr(obj, k)
        else:
            continue
        setattr(new_obj, k, v)
    db.add(new_obj)
    return new_obj


def _duplicate_list(db, lst, replace={}):
    return [_duplicate(db, obj, replace) for obj in lst]


def duplicate_pipeline(db, pipeline, origin):
    """Duplicates a Pipeline.

    This creates a new Pipeline with new modules/parameters/connections. They
    all get new IDs.
    """
    new_pipeline = Pipeline(
        origin=origin,
        task=pipeline.task,
        dataset=pipeline.dataset,
    )
    db.add(new_pipeline)

    modules = {}
    for module in pipeline.modules:
        new_module = _duplicate(db, module,
                                dict(pipeline_id=new_pipeline.id))
        modules[module.id] = new_module.id

    _duplicate_list(db, pipeline.parameters,
                    dict(pipeline_id=new_pipeline.id,
                         module_id=modules))
    _duplicate_list(db, pipeline.connections,
                    dict(pipeline_id=new_pipeline.id,
                         from_module_id=modules,
                         to_module_id=modules))

    db.flush()
    return new_pipeline
