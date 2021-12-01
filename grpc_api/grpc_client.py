import grpc
import json
import logging
import os
import sys
import pickle
import datetime
from urllib import parse
import d3m_automl_rpc.core_pb2 as pb_core
import d3m_automl_rpc.core_pb2_grpc as pb_core_grpc
import d3m_automl_rpc.value_pb2 as pb_value
from d3m_automl_rpc.utils import encode_problem_description, encode_performance_metric, encode_value
from alphad3m.grpc_api.grpc_logger import LoggingStub


logger = logging.getLogger(__name__)


def do_hello(core):
    core.Hello(pb_core.HelloRequest())


def do_listprimitives(core):
    core.ListPrimitives(pb_core.ListPrimitivesRequest())


def do_search(core, problem, dataset_path, time_bound=30.0, pipelines_limit=0, pipeline_template=None):
    version = pb_core.DESCRIPTOR.GetOptions().Extensions[pb_core.protocol_version]
    automl_hyperparameters = {'exclude_primitives': None,
                              'include_primitives': None}
    automl_hyperparams_encoded = {k: encode_value({'type': 'object', 'value': v}, ['RAW'], '/tmp') for k, v in
                                  automl_hyperparameters.items()}
    search = core.SearchSolutions(pb_core.SearchSolutionsRequest(
        user_agent='ta3_stub',
        version=version,
        time_bound_search=time_bound,
        rank_solutions_limit=pipelines_limit,
        allowed_value_types=['CSV_URI'],
        problem=encode_problem_description(problem),
        automl_hyperparameters=automl_hyperparams_encoded,
        template=pipeline_template,
        inputs=[pb_value.Value(
            dataset_uri='file://%s' % dataset_path,
        )],
    ))

    start_time = datetime.datetime.now()
    results = core.GetSearchSolutionsResults(
        pb_core.GetSearchSolutionsResultsRequest(
            search_id=search.search_id,
        )
    )
    solutions = {}
    for result in results:
        if result.solution_id:
            end_time = datetime.datetime.now()
            solutions[result.solution_id] = (
                result.internal_score,
                result.scores,
                str(end_time - start_time)
            )

    return str(search.search_id), solutions


def do_score(core, problem, solutions, dataset_path):
    metrics = []

    for metric in problem['problem']['performance_metrics']:
        metrics.append(encode_performance_metric(metric))

    for solution in solutions:
        try:
            response = core.ScoreSolution(pb_core.ScoreSolutionRequest(
                solution_id=solution,
                inputs=[pb_value.Value(
                    dataset_uri='file://%s' % dataset_path,
                )],
                performance_metrics=metrics,
                users=[],
                configuration=pb_core.ScoringConfiguration(
                    method='K_FOLD',
                    folds=4,
                    train_test_ratio=0.75,
                    shuffle=True,
                    random_seed=0
                ),
            ))
            results = core.GetScoreSolutionResults(
                pb_core.GetScoreSolutionResultsRequest(
                    request_id=response.request_id,
                )
            )
            for _ in results:
                pass
        except Exception:
            logger.exception("Exception during scoring %r", solution)


def do_train(core, solutions, dataset_path):
    fitted = {}
    for solution in solutions:
        try:
            response = core.FitSolution(pb_core.FitSolutionRequest(
                solution_id=solution,
                inputs=[pb_value.Value(
                    dataset_uri='file://%s' % dataset_path,
                )],
                expose_outputs=['outputs.0'],
                expose_value_types=['CSV_URI'],
                users=[],
            ))
            results = core.GetFitSolutionResults(
                pb_core.GetFitSolutionResultsRequest(
                    request_id=response.request_id,
                )
            )
            for result in results:
                if result.progress.state == pb_core.COMPLETED:
                    fitted[solution] = result.fitted_solution_id
        except Exception:
            logger.exception("Exception training %r", solution)
    return fitted


def do_test(core, fitted, dataset_path):
    tested = {}
    for fitted_solution in fitted.values():
        try:
            response = core.ProduceSolution(pb_core.ProduceSolutionRequest(
                fitted_solution_id=fitted_solution,
                inputs=[pb_value.Value(
                    dataset_uri='file://%s' % dataset_path,
                )],
                expose_outputs=['outputs.0'],
                expose_value_types=['CSV_URI'],
                users=[],
            ))
            results = core.GetProduceSolutionResults(
                pb_core.GetProduceSolutionResultsRequest(
                    request_id=response.request_id,
                )
            )
            for result in results:
                if result.progress.state == pb_core.COMPLETED:
                    tested[fitted_solution] = result.exposed_outputs['outputs.0'].csv_uri
        except Exception:
            logger.exception("Exception testing %r", fitted_solution)

    return tested


def do_export(core, fitted):
    for i, fitted_solution in enumerate(fitted.values()):
        try:
            core.SolutionExport(pb_core.SolutionExportRequest(
                solution_id=fitted_solution,
                rank=(i + 1.0) / (len(fitted) + 1.0),
            ))
        except Exception:
            logger.exception("Exception exporting %r", fitted_solution)


def do_describe(core, solutions):
    for solution in solutions:
        try:
            core.DescribeSolution(pb_core.DescribeSolutionRequest(
                solution_id=solution,
            ))
        except Exception:
            logger.exception("Exception during describe %r", solution)


def do_save_solution(core, solution_id):
    response = core.SaveSolution(pb_core.SaveSolutionRequest(solution_id=solution_id))

    return response.solution_uri


def do_load_solution(core, solution_path):
    solution_uri = 'file://%s' % solution_path
    response = core.LoadSolution(pb_core.LoadSolutionRequest(solution_uri=solution_uri))

    return response.solution_id


def do_save_fitted_solution(core, fitted):
    saved = {}

    for fitted_solution_id in fitted.values():
        response = core.SaveFittedSolution(pb_core.SaveFittedSolutionRequest(fitted_solution_id=fitted_solution_id))
        parsed_uri = parse.urlparse(response.fitted_solution_uri, allow_fragments=False)
        fitted_solution_path = parsed_uri.path

        with open(fitted_solution_path, 'rb') as fin:
            fitted_object = pickle.load(fin)

        saved[fitted_solution_id] = fitted_object

    return saved


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s")

    channel = grpc.insecure_channel('localhost:{0}'.format(os.environ.get('D3MPORT', 45042)))
    core = LoggingStub(pb_core_grpc.CoreStub(channel), logger)
    train_dataset_path = '/input/TRAIN/dataset_TRAIN/datasetDoc.json'
    test_dataset_path = '/input/TEST/dataset_TEST/datasetDoc.json'

    with open(sys.argv[1]) as problem:
        problem = json.load(problem)

    # Do a hello
    do_hello(core)

    # Do a list primitives
    do_listprimitives(core)
    # Do a search
    solutions = do_search(core, problem, train_dataset_path)

    # Describe the pipelines
    do_describe(core, solutions)

    # Score all found solutions
    do_score(core, problem, solutions, train_dataset_path)

    # Train all found solutions
    fitted = do_train(core, solutions, train_dataset_path)

    # Test all fitted solutions
    do_test(core, fitted, test_dataset_path)

    # Export all fitted solutions
    do_export(core, fitted)


if __name__ == '__main__':
    main()
