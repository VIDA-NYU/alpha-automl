import json
import shutil
import datamart_profiler
import pandas as pd
from datamart_materialize.d3m import D3mWriter
from os.path import join, exists


def create_d3mdataset(csv_path, destination_path, version='4.0.0'):
    metadata = datamart_profiler.process_dataset(csv_path)
    dataset_path = join(destination_path, 'datasetDoc.json')

    if exists(destination_path):
        shutil.rmtree(destination_path)

    writer = D3mWriter(
        dataset_id='internal_dataset',
        destination=destination_path,
        metadata=metadata,
        format_options={'need_d3mindex': True, 'version': version},
    )
    with open(csv_path, 'rb') as source:
        with writer.open_file('wb') as dest:
            shutil.copyfileobj(source, dest)
    writer.finish()

    return dataset_path


def create_d3mproblem(problem_config, csv_path, destination_path):
    if 'target_index' not in problem_config:
        target_index = pd.read_csv(csv_path).columns.get_loc(problem_config['target_name'])
        problem_config['target_index'] = target_index

    problem_path = join(destination_path, 'problemDoc.json')

    problem_json = {
          "about": {
            "problemID": "",
            "problemName": "",
            "problemDescription": "",
            "problemVersion": "4.0.0",
            "problemSchemaVersion": "4.0.0",
            "taskKeywords": problem_config.get('task_keywords', ["classification", "multiClass"])
          },
          "inputs": {
            "data": [
              {
                "datasetID": "internal_dataset",
                "targets": [
                  {
                    "targetIndex": 0,
                    "resID": "learningData",
                    "colIndex": problem_config.get('target_index'),
                    "colName": problem_config.get('target_name')
                  }
                ]
              }
            ],
            "performanceMetrics": [
              {
                "metric": problem_config.get('metric', "accuracy")
              }
            ]
          },
          "expectedOutputs": {
            "predictionsFile": "predictions.csv"
          }
        }

    with open(problem_path, 'w') as fout:
        json.dump(problem_json, fout, indent=4)

    return problem_path
