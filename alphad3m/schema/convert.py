"""Convert to/from the JSON representation.
"""

import importlib
import pickle
import logging
from d3m.utils import compute_digest

logger = logging.getLogger(__name__)


def get_class(name):
    package, classname = name.rsplit('.', 1)
    return getattr(importlib.import_module(package), classname)


def _add_step(steps, modules, params, module_to_step, mod):
    if mod.id in module_to_step:
        return module_to_step[mod.id]

    # Special case: the "dataset" module
    if mod.package == 'data' and mod.name == 'dataset':
        module_to_step[mod.id] = 'inputs.0'
        return 'inputs.0'
    elif mod.package != 'd3m':
        raise ValueError("Got unknown module '%s:%s'", mod.package, mod.name)

    # Recursively walk upstream modules (to get `steps` in topological order)
    # Add inputs to a dictionary, in deterministic order
    inputs = {}

    for conn in sorted(mod.connections_to, key=lambda c: c.to_input_name):
        step = _add_step(steps, modules, params, module_to_step, modules[conn.from_module_id])

        # index is a special connection to keep the order of steps in fixed pipeline templates
        if 'index' in conn.to_input_name:
            continue

        if step.startswith('inputs.'):
            inputs[conn.to_input_name] = step
        else:
            if conn.to_input_name in inputs:
                previous_value = inputs[conn.to_input_name]
                if isinstance(previous_value, str):
                    inputs[conn.to_input_name] = [previous_value] + ['%s.%s' % (step, conn.from_output_name)]
                else:
                    inputs[conn.to_input_name].append('%s.%s' % (step, conn.from_output_name))
            else:
                inputs[conn.to_input_name] = '%s.%s' % (step, conn.from_output_name)

    outputs = sorted(set([c.from_output_name for c in mod.connections_from if c.from_output_name != 'index']))
    if len(outputs) == 0:  # Add 'produce' output for the last step of the pipeline
        outputs = {'produce'}

    klass = get_class(mod.name)
    primitive_desc = {
        key: value
        for key, value in klass.metadata.query().items()
        if key in {'id', 'version', 'python_path', 'name', 'digest'}
    }

    # Create step description
    if len(inputs) > 0:
        step = {
            'type': 'PRIMITIVE',
            'primitive': primitive_desc,
            'arguments': {
                name: {
                    'type': 'CONTAINER',
                    'data': data,
                }
                for name, data in inputs.items()
            },
            'outputs': [{'id': o} for o in outputs]
        }
    else:
        step = {
            'type': 'PRIMITIVE',
            'primitive': primitive_desc,
        }

    # If hyperparameters are set, export them
    if mod.id in params and 'hyperparams' in params[mod.id]:
        hyperparams = pickle.loads(params[mod.id]['hyperparams'])
        # We check whether the hyperparameters have a value or the complete description
        hyperparams = {
            k: {'type': v['type'] if isinstance(v, dict) and 'type' in v else 'VALUE',
                'data': v['data'] if isinstance(v, dict) and 'data' in v else v}
            for k, v in hyperparams.items()
        }
        step['hyperparams'] = hyperparams

    step_nb = 'steps.%d' % len(steps)
    steps.append(step)
    module_to_step[mod.id] = step_nb

    return step_nb


def to_d3m_json(pipeline):
    """Converts a Pipeline to the JSON schema from metalearning working group.
    """
    steps = []
    modules = {mod.id: mod for mod in pipeline.modules}
    params = {}
    for param in pipeline.parameters:
        params.setdefault(param.module_id, {})[param.name] = param.value
    module_to_step = {}
    for _, mod in sorted(modules.items(), key=lambda x: x[0]):
        _add_step(steps, modules, params, module_to_step, mod)

    json_pipeline = {
        'id': str(pipeline.id),
        'name': str(pipeline.id),
        'description': pipeline.origin or '',
        'schema': 'https://metadata.datadrivendiscovery.org/schemas/'
                  'v0/pipeline.json',
        'created': pipeline.created_date.isoformat() + 'Z',
        'context': 'TESTING',
        'inputs': [
            {'name': "input dataset"},
        ],
        'outputs': [
            {
                'data': 'steps.%d.produce' % (len(steps) - 1),
                'name': "predictions",
            }
        ],
        'steps': steps,
    }

    try:
        json_pipeline['digest'] = compute_digest(json_pipeline)
    except:
        logger.error('Creating digest, skipping this field')

    return json_pipeline
