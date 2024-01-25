import logging
import json
from os.path import join, dirname

logger = logging.getLogger(__name__)

PRIMITIVES_HIERARCHY_PATH = join(dirname(__file__), 'resource/primitives_hierarchy.json')


def create_primitives_hierarchy():
    pass


def load_primitives_hierarchy():
    with open(PRIMITIVES_HIERARCHY_PATH) as fin:
        primitives = json.load(fin)
    logger.debug('Hierarchy of all primitives loaded')

    return primitives


def load_primitives_types():
    primitive_types = {}
    with open(PRIMITIVES_HIERARCHY_PATH) as fin:
        primitives = json.load(fin)

    for primitive_type, primitive_names in primitives.items():
        for primitive_name in primitive_names:
            primitive_types[primitive_name] = primitive_type

    logger.debug('Primitive types loaded')

    return primitive_types


PRIMITIVES_RANKING_PATH = join(dirname(__file__), 'resource/primitives_ranking.json')


def load_primitives_ranking():
    with open(PRIMITIVES_RANKING_PATH) as fin:
        primitives = json.load(fin)
    logger.debug('Ranking of all primitives loaded')

    return primitives


def record_primitive_performance(pipelines):
    primitives_ranking = load_primitives_ranking()
    
    for idx, pipeline in enumerate(pipelines):
        steps = pipeline.get_pipeline().steps
        score = pipeline.get_score()
        for step_name, _ in steps:
            
            if step_name not in PRIMITIVE_TYPES:
                continue
                
            step_type = PRIMITIVE_TYPES[step_name]
            if step_type not in primitives_ranking:
                primitives_ranking[step_type] = {}
            
            if step_name not in primitives_ranking[step_type]:
                primitives_ranking[step_type][step_name] = {"avg_score": 0, "runs": 0}
            
            ranking = primitives_ranking[step_type][step_name]
            
            weight = 1/(idx+1)
            runs = ranking["runs"] + 1
            avg_score = (ranking["avg_score"] * ranking["runs"] + score * weight) / runs
            
            primitives_ranking[step_type][step_name] = {"avg_score": avg_score, "runs": runs}
    
    with open(PRIMITIVES_RANKING_PATH, "w") as outfile:
        json.dump(primitives_ranking, outfile)
    return primitives_ranking


def load_ranked_primitives_hierarchy():
    ranking = load_primitives_ranking()
    hierarchy = load_primitives_hierarchy()
    for key, primitives in ranking.items():
        ranked_primitives = []
        for primitive in sorted(primitives, key=lambda x: primitives[x]["avg_score"], reverse=True):
            if primitive in hierarchy[key]:
                ranked_primitives.append(primitive)
        hierarchy[key] = ranked_primitives
    return hierarchy


PRIMITIVE_TYPES = load_primitives_types()

if __name__ == '__main__':
    # Run this to create the files for the list and hierarchy of primitives
    create_primitives_hierarchy()
