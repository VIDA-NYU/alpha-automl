import os
import logging
import itertools
from os.path import join, dirname
from nltk.grammar import Production, Nonterminal, CFG, is_terminal, is_nonterminal
from alphad3m.utils import load_primitives_types
from alphad3m.primitive_loader import load_primitives_hierarchy
from alphad3m.metalearning.grammar_builder import create_metalearningdb_grammar

logger = logging.getLogger(__name__)
BASE_GRAMMAR_PATH = join(dirname(__file__), 'resource/base_grammar.bnf')
COMPLETE_GRAMMAR_PATH = join(os.environ.get('D3MOUTPUTDIR'), 'temp', 'complete_grammar.bnf')
TASK_GRAMMAR_PATH = join(os.environ.get('D3MOUTPUTDIR'), 'temp', 'task_grammar.bnf')


def create_global_grammar(grammar_string, primitives):
    base_grammar = CFG.fromstring(grammar_string)
    new_productions = []

    for production in base_grammar.productions():
        primitive_type = production.lhs().symbol()
        if primitive_type in primitives:
            new_rhs_list = []
            for token in production.rhs():
                if isinstance(token, str) and token.startswith('primitive_'):
                    new_rhs_list.append(primitives[primitive_type])
                else:
                    new_rhs_list.append([token])
            for new_rhs in itertools.product(*new_rhs_list):
                new_productions.append(Production(production.lhs(), new_rhs))
        else:
            new_productions.append(production)

    complete_grammar = CFG(Nonterminal('S'), new_productions)

    with open(COMPLETE_GRAMMAR_PATH, 'w') as fout:
        fout.write('\n'.join([str(x) for x in complete_grammar.productions()]))

    return complete_grammar


def create_task_grammar(global_grammar, task):
    logger.info('Creating specific grammar for task %s' % task)
    productions = global_grammar.productions(Nonterminal(task))
    start_token = Nonterminal('S')
    new_productions = []

    for start_production in productions:
        first_token = start_production.rhs()[0]
        if is_nonterminal(first_token) and first_token.symbol().endswith('_TASK'):
            for new_start_production in global_grammar.productions(first_token):
                new_productions.append(Production(start_token, new_start_production.rhs()))
        else:
            new_productions.append(Production(start_token, start_production.rhs()))

    for production in global_grammar.productions():
        for new_production in new_productions:
            if production.lhs() in new_production.rhs() and production not in new_productions:
                new_productions.append(production)

    task_grammar = CFG(start_token, new_productions)

    with open(TASK_GRAMMAR_PATH, 'w') as fout:
        fout.write('\n'.join([str(x) for x in task_grammar.productions()]))

    return task_grammar


def create_game_grammar(grammar):
    # Convert a context-free grammar to the game format
    start_symbol = grammar.start().symbol()
    game_grammar = {'START': start_symbol, 'NON_TERMINALS': {}, 'TERMINALS': {}, 'RULES': {}, 'RULES_LOOKUP': {},
                    'RULES_PROBA': {'GLOBAL': {}, 'LOCAL': {}, 'TYPES': {}}}
    terminals = []

    logger.info('Creating game grammar')
    for production in grammar.productions():
        non_terminal = production.lhs().symbol()
        production_str = str(production).replace('\'', '')

        game_grammar['RULES'][production_str] = len(game_grammar['RULES']) + 1

        if non_terminal not in game_grammar['NON_TERMINALS']:
            game_grammar['NON_TERMINALS'][non_terminal] = len(game_grammar['NON_TERMINALS']) + 1

        if non_terminal not in game_grammar['RULES_LOOKUP']:
            game_grammar['RULES_LOOKUP'][non_terminal] = []
        game_grammar['RULES_LOOKUP'][non_terminal].append(production_str)

        for token in production.rhs():
            if is_terminal(token) and token != 'E' and token not in terminals:
                terminals.append(token)

    game_grammar['TERMINALS'] = {t: i+len(game_grammar['NON_TERMINALS']) for i, t in enumerate(terminals, 1)}
    game_grammar['TERMINALS']['E'] = 0  # Special case for the empty symbol

    return game_grammar


def add_probabilities(game_grammar, probabilities):
    for primitive_type, primitive_probabilities in probabilities['global'].items():
        for primitive, probability in primitive_probabilities.items():
            id_rule_str = '%s -> %s' % (primitive_type, primitive)
            if id_rule_str in game_grammar['RULES']:
                id_rule_int = game_grammar['RULES'][id_rule_str]
                game_grammar['RULES_PROBA']['GLOBAL'][id_rule_int] = (id_rule_str, probability)

    for pattern, pattern_probabilities in probabilities['local'].items():
        game_grammar['RULES_PROBA']['LOCAL'][pattern] = {}
        for primitive_type, primitive_probabilities in pattern_probabilities.items():
            for primitive, probability in primitive_probabilities.items():
                id_rule_str = '%s -> %s' % (primitive_type, primitive)
                if id_rule_str in game_grammar['RULES']:
                    id_rule_int = game_grammar['RULES'][id_rule_str]
                    game_grammar['RULES_PROBA']['LOCAL'][pattern][id_rule_int] = (id_rule_str, probability)
    game_grammar['RULES_PROBA']['TYPES'] = probabilities['types']

    return game_grammar


def modify_manual_grammar(encoders, use_imputer):
    new_grammar = ''

    with open(BASE_GRAMMAR_PATH) as fin:
        for production_rule in fin.readlines():
            if production_rule.startswith('ENCODERS -> '):
                if len(encoders) > 0:
                    production_rule = 'ENCODERS -> %s\n' % ' '.join(encoders)
                else:
                    production_rule = "ENCODERS -> 'E'\n"
            elif production_rule.startswith('IMPUTATION -> '):
                if not use_imputer:
                    production_rule = "IMPUTATION -> 'E'\n"
            new_grammar += production_rule

    return new_grammar


def modify_text_primitives(primitives, task_keywords):
    if 'text' not in task_keywords and 'TEXT_FEATURIZER' in primitives:
        # Ignore some text processing primitives for non-text tasks
        ignore_primitives = {'d3m.primitives.feature_extraction.count_vectorizer.SKlearn',
                             'd3m.primitives.feature_extraction.boc.UBC', 'd3m.primitives.feature_extraction.bow.UBC',
                             'd3m.primitives.feature_extraction.nk_sent2vec.Sent2Vec',
                             'd3m.primitives.feature_extraction.tfidf_vectorizer.BBNTfidfTransformer'}
        primitives['TEXT_FEATURIZER'] = [p for p in primitives['TEXT_FEATURIZER'] if p not in ignore_primitives]

    return primitives


def load_manual_grammar(task, task_keywords, encoders, use_imputer, include_primitives, exclude_primitives):
    primitives = load_primitives_hierarchy()
    primitives = modify_text_primitives(primitives, task_keywords)
    primitives = modify_search_space(primitives, include_primitives, exclude_primitives)
    grammar_string = modify_manual_grammar(encoders, use_imputer)
    global_grammar = create_global_grammar(grammar_string, primitives)
    task_grammar = create_task_grammar(global_grammar, task)
    game_grammar = create_game_grammar(task_grammar)

    return game_grammar


def load_automatic_grammar(task, dataset_path, target_column, task_keywords, include_primitives, exclude_primitives,
                           use_probabilities):
    grammar_string, primitives = create_metalearningdb_grammar(task, dataset_path, target_column, task_keywords)
    if grammar_string is None:
        return None

    primitives['hierarchy'] = modify_search_space(primitives['hierarchy'], include_primitives, exclude_primitives)
    global_grammar = create_global_grammar(grammar_string, primitives['hierarchy'])
    task_grammar = create_task_grammar(global_grammar, task)
    game_grammar = create_game_grammar(task_grammar)

    if use_probabilities:
        game_grammar = add_probabilities(game_grammar, primitives['probabilities'])

    return game_grammar


def modify_search_space(primitives, include_primitives, exclude_primitives):
    primitives_types = load_primitives_types()

    for exclude_primitive in exclude_primitives:
        primitive_type = primitives_types.get(exclude_primitive, None)
        if primitive_type in primitives:
            primitives[primitive_type] = [i for i in primitives[primitive_type] if i != exclude_primitive]

    for include_primitive in include_primitives:
        primitive_type = primitives_types.get(include_primitive, None)
        if primitive_type in primitives:
            if include_primitive not in primitives[primitive_type]:
                primitives[primitive_type].append(include_primitive)

    return primitives
