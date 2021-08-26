import os
import logging
import itertools
from nltk.grammar import Production, Nonterminal, CFG, is_terminal, is_nonterminal
from alphad3m.metafeature.metalearningdb_miner import create_grammar_from_metalearningdb

logger = logging.getLogger(__name__)
BASE_GRAMMAR_PATH = os.path.join(os.path.dirname(__file__), '../resource/base_grammar.bnf')
COMPLETE_GRAMMAR_PATH = os.path.join(os.environ.get('D3MOUTPUTDIR'), 'temp', 'complete_grammar.bnf')
TASK_GRAMMAR_PATH = os.path.join(os.environ.get('D3MOUTPUTDIR'), 'temp', 'task_grammar.bnf')


def load_grammar(grammar_string, encoders, use_imputer):
    production_rules = []

    for production_rule in grammar_string.split('\n'):
        if production_rule.startswith('ENCODERS -> '):
            if len(encoders) > 0:
                production_rule = 'ENCODERS -> %s' % ' '.join(encoders)
            else:
                production_rule = "ENCODERS -> 'E'"
        elif production_rule.startswith('IMPUTATION -> '):
            if not use_imputer:
                production_rule = "IMPUTATION -> 'E'"
        production_rules.append(production_rule.strip())

    grammar_string = '\n'.join(production_rules)
    return CFG.fromstring(grammar_string)


def create_global_grammar(base_grammar, primitives):
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
    game_grammar = {'START': start_symbol, 'NON_TERMINALS': {}, 'TERMINALS': {}, 'RULES': {}, 'RULES_LOOKUP': {}}
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


def load_manual_grammar(task, primitives, encoders, use_imputer):
    with open(BASE_GRAMMAR_PATH) as fin:
        grammar_string = fin.read()

    base_grammar = load_grammar(grammar_string, encoders, use_imputer)
    global_grammar = create_global_grammar(base_grammar, primitives)
    task_grammar = create_task_grammar(global_grammar, task)
    game_grammar = create_game_grammar(task_grammar)

    return game_grammar


def load_automatic_grammar(task, task_keywords, encoders, use_imputer):
    grammar_string, primitives = create_grammar_from_metalearningdb(task, task_keywords)
    base_grammar = load_grammar(grammar_string, encoders, use_imputer)
    global_grammar = create_global_grammar(base_grammar, primitives)
    task_grammar = create_task_grammar(global_grammar, task)
    game_grammar = create_game_grammar(task_grammar)

    return game_grammar
