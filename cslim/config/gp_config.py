from cslim.algorithms.GP.operators.crossover_operators import crossover_trees
from cslim.algorithms.GP.operators.initializers import rhh
from cslim.algorithms.GP.operators.selection_algorithms import \
    tournament_selection_min

from cslim.datasets.data_loader import *
from cslim.evaluators.fitness_functions import rmse
from cslim.utils.utils import (get_best_max, get_best_min,
                              protected_div)

# Define functions and constants
# todo use only one dictionary for the parameters of each algorithm

FUNCTIONS = {
    'add': {'function': torch.add, 'arity': 2},
    'subtract': {'function': torch.sub, 'arity': 2},
    'multiply': {'function': torch.mul, 'arity': 2},
    'divide': {'function': protected_div, 'arity': 2}
}

CONSTANTS = {
    'constant_2': lambda _: torch.tensor(2.0),
    'constant_3': lambda _: torch.tensor(3.0),
    'constant_4': lambda _: torch.tensor(4.0),
    'constant_5': lambda _: torch.tensor(5.0),
    'constant__1': lambda _: torch.tensor(-1.0)
}

# Set parameters
settings_dict = {"p_test": 0.2}

# GP solve parameters
gp_solve_parameters = {
    "log": 5,
    "verbose": 1,
    "test_elite": True,
    "run_info": None,
    "max_": False,
    "ffunction": rmse,
    "tree_pruner": None
}

# GP parameters
gp_parameters = {
    "initializer": rhh,
    "pressure": 2,
    "selector": tournament_selection_min(2),
    "crossover": crossover_trees(FUNCTIONS),
    "settings_dict": settings_dict,
    "find_elit_func": get_best_max if gp_solve_parameters["max_"] else get_best_min,
    "torus_dim": 0,
    "radius": 0,
    "cmp_rate": 0.0
}

gp_pi_init = {
    'FUNCTIONS': FUNCTIONS,
    'CONSTANTS': CONSTANTS,
    "p_c": 0.0
}


def update_gp_config(
        p_test: float = 0.2,
        log: int = 5,
        verbose: int = 1,
        test_elite: bool = True,
        max_: bool = False,
        pressure: int = 2,
        p_c: float = 0.0,
        torus_dim: int = 0,
        radius: int = 0,
        cmp_rate: float = 0.0
) -> None:
    settings_dict['p_test'] = p_test
    gp_solve_parameters['log'] = log
    gp_solve_parameters['verbose'] = verbose
    gp_solve_parameters['test_elite'] = test_elite
    gp_solve_parameters['max_'] = max_
    gp_parameters['pressure'] = pressure
    gp_parameters['selector'] = tournament_selection_min(pressure)
    gp_pi_init['p_c'] = p_c

    gp_parameters['torus_dim'] = torus_dim
    gp_parameters['radius'] = radius
    gp_parameters['cmp_rate'] = cmp_rate
