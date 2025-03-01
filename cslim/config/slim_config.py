from cslim.algorithms.GP.operators.initializers import rhh
from cslim.algorithms.GSGP.operators.crossover_operators import geometric_crossover
from cslim.algorithms.SLIM_GSGP.operators.mutators import (deflate_mutation)
from cslim.algorithms.SLIM_GSGP.operators.selection_algorithms import \
    tournament_selection_min_slim
from cslim.datasets.data_loader import *
from cslim.evaluators.fitness_functions import rmse
from cslim.utils.utils import (get_best_min, protected_div)

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

# SLIM GSGP solve parameters
slim_gsgp_solve_parameters = {
    "log": 5,
    "verbose": 1,
    "run_info": None,
    "ffunction": rmse,
    "max_depth": None,
    "reconstruct": True
}

# SLIM GSGP parameters
slim_gsgp_parameters = {
    "initializer": rhh,
    "pressure": 2,
    "selector": tournament_selection_min_slim(2),
    "crossover": geometric_crossover,
    "ms": None,
    "inflate_mutator": None,
    "deflate_mutator": deflate_mutation,
    "settings_dict": settings_dict,
    "find_elit_func": get_best_min,
    "copy_parent": None,
    "operator": None,
    "torus_dim": 0,
    "radius": 0,
    "cmp_rate": 0.0
}

slim_gsgp_pi_init = {
    'FUNCTIONS': FUNCTIONS,
    'CONSTANTS': CONSTANTS,
    "p_c": 0.0
}


def update_slim_config(
        p_test: float = 0.2,
        log: int = 5,
        verbose: int = 1,
        reconstruct: bool = True,
        pressure: int = 2,
        p_c: float = 0.0,
        torus_dim: int = 0,
        radius: int = 0,
        cmp_rate: float = 0.0
) -> None:
    settings_dict['p_test'] = p_test
    slim_gsgp_solve_parameters['log'] = log
    slim_gsgp_solve_parameters['verbose'] = verbose
    slim_gsgp_solve_parameters['reconstruct'] = reconstruct
    slim_gsgp_parameters['pressure'] = pressure
    slim_gsgp_parameters['selector'] = tournament_selection_min_slim(pressure)
    slim_gsgp_pi_init['p_c'] = p_c

    slim_gsgp_parameters['torus_dim'] = torus_dim
    slim_gsgp_parameters['radius'] = radius
    slim_gsgp_parameters['cmp_rate'] = cmp_rate
