from cslim.algorithms.GP.operators.initializers import rhh
from cslim.algorithms.GP.operators.selection_algorithms import \
    tournament_selection_min
from cslim.algorithms.GSGP.operators.crossover_operators import geometric_crossover
from cslim.algorithms.GSGP.operators.mutators import standard_geometric_mutation
from cslim.datasets.data_loader import *
from cslim.evaluators.fitness_functions import rmse
from cslim.utils.utils import (get_best_min, protected_div)

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

# GSGP solve parameters
gsgp_solve_parameters = {
    "log": 5,
    "verbose": 1,
    "test_elite": True,
    "run_info": None,
    "ffunction": rmse,
    "reconstruct": True,
}

# GSGP parameters
gsgp_parameters = {
    "initializer": rhh,
    "pressure": 2,
    "selector": tournament_selection_min(2),
    "crossover": geometric_crossover,
    "mutator": standard_geometric_mutation,
    "settings_dict": settings_dict,
    "find_elit_func": get_best_min,
    "torus_dim": 0,
    "radius": 0,
    "cmp_rate": 0.0
}

gsgp_pi_init = {
    'FUNCTIONS': FUNCTIONS,
    'CONSTANTS': CONSTANTS,
    "p_c": 0.0
}


def update_gsgp_config(
        p_test: float = 0.2,
        log: int = 5,
        verbose: int = 1,
        test_elite: bool = True,
        reconstruct: bool = True,
        pressure: int = 2,
        p_c: float = 0.0,
        torus_dim: int = 0,
        radius: int = 0,
        cmp_rate: float = 0.0
) -> None:
    settings_dict['p_test'] = p_test
    gsgp_solve_parameters['log'] = log
    gsgp_solve_parameters['verbose'] = verbose
    gsgp_solve_parameters['test_elite'] = test_elite
    gsgp_solve_parameters['reconstruct'] = reconstruct
    gsgp_parameters['pressure'] = pressure
    gsgp_parameters['selector'] = tournament_selection_min(pressure)
    gsgp_pi_init['p_c'] = p_c

    gsgp_parameters['torus_dim'] = torus_dim
    gsgp_parameters['radius'] = radius
    gsgp_parameters['cmp_rate'] = cmp_rate
