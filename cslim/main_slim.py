"""
This script runs the SLIM_GSGP algorithm on various datasets and configurations,
logging the results for further analysis.
"""
import math
import time
import uuid

from cslim.algorithms.SLIM_GSGP.slim_gsgp import SLIM_GSGP
from cslim.config.slim_config import *
from cslim.utils.logger import log_settings, compute_path_run_log_and_settings
from cslim.utils.utils import get_terminals, check_slim_version, validate_inputs, generate_random_uniform
from cslim.algorithms.SLIM_GSGP.operators.mutators import *
from cslim.algorithms.SLIM_GSGP.operators.our_geometric_crossover import *
from cslim.algorithms.SLIM_GSGP.operators.standard_geometric_crossover import *
from typing import Callable

ELITES = {}
UNIQUE_RUN_ID = uuid.uuid1()


# todo: would not be better to first log the settings and then perform the algorithm?
# todo: update how the name is saved and make it coherent with the paper
def slim(X_train: torch.Tensor, y_train: torch.Tensor, X_test: torch.Tensor = None, y_test: torch.Tensor = None,
         dataset_name: str = None, slim_version: str = "SLIM+SIG2", pop_size: int = 100,
         n_iter: int = 100, p_xo: float = 0.0, elitism: bool = True, n_elites: int = 1, init_depth: int = 6,
         ms: Callable = generate_random_uniform(0, 1), p_inflate: float = 0.5, p_inflate_post: float = 0.5, iter_post: int = 0,
         slim_crossover: str = 'default',
         pressure: int = 2, torus_dim: int = 0, radius: int = 0, cmp_rate: float = 0.0, pop_shape: tuple[int, ...] = tuple(),
         log_path: str = '', seed: int = 1):
    """
    Main function to execute the SLIM GSGP algorithm on specified datasets

    Parameters
    ----------
    X_train: (torch.Tensor)
        Training input data.
    y_train: (torch.Tensor)
        Training output data.
    X_test: (torch.Tensor), optional
        Testing input data.
    y_test: (torch.Tensor), optional
        Testing output data.
    dataset_name : str, optional
        Dataset name, for logging purposes
    slim_version : list
        The version of SLIM-GSGP that needs to be run
    pop_size : int, optional
        The population size for the genetic programming algorithm (default is 100).
    n_iter : int, optional
        The number of iterations for the genetic programming algorithm (default is 100).
    p_xo : float, optional
        The probability of crossover in the genetic programming algorithm. Must be a number between 0 and 1 (default is 0.8).
    elitism : bool, optional
        Indicate the presence or absence of elitism.
    n_elites : int, optional
        The number of elites.
    init_depth : int, optional
        The depth value for the initial GP trees population.
    ms : Callable, optional
        A function that will generate the mutation step
    p_inflate : float, optional
        Probability to apply the inflate mutation
    p_inflate_post : float, optional
        Probability to apply the inflate mutation after the generation indicated as iter_post
    iter_post : int, optional
        Generation after which the probability of inflate mutation changes from p_inflate to p_inflate_post (the deflate mutation probability is also changed accordingly)
    slim_crossover : str, optional
        The specific SLIM crossover eventually employed if crossover probability is not zero.
    log_path : str, optional
        The path where is created the log directory where results are saved.
    pressure : int, optional
        The tournament size.
    torus_dim: int, optional
        Dimension of the torus in cellular selection (0 if no cellular selection is performed).
    radius: int, optional
        Radius of the torus in cellular selection (makes no sense if no cellular selection is performed).
    cmp_rate: float, optional
        Competitor rate in cellular selection (makes no sense if no cellular selection is performed).
    pop_shape: tuple, optional
        Shape of the grid containing the population in cellular selection (makes no sense if no cellular selection is performed).
    seed : int, optional
        Seed for the randomness

    Returns
    -------
      Tree
        Returns the best individual at the last generation.
    """
    if p_xo == 1.0:
        slim_version = 'SLIM+SIG2'
    op, sig, trees = check_slim_version(slim_version=slim_version)

    if pop_shape == tuple():
        pop_shape = (pop_size,)
    
    if torus_dim == 0:
        radius = 0
        cmp_rate = 0.0
        pop_shape = (pop_size,)
    else:
        pressure = 0

    if iter_post == 0:
        p_inflate_post = p_inflate
    
    if p_inflate == p_inflate_post:
        iter_post = 0

    if p_xo == 0.0:
        slim_crossover = 'default'

    if math.prod(pop_shape) != pop_size:
        raise ValueError(f'The product of dimensions in pop shape {pop_shape} does not math the pop size {pop_size}.')

    validate_inputs(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                    pop_size=pop_size, n_iter=n_iter, elitism=elitism, n_elites=n_elites,
                    pressure=pressure, torus_dim=torus_dim, radius=radius, cmp_rate=cmp_rate, pop_shape=pop_shape,
                    init_depth=init_depth, log_path=log_path)
    assert 0 <= p_xo <= 1, "p_xo must be a number between 0 and 1"

    update_slim_config(pressure=pressure, torus_dim=torus_dim, radius=radius, cmp_rate=cmp_rate)

    if log_path.strip() == '':
        raise ValueError(f'Please, specify a directory in which you can save the log of the run.')

    log_path = compute_path_run_log_and_settings(
        base_path=log_path,
        method='gsgp' + '' + slim_version,
        dataset_name=dataset_name,
        pop_size=pop_size,
        n_iter=n_iter,
        n_elites=n_elites,
        pressure=pressure,
        p_crossover=p_xo,
        p_inflate=p_inflate,
        slim_crossover=slim_crossover,
        torus_dim=torus_dim,
        radius=radius,
        cmp_rate=cmp_rate,
        pop_shape=pop_shape
    )

    slim_gsgp_parameters["two_trees"] = trees
    slim_gsgp_parameters["operator"] = op

    TERMINALS = get_terminals(X_train)

    slim_gsgp_parameters["ms"] = ms
    slim_gsgp_parameters['p_inflate'] = p_inflate
    slim_gsgp_parameters['p_deflate'] = 1 - slim_gsgp_parameters['p_inflate']
    slim_gsgp_parameters['p_inflate_post'] = p_inflate_post
    slim_gsgp_parameters['p_deflate_post'] = 1 - slim_gsgp_parameters['p_inflate_post'] 

    slim_gsgp_pi_init["TERMINALS"] = TERMINALS
    slim_gsgp_pi_init["init_pop_size"] = pop_size
    slim_gsgp_pi_init["init_depth"] = init_depth

    slim_gsgp_parameters["p_xo"] = p_xo
    slim_gsgp_parameters["p_m"] = 1 - slim_gsgp_parameters["p_xo"]
    slim_gsgp_parameters["pop_size"] = pop_size
    slim_gsgp_parameters["pop_shape"] = pop_shape
    slim_gsgp_parameters["inflate_mutator"] = inflate_mutation(
        FUNCTIONS=FUNCTIONS,
        TERMINALS=TERMINALS,
        CONSTANTS=CONSTANTS,
        two_trees=slim_gsgp_parameters['two_trees'],
        operator=slim_gsgp_parameters['operator'],
        sig=sig
    )

    if slim_crossover == "c" or slim_crossover == 'default':
        slim_gsgp_parameters["crossover"] = slim_geometric_crossover(FUNCTIONS=FUNCTIONS,
                                                                     TERMINALS=TERMINALS,
                                                                     CONSTANTS=CONSTANTS,
                                                                     operator=slim_gsgp_parameters[
                                                                         'operator'])
    elif slim_crossover == "ac":
        slim_gsgp_parameters["crossover"] = slim_alpha_geometric_crossover(
            operator=slim_gsgp_parameters[
                'operator'])

    elif slim_crossover == "sc":
        slim_gsgp_parameters["crossover"] = slim_swap_geometric_crossover

    elif slim_crossover == "adc-0.3":
        slim_gsgp_parameters["crossover"] = slim_alpha_deflate_geometric_crossover(
            operator=slim_gsgp_parameters['operator'], perc_off_blocks=0.3)

    elif slim_crossover == "adc-0.7":
        slim_gsgp_parameters["crossover"] = slim_alpha_deflate_geometric_crossover(
            operator=slim_gsgp_parameters['operator'], perc_off_blocks=0.7)

    elif slim_crossover == "sdc-0.3":
        slim_gsgp_parameters["crossover"] = slim_swap_deflate_geometric_crossover(perc_off_blocks=0.3)

    elif slim_crossover == "sdc-0.7":
        slim_gsgp_parameters["crossover"] = slim_swap_deflate_geometric_crossover(perc_off_blocks=0.7)

    elif slim_crossover == 'dgx':
        slim_gsgp_parameters["crossover"] = donor_gxo

    else:
        raise ValueError(f'SLIM Crossover {slim_crossover} not recognized.')

    slim_gsgp_solve_parameters["log_path"] = log_path
    slim_gsgp_solve_parameters["elitism"] = elitism
    slim_gsgp_solve_parameters["n_elites"] = n_elites
    slim_gsgp_solve_parameters["n_iter"] = n_iter
    slim_gsgp_solve_parameters["iter_post"] = iter_post
    slim_gsgp_solve_parameters['run_info'] = [slim_version, UNIQUE_RUN_ID, dataset_name]
    if X_test is not None and y_test is not None:
        slim_gsgp_solve_parameters["test_elite"] = True
    else:
        slim_gsgp_solve_parameters["test_elite"] = False

    optimizer = SLIM_GSGP(
        pi_init=slim_gsgp_pi_init,
        **slim_gsgp_parameters,
        seed=seed
    )

    optimizer.solve(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        curr_dataset=dataset_name,
        **slim_gsgp_solve_parameters
    )

    log_settings(
        path=os.path.join(log_path, f"seed{seed}_settings.csv"),
        settings_dict=[slim_gsgp_solve_parameters,
                       slim_gsgp_parameters,
                       slim_gsgp_pi_init,
                       settings_dict],
        unique_run_id=UNIQUE_RUN_ID
    )

    return optimizer.elite


if __name__ == "__main__":
    from datasets.data_loader import load_ppb
    from cslim.utils.utils import train_test_split, show_individual

    X, y = load_ppb(X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, p_test=0.4)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, p_test=0.5)

    algorithm = "SLIM+SIG2"

    final_tree = slim(X_train=X_train, y_train=y_train, X_test=X_val, y_test=y_val,
                      dataset_name='ppb', slim_version=algorithm, pop_size=100, n_iter=2, pressure=4,
                      log_path='log')

    print(show_individual(final_tree, operator='sum'))
    predictions = final_tree.predict(data=X_test, slim_version=algorithm)
    print(float(rmse(y_true=y_test, y_pred=predictions)))
