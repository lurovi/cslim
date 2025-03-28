"""
This script runs the StandardGP algorithm on various datasets and configurations,
logging the results for further analysis.
"""
import math
import time
import uuid

from cslim.algorithms.GP.gp import GP
from cslim.algorithms.GP.operators.mutators import mutate_tree_subtree
from cslim.algorithms.GP.representations.tree_utils import tree_depth, tree_pruning
from cslim.config.gp_config import *
from cslim.utils.logger import log_settings, compute_path_run_log_and_settings
from cslim.utils.utils import get_terminals, validate_inputs


# todo: would not be better to first log the settings and then perform the algorithm?
def gp(X_train: torch.Tensor, y_train: torch.Tensor, X_test: torch.Tensor = None, y_test: torch.Tensor = None,
       dataset_name: str = None, pop_size: int = 100, n_iter: int = 1000, p_xo: float = 0.8,
       elitism: bool = True, n_elites: int = 1, max_depth: int = 17, init_depth: int = 6,
       pressure: int = 2, torus_dim: int = 0, radius: int = 0, cmp_rate: float = 0.0, pop_shape: tuple[int, ...] = tuple(),
       log_path: str = '', seed: int = 1, verbose: int = 1):
    """
    Main function to execute the StandardGP algorithm on specified datasets

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
    max_depth : int, optional
        The maximum depth for the GP trees.
    init_depth : int, optional
        The depth value for the initial GP trees population.
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
    log_path : str, optional
        The path where is created the log directory where results are saved.
    seed : int, optional
        Seed for the randomness
    verbose: int, optional
        Verbosity level.

    Returns
    -------
    Tree
        Returns the best individual at the last generation.
    """

    if pop_shape == tuple():
        pop_shape = (pop_size,)
    
    if torus_dim == 0:
        radius = 0
        cmp_rate = 0.0
        pop_shape = (pop_size,)
    else:
        pressure = 0

    if math.prod(pop_shape) != pop_size:
        raise ValueError(f'The product of dimensions in pop shape {pop_shape} does not math the pop size {pop_size}.')

    validate_inputs(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                    pop_size=pop_size, n_iter=n_iter, elitism=elitism, n_elites=n_elites,
                    pressure=pressure, torus_dim=torus_dim, radius=radius, cmp_rate=cmp_rate, pop_shape=pop_shape,
                    init_depth=init_depth, log_path=log_path)
    assert 0 <= p_xo <= 1, "p_xo must be a number between 0 and 1"
    assert isinstance(max_depth, int), "Input must be a int"

    if not elitism:
        n_elites = 0

    unique_run_id = uuid.uuid1()

    update_gp_config(pressure=pressure, torus_dim=torus_dim, radius=radius, cmp_rate=cmp_rate, verbose=verbose)

    if log_path.strip() == '':
        raise ValueError(f'Please, specify a directory in which you can save the log of the run.')

    log_path = compute_path_run_log_and_settings(
        base_path=log_path,
        method='gp',
        dataset_name=dataset_name,
        pop_size=pop_size,
        n_iter=n_iter,
        n_elites=n_elites,
        pressure=pressure,
        p_crossover=p_xo,
        p_inflate=0.0,
        slim_crossover='default',
        torus_dim=torus_dim,
        radius=radius,
        cmp_rate=cmp_rate,
        pop_shape=pop_shape
    )

    algo = "StandardGP"
    gp_solve_parameters['run_info'] = [algo, unique_run_id, dataset_name]

    TERMINALS = get_terminals(X_train)
    gp_pi_init["TERMINALS"] = TERMINALS
    gp_pi_init["init_pop_size"] = pop_size
    gp_pi_init["init_depth"] = init_depth

    gp_parameters["p_xo"] = p_xo
    gp_parameters["p_m"] = 1 - gp_parameters["p_xo"]
    gp_parameters["pop_size"] = pop_size
    gp_parameters["pop_shape"] = pop_shape
    gp_parameters["mutator"] = mutate_tree_subtree(
        gp_pi_init['init_depth'], TERMINALS, CONSTANTS, FUNCTIONS, p_c=gp_pi_init['p_c']
    )

    gp_solve_parameters["log_path"] = log_path
    gp_solve_parameters["elitism"] = elitism
    gp_solve_parameters["n_elites"] = n_elites
    gp_solve_parameters["max_depth"] = max_depth
    gp_solve_parameters["n_iter"] = n_iter
    gp_solve_parameters["tree_pruner"] = tree_pruning(
        TERMINALS=TERMINALS, CONSTANTS=CONSTANTS, FUNCTIONS=FUNCTIONS, p_c=gp_pi_init["p_c"]
    )
    gp_solve_parameters['depth_calculator'] = tree_depth(FUNCTIONS=FUNCTIONS)
    if X_test is not None and y_test is not None:
        gp_solve_parameters["test_elite"] = True
    else:
        gp_solve_parameters["test_elite"] = False

    optimizer = GP(pi_init=gp_pi_init, **gp_parameters, seed=seed)
    optimizer.solve(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        curr_dataset=dataset_name,
        **gp_solve_parameters
    )

    log_settings(
        path=os.path.join(log_path, f"seed{seed}_settings.csv"),
        settings_dict=[gp_solve_parameters,
                       gp_parameters,
                       gp_pi_init,
                       settings_dict],
        unique_run_id=unique_run_id,
    )

    return optimizer.elite


if __name__ == "__main__":
    from datasets.data_loader import load_ppb
    from cslim.utils.utils import train_test_split

    X, y = load_ppb(X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, p_test=0.4)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, p_test=0.5)

    final_tree = gp(X_train=X_train, y_train=y_train,
                    X_test=X_val, y_test=y_val,
                    dataset_name='ppb', pop_size=100, n_iter=10, pressure=4,
                    log_path='log')

    final_tree.print_tree_representation()
    predictions = final_tree.predict(X_test)
    print(float(rmse(y_true=y_test, y_pred=predictions)))
