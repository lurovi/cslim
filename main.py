import math
import traceback
import cProfile

from cslim.datasets.data_loader import read_csv_data
from cslim.main_gp import gp
from cslim.main_slim import slim
from cslim.main_gsgp import gsgp
import os
import zlib
import threading
from argparse import ArgumentParser, Namespace
import torch.multiprocessing as mp
import torch

from cslim.utils.logger import is_valid_filename

completed_csv_lock = threading.Lock()


def main():
    torch.use_deterministic_algorithms(True)
    # Setting the device in which data have to be loaded. It can be either CPU or GPU (cuda), if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    results_path: str = 'results/'
    run_with_exceptions_path: str = 'run_with_exceptions/'

    if not os.path.isdir(results_path):
        os.makedirs(results_path, exist_ok=True)

    if not os.path.isdir(run_with_exceptions_path):
        os.makedirs(run_with_exceptions_path, exist_ok=True)

    arg_parser: ArgumentParser = ArgumentParser(description="cslim arguments.")

    arg_parser.add_argument("--seed_index", type=int,
                            help=f"Index of the random seed.")
    arg_parser.add_argument("--algorithm", type=str,
                            help=f"Algorithm.")
    arg_parser.add_argument("--dataset", type=str,
                            help=f"Dataset name.")
    arg_parser.add_argument("--pop_size", type=int,
                            help=f"Population size.")
    arg_parser.add_argument("--n_iter", type=int,
                            help="Number of generations.")
    arg_parser.add_argument("--n_elites", type=int,
                            help="Number of elites.")
    arg_parser.add_argument("--pressure", type=int,
                            help="Tournament size.")
    arg_parser.add_argument("--slim_crossover", type=str,
                            help=f"Crossover of SLIM-GSGP.")
    arg_parser.add_argument("--p_inflate", type=float,
                            help=f"Inflate probability.")
    arg_parser.add_argument("--p_crossover", type=float,
                            help=f"Crossover probability.")
    arg_parser.add_argument("--torus_dim", type=int,
                            help="Dimension of the torus.")
    arg_parser.add_argument("--pop_shape", type=str,
                            help="Shape of the population as string with dimensions separated by x (e.g. 10x10, 20x20, 30x30, 5x4x4).")
    arg_parser.add_argument("--radius", type=int,
                            help="Radius.")
    arg_parser.add_argument("--cmp_rate", type=float,
                            help=f"Competitor rate.")
    arg_parser.add_argument("--run_id", type=str, default='default',
                            help="The run id, used for logging purposes of successful runs.")
    arg_parser.add_argument("--verbose", required=False, action="store_true",
                            help="Verbose flag.")
    arg_parser.add_argument("--profile", required=False, action="store_true",
                            help="Whether to run and log profiling of code or not.")

    cmd_args: Namespace = arg_parser.parse_args()

    seed_index: int = cmd_args.seed_index
    algorithm: str = cmd_args.algorithm
    dataset: str = cmd_args.dataset
    pop_size: int = cmd_args.pop_size
    n_iter: int = cmd_args.n_iter
    n_elites: int = cmd_args.n_elites
    pressure: int = cmd_args.pressure
    slim_crossover: str = cmd_args.slim_crossover
    p_inflate: float = cmd_args.p_inflate
    p_crossover: float = cmd_args.p_crossover
    torus_dim: int = cmd_args.torus_dim
    pop_shape: str = cmd_args.pop_shape
    radius: int = cmd_args.radius
    cmp_rate: float = cmd_args.cmp_rate
    run_id: str = cmd_args.run_id

    verbose: int = int(cmd_args.verbose)
    profiling: int = int(cmd_args.profile)

    args_string = f"{seed_index},{algorithm},{dataset},{pop_size},{n_iter},{n_elites},{pressure},{slim_crossover},{p_inflate},{p_crossover},{torus_dim},{pop_shape},{radius},{cmp_rate},{run_id}"
    all_items_string = ",".join(f"{key}={value}" for key, value in vars(cmd_args).items())

    pr = None
    if profiling != 0:
        pr = cProfile.Profile()
        pr.enable()

    try:
        if not is_valid_filename(run_id):
            raise ValueError(f'run_id {run_id} is not a valid filename.')

        if seed_index < 1:
            raise AttributeError(f'seed_index does not start from 1, it is {seed_index}.')

        d = read_csv_data('cslim/datasets/data_csv/', dataset, seed_index)  # DATA-SPLIT INDEXES START FROM 1
        X_train = d['train'][0]
        y_train = d['train'][1]
        X_test = d['test'][0]
        y_test = d['test'][1]

        with open('random_seeds.txt', 'r') as f:
            # THE ACTUAL SEED TO BE USED IS LOCATED AT POSITION seed_index - 1 SINCE seed_index IS AN INDEX THAT STARTS FROM 1
            all_actual_seeds: list[int] = [int(curr_actual_seed_as_str) for curr_actual_seed_as_str in f.readlines()]
        seed: int = all_actual_seeds[seed_index - 1]

        init_depth: int = 6
        elitism: bool = True

        if torus_dim == 0:
            radius = 0
            cmp_rate = 0.0
            actual_pop_shape = (pop_size,)
        else:
            pressure = 0
            actual_pop_shape = tuple([int(np) for np in pop_shape.split('x')])
            if math.prod(actual_pop_shape) != pop_size:
                raise ValueError(f'The product of dimensions in pop shape {actual_pop_shape} does not math the pop size {pop_size}.')

        if algorithm in ('gp', 'gsgp'):
            slim_crossover = 'default'
            p_inflate = 0.0

        if 'slim' in algorithm and p_crossover == 0.0:
            slim_crossover = 'default'
        if 'slim' in algorithm and p_crossover == 1.0:
            algorithm = 'slim+sig2'

        if algorithm == 'gp':
            gp(X_train=X_train,
               y_train=y_train,
               X_test=X_test,
               y_test=y_test,
               dataset_name=dataset,
               pop_size=pop_size,
               n_iter=n_iter,
               p_xo=p_crossover,
               elitism=elitism,
               n_elites=n_elites,
               init_depth=init_depth,
               pressure=pressure,
               torus_dim=torus_dim,
               radius=radius,
               cmp_rate=cmp_rate,
               pop_shape=actual_pop_shape,
               log_path=results_path,
               seed=seed,
               verbose=verbose,
               )
        elif algorithm == 'gsgp':
            gsgp(X_train=X_train,
                 y_train=y_train,
                 X_test=X_test,
                 y_test=y_test,
                 dataset_name=dataset,
                 pop_size=pop_size,
                 n_iter=n_iter,
                 p_xo=p_crossover,
                 elitism=elitism,
                 n_elites=n_elites,
                 init_depth=init_depth,
                 pressure=pressure,
                 torus_dim=torus_dim,
                 radius=radius,
                 cmp_rate=cmp_rate,
                 pop_shape=actual_pop_shape,
                 log_path=results_path,
                 seed=seed,
                 verbose=verbose,
                 )
        elif 'slim' in algorithm:
            slim(X_train=X_train,
                 y_train=y_train,
                 X_test=X_test,
                 y_test=y_test,
                 dataset_name=dataset,
                 slim_version=algorithm.upper(),
                 pop_size=pop_size,
                 n_iter=n_iter,
                 p_xo=p_crossover,
                 elitism=elitism,
                 n_elites=n_elites,
                 init_depth=init_depth,
                 p_inflate=p_inflate,
                 p_inflate_post=p_inflate,
                 iter_post=0,
                 slim_crossover=slim_crossover,
                 pressure=pressure,
                 torus_dim=torus_dim,
                 radius=radius,
                 cmp_rate=cmp_rate,
                 pop_shape=actual_pop_shape,
                 log_path=results_path,
                 seed=seed,
                 verbose=verbose,
                 )
        else:
            raise AttributeError(f'Unrecognized algorithm {algorithm}.')

        with completed_csv_lock:
            with open(os.path.join(results_path, f'completed_{run_id}.txt'), 'a+') as terminal_std_out:
                terminal_std_out.write(args_string)
                terminal_std_out.write('\n')
            print(f'Completed run: {all_items_string}.')
    except Exception as e:
        try:
            error_string = str(traceback.format_exc())
            with open(os.path.join(run_with_exceptions_path, f'error_{zlib.adler32(bytes(args_string, "utf-8"))}.txt'), 'w') as f:
                f.write(all_items_string + '\n\n' + error_string)
            print(f'\nException in run: {all_items_string}.\n\n{str(e)}\n\n')
        except Exception as ee:
            with open(os.path.join(run_with_exceptions_path, 'error_in_error.txt'), 'w') as f:
                f.write(str(traceback.format_exc()) + '\n\n')
            print(str(ee))

    if profiling != 0:
        pr.disable()
        pr.print_stats(sort='tottime')


if __name__ == '__main__':
    main()
