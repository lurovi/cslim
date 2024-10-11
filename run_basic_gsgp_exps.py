from cslim.datasets.data_loader import read_csv_data
from cslim.utils.parallel import torch_multiprocessing_parallelize
from cslim.main_gsgp import gsgp
import os
import time
import datetime
import torch.multiprocessing as mp
import torch


def execute(
        path: str,
        dataset_name: str,
        seed: int,
        pop_size: int,
        n_iter: int,
        n_elites: int,
        pressure: int,
        torus_dim: int,
        radius: int,
        cmp_rate: float,
        pop_shape: tuple[int, ...]
    ) -> None:

    d = read_csv_data('cslim/datasets/data_csv/', dataset_name, seed) # DATA SPLIT INDEXES START FROM 1
    X_train = d['train'][0]
    y_train = d['train'][1]
    X_test = d['test'][0]
    y_test = d['test'][1]

    with open('random_seeds.txt', 'r') as f:
        # THE ACTUAL SEED TO BE USED IS LOCATED AT POSITION SEED - 1 SINCE SEED IS AN INDEX THAT STARTS FROM 1
        all_actual_seeds: list[int] = [int(curr_actual_seed_as_str) for curr_actual_seed_as_str in f.readlines()]
    
    gsgp(X_train=X_train,
         y_train=y_train,
         X_test=X_test,
         y_test=y_test,
         dataset_name=dataset_name,
         pop_size=pop_size,
         n_iter=n_iter,
         elitism=True,
         n_elites=n_elites,
         init_depth=6,
         pressure=pressure,
         torus_dim=torus_dim,
         radius=radius,
         cmp_rate=cmp_rate,
         pop_shape=pop_shape,
         log_path=path,
         seed=all_actual_seeds[seed - 1]
    )

    verbose_output: str = f'SEED{seed} PopSize {pop_size} NIter {n_iter} NElites {n_elites} Pressure {pressure} Dataset {dataset_name} TorusDim {torus_dim} Radius {radius} CmpRate {cmp_rate} PopShape {str(pop_shape)}'
    print(verbose_output)
    with open(os.path.join(path, 'terminal_std_out.txt'), 'a+') as terminal_std_out:
        terminal_std_out.write(verbose_output)
        terminal_std_out.write('\n')

    return


if __name__ == '__main__':
    torch.use_deterministic_algorithms(True)
    # Setting the device in which data have to be loaded. It can be either CPU or GPU (cuda), if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    p_inflate: float = 0.3
    p_inflate_as_str: str = str(p_inflate).replace(".", "d")
    path: str = f'results_p_inflate_{p_inflate_as_str}/'

    if not os.path.isdir(path):
        os.makedirs(path)

    # ==============================
    # FIXED PARAMETERS
    # ==============================

    n_reps: int = 30

    pop_size: int = 100
    n_iter: int = 300
    n_elites: int = 1
    pressure: int = 4

    # ==============================
    # CELLULAR PARAMETERS
    # ==============================

    torus_dim: int = 2
    pop_shape: tuple[int, ...] = (int(pop_size ** 0.5), int(pop_size ** 0.5))
    all_radius: list[int] = [1, 2, 3]
    all_cmp_rates: list[float] = [0.6, 1.0]

    # ==============================
    # DATA AND ALGORITHMS
    # ==============================

    dataset_names: list[str] = ['airfoil', 'concrete', 'slump', 'parkinson', 'yacht']
    
    # ==============================
    # POPULATING PARAMETERS SETS
    # ==============================

    parameters = []

    # UNCOMMENT FOR RUNNING A SINGLE EXECUTION

    # parameters.append({'path': path,
    #                    'dataset_name': 'parkinson',
    #                    'seed': 1,
    #                    'pop_size': pop_size,
    #                    'n_iter': n_iter,
    #                    'n_elites': n_elites,
    #                    'pop_shape': pop_shape,
    #                    'pressure': pressure,
    #                    'torus_dim': 2,
    #                    'radius': 3,
    #                    'cmp_rate': 1.0
    #                  })

    # UNCOMMENT FOR POPULATING PARAMETERS FOR PARALLEL EXECUTION

    for dataset_name in dataset_names:
        for seed in range(1, n_reps + 1): # SEED INDEX MUST START FROM 1
            parameters.append({'path': path,
                                'dataset_name': dataset_name,
                                'seed': seed,
                                'pop_size': pop_size,
                                'n_iter': n_iter,
                                'n_elites': n_elites,
                                'pop_shape': (pop_size,),
                                'pressure': pressure,
                                'torus_dim': 0,
                                'radius': 0,
                                'cmp_rate': 0.0
                                })
            for radius in all_radius:
                for cmp_rate in all_cmp_rates:
                    parameters.append({'path': path,
                                        'dataset_name': dataset_name,
                                        'seed': seed,
                                        'pop_size': pop_size,
                                        'n_iter': n_iter,
                                        'n_elites': n_elites,
                                        'pop_shape': pop_shape,
                                        'pressure': 0,
                                        'torus_dim': torus_dim,
                                        'radius': radius,
                                        'cmp_rate': cmp_rate
                                    })


    with open(os.path.join(path, 'terminal_std_out.txt'), 'a+') as terminal_std_out:
        terminal_std_out.write(str(datetime.datetime.now()))
        terminal_std_out.write('\n\n\n')

    start_time: float = time.time()
    _ = torch_multiprocessing_parallelize(execute, parameters, num_workers=-2)
    end_time: float = time.time()
    print(f"TOTAL EXECUTION TIME (minutes): {(end_time - start_time) * (1 / 60)}")
