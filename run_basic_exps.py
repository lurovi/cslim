from slim.datasets.data_loader import read_csv_data
from slim.utils.parallel import torch_multiprocessing_parallelize
from slim.main_slim import slim
import os
import datetime
import torch.multiprocessing as mp
import torch


def execute(path: str, dataset_name: str, slim_version: str, seed: int) -> None:
    pop_size: int = 100
    n_iter: int = 200
    n_elites: int = 1
    pressure: int = 4

    torus_dim: int = 2
    pop_shape: tuple[int, ...] = (int(pop_size ** 0.5), int(pop_size ** 0.5))
    all_radius: list[int] = [2, 3]
    all_cmp_rate: list[float] = [0.6, 1.0]

    d = read_csv_data('slim/datasets/data_csv/', dataset_name, seed)
    X_train = d['train'][0]
    y_train = d['train'][1]
    X_test = d['test'][0]
    y_test = d['test'][1]
    
    slim(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, dataset_name=dataset_name,
            slim_version=slim_version, pop_size=pop_size, n_iter=n_iter, elitism=True, n_elites=n_elites,
            pressure=pressure, torus_dim=0, radius=0, cmp_rate=0.0, pop_shape=(pop_size,),
            log_path=path, seed=seed)
    for radius in all_radius:
        for cmp_rate in all_cmp_rate:
            slim(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, dataset_name=dataset_name,
                slim_version=slim_version, pop_size=pop_size, n_iter=n_iter, elitism=True, n_elites=n_elites,
                pressure=0, torus_dim=torus_dim, radius=radius, cmp_rate=cmp_rate, pop_shape=pop_shape,
                log_path=path, seed=seed)

    verbose_output: str = f'SEED{seed} SlimVersion {slim_version} PopSize {pop_size} NIter {n_iter} NElites {n_elites} Pressure {pressure} Dataset {dataset_name} TorusDim {torus_dim} Radius {str(all_radius)} CmpRate {str(all_cmp_rate)} PopShape {str(pop_shape)}'
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

    path: str = 'results/'

    dataset_names: list[str] = ['vladislavleva4', 'keijzer6', 'nguyen7', 'pagie1', 'airfoil', 'concrete', 'slump', 'parkinson', 'yacht', 'qsaraquatic']
    slim_versions: list[str] = ['SLIM+SIG2', 'SLIM+ABS', 'SLIM+SIG1']
    n_reps: int = 10
    
    parameters = []

    for dataset_name in dataset_names:
        for seed in range(1, n_reps + 1):
            for slim_version in slim_versions:
                parameters.append({'path': path, 'dataset_name': dataset_name, 'slim_version': slim_version, 'seed': seed})

    with open(os.path.join(path, 'terminal_std_out.txt'), 'a+') as terminal_std_out:
        terminal_std_out.write(str(datetime.datetime.now()))
        terminal_std_out.write('\n\n\n')

    _ = torch_multiprocessing_parallelize(execute, parameters, num_workers=-2)
