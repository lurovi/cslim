import json
import os
import statistics
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
from slim.utils.logger import compute_path_run_log_and_settings
from slim.utils.stats import perform_mannwhitneyu_holm_bonferroni, is_mannwhitneyu_passed, is_kruskalwallis_passed


def load_value_from_run_csv(
        base_path: str,
        method: str,
        dataset_name: str,
        pop_size: int,
        n_iter: int,
        n_elites: int,
        pressure: int,
        torus_dim: int,
        radius: int,
        cmp_rate: float,
        pop_shape: tuple[int, ...],
        seed: int,
        type_of_result: str
) -> float:
    folder: str = compute_path_run_log_and_settings(
        base_path=base_path,
        method=method,
        dataset_name=dataset_name,
        pop_size=pop_size,
        n_iter=n_iter,
        n_elites=n_elites,
        pressure=pressure,
        torus_dim=torus_dim,
        radius=radius,
        cmp_rate=cmp_rate,
        pop_shape=pop_shape
    )
    if type_of_result == 'best_overall_test_fitness':
        index_col: int = 10
    else:
        raise ValueError(f'Not recognized type of results {type_of_result}.')

    file: str = os.path.join(folder, f'seed{seed}_run.csv')
    f = pd.read_csv(file)
    last_uuid = f.iloc[f.shape[0] - 1, 1]
    f = f.loc[f.iloc[:, 1] == last_uuid]
    f = f.iloc[f.shape[0] - 1, index_col]
    return float(f)


if __name__ == '__main__':
    

    path: str = '../slim-DATA/results/'

    dataset_names: list[str] = ['vladislavleva4', 'keijzer6', 'nguyen7', 'pagie1', 'airfoil', 'concrete', 'slump', 'parkinson', 'yacht']
    slim_versions: list[str] = ['SLIM+SIG2', 'SLIM+ABS', 'SLIM+SIG1']
    n_reps: int = 30

    pop_size: int = 100
    n_iter: int = 300
    n_elites: int = 1
    pressure: int = 4

    torus_dim: int = 2
    pop_shape: tuple[int, ...] = (int(pop_size ** 0.5), int(pop_size ** 0.5))
    all_radius: list[int] = [2, 3]
    all_cmp_rate: list[float] = [0.6, 1.0]
    
    results = {slim_version.lower(): {dataset_name.lower(): {} for dataset_name in dataset_names} for slim_version in slim_versions}

    for slim_version in slim_versions:
        print(slim_version.lower())
        for dataset_name in dataset_names:
            print(dataset_name)
            results[slim_version.lower()][dataset_name]['baseline'] = []
            for seed in range(1, n_reps + 1):
                v = load_value_from_run_csv(
                    base_path=path,
                    method='gsgp'+slim_version.lower(),
                    dataset_name=dataset_name,
                    pop_size=pop_size,
                    n_iter=n_iter,
                    n_elites=n_elites,
                    pressure=pressure,
                    torus_dim=0,
                    radius=0,
                    cmp_rate=0.0,
                    pop_shape=(pop_size,),
                    seed=seed,
                    type_of_result='best_overall_test_fitness'
                )
                results[slim_version.lower()][dataset_name]['baseline'].append(v)

            results[slim_version.lower()][dataset_name]['mannwhitney'] = {}

            for radius in all_radius:
                for cmp_rate in all_cmp_rate:
                    results[slim_version.lower()][dataset_name][f'{torus_dim}_{radius}_{cmp_rate}'] = []
                    for seed in range(1, n_reps + 1):
                        v = load_value_from_run_csv(
                            base_path=path,
                            method='gsgp'+slim_version.lower(),
                            dataset_name=dataset_name,
                            pop_size=pop_size,
                            n_iter=n_iter,
                            n_elites=n_elites,
                            pressure=0,
                            torus_dim=torus_dim,
                            radius=radius,
                            cmp_rate=cmp_rate,
                            pop_shape=pop_shape,
                            seed=seed,
                            type_of_result='best_overall_test_fitness'
                        )
                        results[slim_version.lower()][dataset_name][f'{torus_dim}_{radius}_{cmp_rate}'].append(v)
            
                    s, _ = is_mannwhitneyu_passed(
                        results[slim_version.lower()][dataset_name][f'{torus_dim}_{radius}_{cmp_rate}'],
                        results[slim_version.lower()][dataset_name]['baseline'],
                        alternative='less'
                    )
                    results[slim_version.lower()][dataset_name]['mannwhitney'][f'{torus_dim}_{radius}_{cmp_rate}'] = s
    
    with open(f'complete_results.json', 'w') as f:
        json.dump(results, f, indent=4)

