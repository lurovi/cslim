import json
import os
import statistics
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
from cslim.utils.logger import compute_path_run_log_and_settings
from cslim.utils.stats import perform_mannwhitneyu_holm_bonferroni, is_mannwhitneyu_passed, is_kruskalwallis_passed


def load_value_from_run_csv(
        base_path: str,
        method: str,
        dataset_name: str,
        pop_size: int,
        n_iter: int,
        n_iter_backup: list[int],
        iter_cut: int,
        n_elites: int,
        pressure: int,
        torus_dim: int,
        radius: int,
        cmp_rate: float,
        pop_shape: tuple[int, ...],
        seed: int,
        seed_index: int,
        type_of_result: str
) -> float:
    if type_of_result == 'best_overall_test_fitness':
        index_col: int = 10
    elif type_of_result == 'log_10_num_nodes':
        index_col: int = 11
    else:
        raise ValueError(f'Not recognized type of results {type_of_result}.')

    for n_iter_b in [n_iter] + n_iter_backup:
        folder: str = compute_path_run_log_and_settings(
                            base_path=base_path,
                            method=method,
                            dataset_name=dataset_name,
                            pop_size=pop_size,
                            n_iter=n_iter_b,
                            n_elites=n_elites,
                            pressure=pressure,
                            torus_dim=torus_dim,
                            radius=radius,
                            cmp_rate=cmp_rate,
                            pop_shape=pop_shape
                        )
        file: str = os.path.join(folder, f'seed{seed}_run.csv')
        try:
            f = pd.read_csv(file)
            break
        except:
            f = None
    
    if f is None:
        for n_iter_b in [n_iter] + n_iter_backup:
            folder: str = compute_path_run_log_and_settings(
                                base_path=base_path,
                                method=method,
                                dataset_name=dataset_name,
                                pop_size=pop_size,
                                n_iter=n_iter_b,
                                n_elites=n_elites,
                                pressure=pressure,
                                torus_dim=torus_dim,
                                radius=radius,
                                cmp_rate=cmp_rate,
                                pop_shape=pop_shape
                            )
            file: str = os.path.join(folder, f'seed{seed_index}_run.csv')
            try:
                f = pd.read_csv(file)
                break
            except:
                f = None

    if f is None:
        raise ValueError(f"{file} does not exists.")

    last_uuid = f.iloc[f.shape[0] - 1, 1]
    f = f.loc[f.iloc[:, 1] == last_uuid]
    #print(file)
    f = f.iloc[iter_cut, index_col]
    return float(f)


if __name__ == '__main__':
    p_inflate: float = 0.3
    p_inflate_as_str: str = str(p_inflate).replace(".", "d")
    path: str = f'../cslim-DATA/results_p_inflate_{p_inflate_as_str}/'

    type_of_result: str = 'best_overall_test_fitness'
    alternative: str = 'less'

    dataset_names: list[str] = ['airfoil', 'concrete', 'slump', 'parkinson', 'yacht']
    slim_versions: list[str] = ['SLIM*ABS', 'SLIM*SIG1', 'SLIM+ABS', 'SLIM+SIG1']
    n_reps: int = 30

    pop_size: int = 100
    n_iter: int = 300
    n_iter_backup: int = [500]
    n_elites: int = 1
    pressure: int = 4

    torus_dim: int = 2
    pop_shape: tuple[int, ...] = (int(pop_size ** 0.5), int(pop_size ** 0.5))
    all_radius: list[int] = [1, 2, 3]
    all_cmp_rate: list[float] = [0.6, 1.0]
    
    iter_cut: int = 300 - 1

    results = {slim_version.lower(): {dataset_name.lower(): {} for dataset_name in dataset_names} for slim_version in slim_versions}

    with open('random_seeds.txt', 'r') as f:
        # THE ACTUAL SEED TO BE USED IS LOCATED AT POSITION SEED - 1 SINCE SEED IS AN INDEX THAT STARTS FROM 1
        all_actual_seeds: list[int] = [int(curr_actual_seed_as_str) for curr_actual_seed_as_str in f.readlines()]

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
                    n_iter_backup=n_iter_backup,
                    iter_cut=iter_cut,
                    n_elites=n_elites,
                    pressure=pressure,
                    torus_dim=0,
                    radius=0,
                    cmp_rate=0.0,
                    pop_shape=(pop_size,),
                    seed=all_actual_seeds[seed - 1],
                    seed_index=seed,
                    type_of_result=type_of_result
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
                            n_iter_backup=n_iter_backup,
                            iter_cut=iter_cut,
                            n_elites=n_elites,
                            pressure=0,
                            torus_dim=torus_dim,
                            radius=radius,
                            cmp_rate=cmp_rate,
                            pop_shape=pop_shape,
                            seed=all_actual_seeds[seed - 1],
                            seed_index=seed,
                            type_of_result=type_of_result
                        )
                        results[slim_version.lower()][dataset_name][f'{torus_dim}_{radius}_{cmp_rate}'].append(v)
            
                    s, _ = is_mannwhitneyu_passed(
                        results[slim_version.lower()][dataset_name][f'{torus_dim}_{radius}_{cmp_rate}'],
                        results[slim_version.lower()][dataset_name]['baseline'],
                        alternative=alternative
                    )
                    results[slim_version.lower()][dataset_name]['mannwhitney'][f'{torus_dim}_{radius}_{cmp_rate}'] = s
    
    with open(f'complete_results_p_inflate_{p_inflate_as_str}_{alternative}_{type_of_result}.json', 'w') as f:
        json.dump(results, f, indent=4)

