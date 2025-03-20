import json
import os
import math
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
from cslim.utils.logger import compute_path_run_log_and_settings


def load_value_from_run_csv(
        base_path: str,
        method: str,
        dataset_name: str,
        pop_size: int,
        n_iter: int,
        iter_cut: int,
        n_elites: int,
        pressure: int,
        slim_crossover: str,
        p_inflate: float,
        p_crossover: float,
        torus_dim: int,
        radius: int,
        cmp_rate: float,
        pop_shape: tuple[int, ...],
        seed: int,
        type_of_result: str,
        for_each_gen: bool
) -> list[float]:
    if type_of_result == 'best_overall_test_fitness':
        index_col: int = 10
    elif type_of_result == 'log_10_num_nodes':
        index_col: int = 11
    elif type_of_result == 'moran':
        index_col: int = 14
    elif type_of_result == 'training_time':
        index_col: int = 6
    else:
        raise ValueError(f'Not recognized type of results {type_of_result}.')
    
    if method.strip().lower() == 'gsgp' and type_of_result in ('best_overall_test_fitness', 'log_10_num_nodes', 'moran'):
        index_col -= 1

    if method.strip().lower() == 'gp' and type_of_result == 'moran':
        index_col += 1

    folder: str = compute_path_run_log_and_settings(
                            base_path=base_path,
                            method=method,
                            dataset_name=dataset_name,
                            pop_size=pop_size,
                            n_iter=n_iter,
                            n_elites=n_elites,
                            pressure=pressure,
                            p_crossover=p_crossover,
                            p_inflate=p_inflate,
                            slim_crossover=slim_crossover,
                            torus_dim=torus_dim,
                            radius=radius,
                            cmp_rate=cmp_rate,
                            pop_shape=pop_shape
                        )
    file: str = os.path.join(folder, f'seed{seed}_run.csv')
    try:
        f = pd.read_csv(file)
    except Exception as e:
        f = None

    if f is None:
        raise ValueError(f"{file} does not exists.")

    last_uuid = f.iloc[f.shape[0] - 1, 1]
    f = f.loc[f.iloc[:, 1] == last_uuid]
    
    try:
        if type_of_result == 'training_time' and not for_each_gen:
            f = [sum(f.iloc[:(iter_cut + 1), index_col])]
        else:
            if for_each_gen:
                f = f.iloc[:(iter_cut + 1), index_col].to_list()
            else:
                f = [f.iloc[iter_cut, index_col]]
    except Exception as e:
        print(file)
        print(e)
        print()
        f = [-1000.0]

    if type_of_result == 'log_10_num_nodes' and method.strip().lower() == 'gp':
        return [round(math.log10(float(fff)), 6) for fff in f]

    return [float(fff) for fff in f]


def create_single_result_dict(path: str, type_of_result: str, for_each_gen: bool):
    dataset_names: list[str] = ['airfoil', 'concrete', 'slump', 'parkinson', 'yacht', 'qsaraquatic']
    slim_versions: list[str] = ['SLIM+ABS', 'SLIM+SIG1', 'SLIM+SIG2']
    n_reps: int = 30

    p_inflate: float = 0.3
    p_crossover: float = 0.2
    slim_crossover: str = 'sc'

    pop_size: int = 100
    n_iter: int = 1000
    iter_cut: int = n_iter - 1
    n_elites: int = 1
    pressure: int = 4

    torus_dim: int = 2
    pop_shape: tuple[int, ...] = (int(pop_size ** 0.5), int(pop_size ** 0.5))
    all_radius: list[int] = [2, 3]
    all_cmp_rate: list[float] = [1.0]

    map_expl_pipe_to_cx_prob = {'cx': 1.0, 'mut': 0.0, 'cxmut': p_crossover}

    results = {'cx': {}, 'mut': {}, 'cxmut': {}}
    results['mut'] = {slim_version.lower(): {dataset_name.lower(): {} for dataset_name in dataset_names} for slim_version in slim_versions}
    results['cxmut'] = {slim_version.lower(): {dataset_name.lower(): {} for dataset_name in dataset_names} for slim_version in slim_versions}
    results['cx']['SLIM+SIG2'.lower()] = {dataset_name.lower(): {} for dataset_name in dataset_names}

    results['cx']['gsgp'] = {dataset_name.lower(): {} for dataset_name in dataset_names}
    results['mut']['gsgp'] = {dataset_name.lower(): {} for dataset_name in dataset_names}
    results['cxmut']['gsgp'] = {dataset_name.lower(): {} for dataset_name in dataset_names}

    results['cxmut']['gp'] = {dataset_name.lower(): {} for dataset_name in dataset_names}

    with open('../random_seeds.txt', 'r') as f:
        # THE ACTUAL SEED TO BE USED IS LOCATED AT POSITION SEED - 1 SINCE SEED IS AN INDEX THAT STARTS FROM 1
        all_actual_seeds: list[int] = [int(curr_actual_seed_as_str) for curr_actual_seed_as_str in f.readlines()]

    for expl_pipe in ['cx', 'mut', 'cxmut']:
        print('gsgp' + ' ' + expl_pipe)
        for dataset_name in dataset_names:
            print(dataset_name)
            results[expl_pipe]['gsgp'][dataset_name]['baseline'] = []
            for seed in range(1, n_reps + 1):
                v = load_value_from_run_csv(
                    base_path=path,
                    method='gsgp',
                    dataset_name=dataset_name,
                    pop_size=pop_size,
                    n_iter=n_iter,
                    iter_cut=iter_cut,
                    n_elites=n_elites,
                    pressure=pressure,
                    slim_crossover='default',
                    p_inflate=0.0,
                    p_crossover=map_expl_pipe_to_cx_prob[expl_pipe],
                    torus_dim=0,
                    radius=0,
                    cmp_rate=0.0,
                    pop_shape=(pop_size,),
                    seed=all_actual_seeds[seed - 1],
                    type_of_result=type_of_result,
                    for_each_gen=for_each_gen
                )
                if for_each_gen:
                    results[expl_pipe]['gsgp'][dataset_name]['baseline'].append(v)
                else:
                    results[expl_pipe]['gsgp'][dataset_name]['baseline'].append(v[0])

            # results[expl_pipe]['gsgp'][dataset_name]['mannwhitney'] = {}

            for radius in all_radius:
                for cmp_rate in all_cmp_rate:
                    results[expl_pipe]['gsgp'][dataset_name][f'{torus_dim}_{radius}_{cmp_rate}'] = []
                    for seed in range(1, n_reps + 1):
                        v = load_value_from_run_csv(
                            base_path=path,
                            method='gsgp',
                            dataset_name=dataset_name,
                            pop_size=pop_size,
                            n_iter=n_iter,
                            iter_cut=iter_cut,
                            n_elites=n_elites,
                            pressure=0,
                            slim_crossover='default',
                            p_inflate=0.0,
                            p_crossover=map_expl_pipe_to_cx_prob[expl_pipe],
                            torus_dim=torus_dim,
                            radius=radius,
                            cmp_rate=cmp_rate,
                            pop_shape=pop_shape,
                            seed=all_actual_seeds[seed - 1],
                            type_of_result=type_of_result,
                            for_each_gen=for_each_gen
                        )
                        if for_each_gen:
                            results[expl_pipe]['gsgp'][dataset_name][f'{torus_dim}_{radius}_{cmp_rate}'].append(v)
                        else:
                            results[expl_pipe]['gsgp'][dataset_name][f'{torus_dim}_{radius}_{cmp_rate}'].append(v[0])

                    # if not for_each_gen:
                    #     s, _ = is_mannwhitneyu_passed(
                    #         results[expl_pipe]['gsgp'][dataset_name][f'{torus_dim}_{radius}_{cmp_rate}'],
                    #         results[expl_pipe]['gsgp'][dataset_name]['baseline'],
                    #         alternative=alternative
                    #     )
                    #     results[expl_pipe]['gsgp'][dataset_name]['mannwhitney'][f'{torus_dim}_{radius}_{cmp_rate}'] = s

    for expl_pipe in ['cxmut']:
        print('gp' + ' ' + expl_pipe)
        for dataset_name in dataset_names:
            print(dataset_name)
            results[expl_pipe]['gp'][dataset_name]['baseline'] = []
            for seed in range(1, n_reps + 1):
                v = load_value_from_run_csv(
                    base_path=path,
                    method='gp',
                    dataset_name=dataset_name,
                    pop_size=pop_size,
                    n_iter=n_iter,
                    iter_cut=iter_cut,
                    n_elites=n_elites,
                    pressure=pressure,
                    slim_crossover='default',
                    p_inflate=0.0,
                    p_crossover=0.8,
                    torus_dim=0,
                    radius=0,
                    cmp_rate=0.0,
                    pop_shape=(pop_size,),
                    seed=all_actual_seeds[seed - 1],
                    type_of_result=type_of_result,
                    for_each_gen=for_each_gen
                )
                if for_each_gen:
                    results[expl_pipe]['gp'][dataset_name]['baseline'].append(v)
                else:
                    results[expl_pipe]['gp'][dataset_name]['baseline'].append(v[0])

            # results[expl_pipe]['gp'][dataset_name]['mannwhitney'] = {}

            for radius in all_radius:
                for cmp_rate in all_cmp_rate:
                    results[expl_pipe]['gp'][dataset_name][f'{torus_dim}_{radius}_{cmp_rate}'] = []
                    for seed in range(1, n_reps + 1):
                        v = load_value_from_run_csv(
                            base_path=path,
                            method='gp',
                            dataset_name=dataset_name,
                            pop_size=pop_size,
                            n_iter=n_iter,
                            iter_cut=iter_cut,
                            n_elites=n_elites,
                            pressure=0,
                            slim_crossover='default',
                            p_inflate=0.0,
                            p_crossover=0.8,
                            torus_dim=torus_dim,
                            radius=radius,
                            cmp_rate=cmp_rate,
                            pop_shape=pop_shape,
                            seed=all_actual_seeds[seed - 1],
                            type_of_result=type_of_result,
                            for_each_gen=for_each_gen
                        )
                        if for_each_gen:
                            results[expl_pipe]['gp'][dataset_name][f'{torus_dim}_{radius}_{cmp_rate}'].append(v)
                        else:
                            results[expl_pipe]['gp'][dataset_name][f'{torus_dim}_{radius}_{cmp_rate}'].append(v[0])
            
                    # if not for_each_gen:
                    #     s, _ = is_mannwhitneyu_passed(
                    #         results[expl_pipe]['gp'][dataset_name][f'{torus_dim}_{radius}_{cmp_rate}'],
                    #         results[expl_pipe]['gp'][dataset_name]['baseline'],
                    #         alternative=alternative
                    #     )
                    #     results[expl_pipe]['gp'][dataset_name]['mannwhitney'][f'{torus_dim}_{radius}_{cmp_rate}'] = s

    for expl_pipe in ['cx', 'mut', 'cxmut']:
        for slim_version in slim_versions:
            if expl_pipe == 'cx' and slim_version != 'SLIM+SIG2':
                print(slim_version.lower() + ' ' + expl_pipe + ' ' + 'skip')
                continue
            print(slim_version.lower() + ' ' + expl_pipe)
            for dataset_name in dataset_names:
                print(dataset_name)
                results[expl_pipe][slim_version.lower()][dataset_name]['baseline'] = []
                for seed in range(1, n_reps + 1):
                    v = load_value_from_run_csv(
                        base_path=path,
                        method='gsgp'+slim_version.lower(),
                        dataset_name=dataset_name,
                        pop_size=pop_size,
                        n_iter=n_iter,
                        iter_cut=iter_cut,
                        n_elites=n_elites,
                        pressure=pressure,
                        slim_crossover='default' if expl_pipe == 'mut' else slim_crossover,
                        p_inflate=p_inflate,
                        p_crossover=map_expl_pipe_to_cx_prob[expl_pipe],
                        torus_dim=0,
                        radius=0,
                        cmp_rate=0.0,
                        pop_shape=(pop_size,),
                        seed=all_actual_seeds[seed - 1],
                        type_of_result=type_of_result,
                        for_each_gen=for_each_gen
                    )
                    if for_each_gen:
                        results[expl_pipe][slim_version.lower()][dataset_name]['baseline'].append(v)
                    else:
                        results[expl_pipe][slim_version.lower()][dataset_name]['baseline'].append(v[0])

                # if compare_with_gsgp and not for_each_gen:
                #     results[expl_pipe][slim_version.lower() + '_vs_' + 'gsgp'][dataset_name]['mannwhitney'] = {}
                #
                #     s, _ = is_mannwhitneyu_passed(
                #             results[expl_pipe][slim_version.lower()][dataset_name][f'baseline'],
                #             results[expl_pipe]['gsgp'][dataset_name][f'baseline'],
                #             alternative=alternative
                #         )
                #     results[expl_pipe][slim_version.lower() + '_vs_' + 'gsgp'][dataset_name]['mannwhitney'][f'baseline'] = s
                #
                # if compare_with_gp and not for_each_gen:
                #     results[expl_pipe][slim_version.lower() + '_vs_' + 'gp'][dataset_name]['mannwhitney'] = {}
                #
                #     s, _ = is_mannwhitneyu_passed(
                #             results[expl_pipe][slim_version.lower()][dataset_name][f'baseline'],
                #             results[expl_pipe]['gp'][dataset_name][f'baseline'],
                #             alternative=alternative
                #         )
                #     results[expl_pipe][slim_version.lower() + '_vs_' + 'gp'][dataset_name]['mannwhitney'][f'baseline'] = s
                #
                # results[expl_pipe][slim_version.lower()][dataset_name]['mannwhitney'] = {}

                for radius in all_radius:
                    for cmp_rate in all_cmp_rate:
                        results[expl_pipe][slim_version.lower()][dataset_name][f'{torus_dim}_{radius}_{cmp_rate}'] = []
                        for seed in range(1, n_reps + 1):
                            v = load_value_from_run_csv(
                                base_path=path,
                                method='gsgp'+slim_version.lower(),
                                dataset_name=dataset_name,
                                pop_size=pop_size,
                                n_iter=n_iter,
                                iter_cut=iter_cut,
                                n_elites=n_elites,
                                pressure=0,
                                slim_crossover='default' if expl_pipe == 'mut' else slim_crossover,
                                p_inflate=p_inflate,
                                p_crossover=map_expl_pipe_to_cx_prob[expl_pipe],
                                torus_dim=torus_dim,
                                radius=radius,
                                cmp_rate=cmp_rate,
                                pop_shape=pop_shape,
                                seed=all_actual_seeds[seed - 1],
                                type_of_result=type_of_result,
                                for_each_gen=for_each_gen
                            )
                            if for_each_gen:
                                results[expl_pipe][slim_version.lower()][dataset_name][f'{torus_dim}_{radius}_{cmp_rate}'].append(v)
                            else:
                                results[expl_pipe][slim_version.lower()][dataset_name][f'{torus_dim}_{radius}_{cmp_rate}'].append(v[0])

                        # if not for_each_gen:
                        #     s, _ = is_mannwhitneyu_passed(
                        #         results[expl_pipe][slim_version.lower()][dataset_name][f'{torus_dim}_{radius}_{cmp_rate}'],
                        #         results[expl_pipe][slim_version.lower()][dataset_name]['baseline'],
                        #         alternative=alternative
                        #     )
                        #     results[expl_pipe][slim_version.lower()][dataset_name]['mannwhitney'][f'{torus_dim}_{radius}_{cmp_rate}'] = s
                        #
                        # if compare_with_gsgp and not for_each_gen:
                        #     s, _ = is_mannwhitneyu_passed(
                        #             results[expl_pipe][slim_version.lower()][dataset_name][f'{torus_dim}_{radius}_{cmp_rate}'],
                        #             results[expl_pipe]['gsgp'][dataset_name][f'{torus_dim}_{radius}_{cmp_rate}'],
                        #             alternative=alternative
                        #         )
                        #     results[expl_pipe][slim_version.lower() + '_vs_' + 'gsgp'][dataset_name]['mannwhitney'][f'{torus_dim}_{radius}_{cmp_rate}'] = s
                        #
                        # if compare_with_gp and not for_each_gen:
                        #     s, _ = is_mannwhitneyu_passed(
                        #             results[expl_pipe][slim_version.lower()][dataset_name][f'{torus_dim}_{radius}_{cmp_rate}'],
                        #             results[expl_pipe]['gp'][dataset_name][f'{torus_dim}_{radius}_{cmp_rate}'],
                        #             alternative=alternative
                        #         )
                        #     results[expl_pipe][slim_version.lower() + '_vs_' + 'gp'][dataset_name]['mannwhitney'][f'{torus_dim}_{radius}_{cmp_rate}'] = s

    # if not for_each_gen:
    #     if compare_with_gsgp and compare_with_gp:
    #         with open(f'complete_results_vs_gp_gsgp_p_inflate_{p_inflate_as_str}_{alternative}_{type_of_result}.json', 'w') as f:
    #             json.dump(results, f, indent=4)
    #     elif compare_with_gsgp:
    #         with open(f'complete_results_vs_gsgp_p_inflate_{p_inflate_as_str}_{alternative}_{type_of_result}.json', 'w') as f:
    #             json.dump(results, f, indent=4)
    #     elif compare_with_gp:
    #         with open(f'complete_results_vs_gp_p_inflate_{p_inflate_as_str}_{alternative}_{type_of_result}.json', 'w') as f:
    #             json.dump(results, f, indent=4)
    #     else:
    #         with open(f'complete_results_p_inflate_{p_inflate_as_str}_{alternative}_{type_of_result}.json', 'w') as f:
    #             json.dump(results, f, indent=4)
    # else:
    #     if compare_with_gsgp and compare_with_gp:
    #         with open(f'complete_results_for_each_gen_vs_gp_gsgp_p_inflate_{p_inflate_as_str}_{alternative}_{type_of_result}.json', 'w') as f:
    #             json.dump(results, f, indent=4)
    #     elif compare_with_gsgp:
    #         with open(f'complete_results_for_each_gen_vs_gsgp_p_inflate_{p_inflate_as_str}_{alternative}_{type_of_result}.json', 'w') as f:
    #             json.dump(results, f, indent=4)
    #     elif compare_with_gp:
    #         with open(f'complete_results_for_each_gen_vs_gp_p_inflate_{p_inflate_as_str}_{alternative}_{type_of_result}.json', 'w') as f:
    #             json.dump(results, f, indent=4)
    #     else:
    #         with open(f'complete_results_for_each_gen_p_inflate_{p_inflate_as_str}_{alternative}_{type_of_result}.json', 'w') as f:
    #             json.dump(results, f, indent=4)

    # if not for_each_gen:
    #     with open(f'complete_results_{type_of_result}.json', 'w') as f:
    #         json.dump(results, f, indent=4)
    # else:
    #     with open(f'complete_results_for_each_gen_{type_of_result}.json', 'w') as f:
    #         json.dump(results, f, indent=4)

    return results


def unify_results(results_dict: dict[str, dict], for_each_gen: bool):
    dataset_names: list[str] = ['airfoil', 'concrete', 'slump', 'parkinson', 'yacht', 'qsaraquatic']
    slim_versions: list[str] = ['SLIM+ABS', 'SLIM+SIG1', 'SLIM+SIG2']

    torus_dim: int = 2
    all_radius: list[int] = [2, 3]
    all_cmp_rate: list[float] = [1.0]

    map_expl_pipe_to_cx_prob = {'cx', 'mut', 'cxmut'}

    type_of_results: list[str] = ['best_overall_test_fitness', 'log_10_num_nodes', 'moran']

    if sorted(list(results_dict.keys())) != sorted(type_of_results):
        raise ValueError(f'Mismatch between keys in the results_dict ({results_dict.keys()}) and keys in the type_of_results dict ({type_of_results}).')

    data = {type_of_result: {} for type_of_result in type_of_results}
    for type_of_result in type_of_results:
        data[type_of_result] = results_dict[type_of_result]
    
    methods: list[str] = slim_versions + ['GP', 'GSGP']
    methods = [m.lower() for m in methods]
    
    for radius in all_radius:
        for cmp_rate in all_cmp_rate:
            methods.append('cgp_' + str(radius) + '_' + str(cmp_rate))
            methods.append('cgsgp_' + str(radius) + '_' + str(cmp_rate))
            for slim_version in slim_versions:
                methods.append('c' + slim_version.lower() + '_' + str(radius) + '_' + str(cmp_rate))

    values = {type_of_result: {expl_pipe: {method: {dataset_name: [] for dataset_name in dataset_names} for method in methods} for expl_pipe in map_expl_pipe_to_cx_prob} for type_of_result in type_of_results}
    
    for type_of_result in type_of_results:
        curr_data = data[type_of_result]
        for dataset_name in dataset_names:
            values[type_of_result]['cxmut']['gp'][dataset_name] = curr_data['cxmut']['gp'][dataset_name]['baseline']
            for expl_pipe in map_expl_pipe_to_cx_prob:
                values[type_of_result][expl_pipe]['gsgp'][dataset_name] = curr_data[expl_pipe]['gsgp'][dataset_name]['baseline']
            for radius in all_radius:
                for cmp_rate in all_cmp_rate:
                    values[type_of_result]['cxmut']['cgp_' + str(radius) + '_' + str(cmp_rate)][dataset_name] = curr_data['cxmut']['gp'][dataset_name][f'{torus_dim}_{radius}_{cmp_rate}']
                    for expl_pipe in map_expl_pipe_to_cx_prob:
                        values[type_of_result][expl_pipe]['cgsgp_' + str(radius) + '_' + str(cmp_rate)][dataset_name] = curr_data[expl_pipe]['gsgp'][dataset_name][f'{torus_dim}_{radius}_{cmp_rate}']

            for expl_pipe in map_expl_pipe_to_cx_prob:
                for slim_version in slim_versions:
                    if expl_pipe == 'cx' and slim_version != 'SLIM+SIG2':
                        continue
                    values[type_of_result][expl_pipe][slim_version.lower()][dataset_name] = curr_data[expl_pipe][slim_version.lower()][dataset_name]['baseline']
                    for radius in all_radius:
                        for cmp_rate in all_cmp_rate:
                            values[type_of_result][expl_pipe]['c' + slim_version.lower() + '_' + str(radius) + '_' + str(cmp_rate)][dataset_name] = curr_data[expl_pipe][slim_version.lower()][dataset_name][f'{torus_dim}_{radius}_{cmp_rate}']

    with open(f'all_values{"_for_each_gen" if for_each_gen else ""}.json', 'w') as f:
        json.dump(values, f, indent=4) # type: ignore


def main():
    path: str = f'../../cslim-DATA/ACTUAL_RESULTS/'
    type_of_results: list[str] = ['best_overall_test_fitness', 'log_10_num_nodes', 'moran']
    for for_each_gen in [False, True]:
        results_dict: dict[str, dict] = {type_of_result: create_single_result_dict(path, type_of_result, for_each_gen) for type_of_result in type_of_results}
        unify_results(results_dict, for_each_gen)


if __name__ == '__main__':
    main()
