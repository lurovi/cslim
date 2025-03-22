import json
import statistics
from collections import defaultdict
from copy import deepcopy

from cslim.utils.stats import perform_mannwhitneyu_holm_bonferroni


def fill_single_macro_row_of_stat_table(
        data,
        table_string,
        macro_row_name_alias,
        dataset_names,
        expl_pipe,
        variants,
        variants_alias,
        mann_asterisk_criterion,
        swap_expl_pipe_and_meth=False
):
    FIRST_ONE_MANN = 'first_one'
    AT_LEAST_ONE_MANN = 'at_least_one'
    if mann_asterisk_criterion not in (FIRST_ONE_MANN, AT_LEAST_ONE_MANN):
        raise ValueError(f'Invalid mann_asterisk_criterion found: {mann_asterisk_criterion}.')

    # =============
    # Computing necessary data for printing the values in this section
    # =============

    # metric variant dataset value
    medians = {'rmse': defaultdict(dict), 'size': defaultdict(dict)}
    # metric dataset variants
    lowest_median = {'rmse': dict(), 'size': dict()}
    # metric dataset variants
    mann = {'rmse': dict(), 'size': dict()}
    # metric dataset variant
    holm = {'rmse': dict(), 'size': dict()}

    for metric in ['rmse', 'size']:
        for dataset in dataset_names:
            var_dict = {}
            for var in variants:
                l = data[metric][expl_pipe.lower()][var.lower()][dataset] if not swap_expl_pipe_and_meth else data[metric][var.lower()][expl_pipe.lower()][dataset]
                medians[metric][var][dataset] = statistics.median(l)
                var_dict[var] = l

            m = min([medians[metric][var][dataset] for var in variants])
            lowest_median[metric][dataset] = [var for var in variants if medians[metric][var][dataset] == m]

            h_dict, m_dict = perform_mannwhitneyu_holm_bonferroni(var_dict, alternative='less')
            holm[metric][dataset] = None
            for key in h_dict:
                if h_dict[key]:
                    holm[metric][dataset] = key
            mann[metric][dataset] = []
            for var in variants:
                if mann_asterisk_criterion == FIRST_ONE_MANN and m_dict[var][variants[0]]:
                    mann[metric][dataset].append(var)
                elif mann_asterisk_criterion == AT_LEAST_ONE_MANN and any([m_dict[var][var_2] for var_2 in variants]):
                    mann[metric][dataset].append(var)


    # =============
    # Actually including the values in the section of the table
    # =============

    is_first_row = True
    for var in variants:
        if is_first_row:
            table_string += r'\multirow{' + str(len(variants)) + '}{*}{' + macro_row_name_alias + '} & '
            is_first_row = False
        else:
            table_string += ' & '
        table_string += variants_alias[var] + ' & '
        for metric in ['rmse', 'size']:
            for dataset in dataset_names:
                table_string += r'\bfseries ' if var in lowest_median[metric][dataset] else ''
                table_string += str(round(medians[metric][var][dataset], 2))
                if var == holm[metric][dataset]:
                    table_string += r'{$^{\scalebox{0.90}{\textbf{\color{blue}*}}}$}'
                elif var in mann[metric][dataset]:
                    table_string += r'{$^{\scalebox{0.90}{\textbf{\color{black}*}}}$}'
                table_string += ' & '
            if metric == 'rmse':
                table_string += '{} & '
            else:
                table_string = table_string[:-2]
                table_string += r'\\' + '\n'
    table_string += r'\midrule' + '\n'

    return table_string


def print_table_cellular_methods_per_algorithm_with_expl_pipe_section(path, expl_pipelines, expl_pipe_alias, algorithms, algorithms_alias, dataset_names):
    with open(path, 'r') as f:
        data = json.load(f)

    data = {'rmse': data['best_overall_test_fitness'], 'size': data['log_10_num_nodes']}

    table_string = r"\midrule" + '\n'

    for expl_pipe in expl_pipelines:
        table_string +=  r"\multicolumn{" + str(len(dataset_names) * 2 + 3) + "}{r}{" + expl_pipe_alias[expl_pipe] + "}" + r"\\" + '\n' + r"\midrule" + '\n'
        for algorithm in algorithms:
            if algorithm == 'GP' and expl_pipe != 'cxmut':
                continue
            if expl_pipe == 'cx' and 'SLIM' in algorithm:
                continue

            variants = [algorithm, 'c' + algorithm + '_2_1.0', 'c' + algorithm + '_3_1.0']
            variants_alias = {algorithm: r'\notoroid',
                              'c' + algorithm + '_2_1.0': r'\toroid{2}{2}',
                              'c' + algorithm + '_3_1.0': r'\toroid{2}{3}'}

            table_string = fill_single_macro_row_of_stat_table(
                data=data,
                table_string=table_string,
                macro_row_name_alias=algorithms_alias[algorithm],
                dataset_names=dataset_names,
                expl_pipe=expl_pipe,
                variants=variants,
                variants_alias=variants_alias,
                mann_asterisk_criterion='at_least_one'
            )

    print(table_string)


def print_table_algorithms_per_expl_pipe_with_cellular_section(path, expl_pipelines, expl_pipe_alias, algorithms, algorithms_alias, dataset_names):
    with open(path, 'r') as f:
        data = json.load(f)

    data = {'rmse': data['best_overall_test_fitness'], 'size': data['log_10_num_nodes']}

    cellular_toroids = [r'\notoroid', r'\toroid{2}{2}', r'\toroid{2}{3}']

    table_string = r"\midrule" + '\n'

    for c_toroid in cellular_toroids:
        table_string +=  r"\multicolumn{" + str(len(dataset_names) * 2 + 3) + "}{r}{" + c_toroid + "}" + r"\\" + '\n' + r"\midrule" + '\n'
        for expl_pipe in expl_pipelines:
            variants = []
            variants_alias = deepcopy(algorithms_alias)
            for var_al in variants_alias:
                if '-' in variants_alias[var_al]:
                    variants_alias[var_al] = variants_alias[var_al].split('-')[0]

            for algorithm in algorithms:
                if algorithm == 'GP' and expl_pipe != 'cxmut':
                    continue
                if expl_pipe == 'cx' and 'SLIM' in algorithm:
                    continue
                if c_toroid == r'\notoroid':
                    variants.append(algorithm)
                elif c_toroid == r'\toroid{2}{2}':
                    variants.append('c' + algorithm + '_2_1.0')
                elif c_toroid == r'\toroid{2}{3}':
                    variants.append('c' + algorithm + '_3_1.0')

            table_string = fill_single_macro_row_of_stat_table(
                data=data,
                table_string=table_string,
                macro_row_name_alias=expl_pipe_alias[expl_pipe],
                dataset_names=dataset_names,
                expl_pipe=expl_pipe,
                variants=variants,
                variants_alias=variants_alias,
                mann_asterisk_criterion='at_least_one'
            )

    print(table_string)


def print_table_expl_pipe_per_cellular_with_algorithm_section(path, expl_pipelines, expl_pipe_alias, algorithms, algorithms_alias, dataset_names):
    with open(path, 'r') as f:
        data = json.load(f)

    data = {'rmse': data['best_overall_test_fitness'], 'size': data['log_10_num_nodes']}

    cellular_toroids = [r'\notoroid', r'\toroid{2}{2}', r'\toroid{2}{3}']

    table_string = r"\midrule" + '\n'

    for algorithm in algorithms:
        table_string +=  r"\multicolumn{" + str(len(dataset_names) * 2 + 3) + "}{r}{" + algorithms_alias[algorithm] + "}" + r"\\" + '\n' + r"\midrule" + '\n'
        for c_toroid in cellular_toroids:
            actual_meth = ''
            if c_toroid == r'\notoroid':
                actual_meth = algorithm
            elif c_toroid == r'\toroid{2}{2}':
                actual_meth = 'c' + algorithm + '_2_1.0'
            elif c_toroid == r'\toroid{2}{3}':
                actual_meth = 'c' + algorithm + '_3_1.0'
            variants = expl_pipelines
            variants_alias = expl_pipe_alias

            if 'slim' in algorithm.lower():
                variants = ['cxmut', 'mut']

            table_string = fill_single_macro_row_of_stat_table(
                data=data,
                table_string=table_string,
                macro_row_name_alias=c_toroid,
                dataset_names=dataset_names,
                expl_pipe=actual_meth.lower(),
                variants=variants,
                variants_alias=variants_alias,
                mann_asterisk_criterion='at_least_one',
                swap_expl_pipe_and_meth=True
            )

    print(table_string)


def main():
    path: str = f'../../cslim-DATA/GPEM/'

    dataset_names: list[str] = ['airfoil', 'concrete', 'slump', 'parkinson', 'yacht', 'qsaraquatic']
    slim_versions: list[str] = ['SLIM+ABS', 'SLIM+SIG1', 'SLIM+SIG2']

    expl_pipe_alias = {'cx': r'\Cx', 'mut': r'\Mut', 'cxmut': r'\CxMut'}
    methods_alias = {'GP': 'GP', 'GSGP': 'GSGP',
                     'SLIM+': r'\slimplus', 'SLIM*': r'\slimmul',
                     'SLIM+SIG1': r'\slimplussigone', 'SLIM+SIG2': r'\slimplussigtwo', 'SLIM+ABS': r'\slimplusabs',
                     'SLIM*SIG1': r'\slimmulsigone', 'SLIM*SIG2': r'\slimmulsigtwo', 'SLIM*ABS': r'\slimmulabs'}
    for radius in [1, 2, 3]:
        for alias in list(methods_alias.keys()):
            methods_alias['c' + alias + '_' + str(radius)] = methods_alias[alias] + '-' + r'\toroid{2}{' + str(radius) + '}'
            methods_alias['c' + alias + '_' + str(radius) + '_' + '1.0'] = methods_alias[alias] + '-' + r'\toroid{2}{' + str(radius) + '}'
    for expl_pipe in expl_pipe_alias:
        for alias in list(methods_alias.keys()):
            methods_alias[alias + '-' + expl_pipe] = expl_pipe_alias[expl_pipe] + '-' + methods_alias[alias]

    print_table_cellular_methods_per_algorithm_with_expl_pipe_section(path=path + 'all_values.json', expl_pipelines=['cxmut', 'cx', 'mut'],
                                               expl_pipe_alias=expl_pipe_alias,
                                               algorithms=['GP', 'GSGP'] + slim_versions,
                                               algorithms_alias=methods_alias,
                                               dataset_names=dataset_names)

    # print_table_algorithms_per_expl_pipe_with_cellular_section(path=path + 'all_values.json',
    #                                                                   expl_pipelines=['cxmut', 'mut'],
    #                                                                   expl_pipe_alias=expl_pipe_alias,
    #                                                                   algorithms=['GP', 'GSGP'] + slim_versions,
    #                                                                   algorithms_alias=methods_alias,
    #                                                                   dataset_names=dataset_names)

    # print_table_expl_pipe_per_cellular_with_algorithm_section(path=path + 'all_values.json',
    #                                                                expl_pipelines=['cxmut', 'cx', 'mut'],
    #                                                                expl_pipe_alias=expl_pipe_alias,
    #                                                                algorithms=['GSGP'] + slim_versions,
    #                                                                algorithms_alias=methods_alias,
    #                                                                dataset_names=dataset_names)

if __name__ == '__main__':
    main()

