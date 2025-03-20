import json
import statistics
import fastplot
import seaborn as sns
import numpy as np
from typing import Optional
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


def compute_pareto_fronts(all_scatter_points):
    df = pd.DataFrame(all_scatter_points, columns=['RMSE', 'Log10NNodes'])
    fronts = []

    while not df.empty:
        pareto_front = []
        for i, (rmse, log10nnodes) in enumerate(zip(df['RMSE'], df['Log10NNodes'])):
            dominated = False
            for j, (other_rmse, other_log10nnodes) in enumerate(zip(df['RMSE'], df['Log10NNodes'])):
                if (other_rmse <= rmse and other_log10nnodes <= log10nnodes) and (
                        other_rmse < rmse or other_log10nnodes < log10nnodes):
                    dominated = True
                    break
            if not dominated:
                pareto_front.append(i)

        fronts.append(df.iloc[pareto_front])
        df = df.drop(df.index[pareto_front])

    return fronts


def methods_pareto_front(
        path: str,
        algorithms: list[str],
        PLOT_ARGS: dict[str, dict],
        dataset_names: list[str],
        dataset_acronyms: dict[str, str],
        dataset: Optional[str] = None,
        grid: bool = False,
        to_normalize: bool = False
) -> None:
    with open(path, 'r') as f:
        data = json.load(f)
    
    data_rmse = data['best_overall_test_fitness']
    data_size = data['log_10_num_nodes']

    d = {"RMSE": [], "Log10NNodes": [], "Algorithm": [], "Color": [], "Marker": [], "Topology": []}
  
    all_methods = list(data_rmse.keys())
    all_datasets = list(data_rmse[all_methods[0]].keys())
    methods_alias = {'GP': 'GP', 'GSGP': 'GSGP',
                     'SLIM+SIG1': r'$\text{SLIM}^{+}_{\text{SIG1}}$', 'SLIM+SIG2': r'$\text{SLIM}^{+}_{\text{SIG2}}$', 'SLIM+ABS': r'$\text{SLIM}^{+}_{\text{ABS}}$',
                     'SLIM*SIG1': r'$\text{SLIM}^{*}_{\text{SIG1}}$', 'SLIM*SIG2': r'$\text{SLIM}^{*}_{\text{SIG2}}$', 'SLIM*ABS': r'$\text{SLIM}^{*}_{\text{ABS}}$'}
    methods_markers = {
        'GP': 'H', 'GSGP': '*',
        'SLIM+SIG1': '>', 'SLIM+SIG2': '^', 'SLIM+ABS': 'P',
        'SLIM*SIG1': '<', 'SLIM*SIG2': 'v', 'SLIM*ABS': 'X'
    }
    all_alias_keys = list(methods_alias.keys())    
    for radius in [1, 2, 3]:
        for alias in all_alias_keys:
            methods_alias['c' + alias + '_' + str(radius)] = methods_alias[alias] + '-' + '$\mathcal{T}^{2}_{' + str(radius) + '}$'
    
    if grid:
        d = {dataset_name: {"RMSE": [], "Log10NNodes": [], "Algorithm": [], "Color": [], "Marker": [], "Topology": []} for dataset_name in all_datasets}
    else:
        if dataset is not None:
            all_datasets = [dataset]
    
    all_rmse_temp = []
    for method in all_methods:
        for dataset in all_datasets:
            all_rmse_temp.extend(data_rmse[method][dataset])
    
    min_rmse, max_rmse = min(all_rmse_temp), max(all_rmse_temp)

    max_rmse_per_dataset = {}
    for dataset in all_datasets:
        per_data_rmse_temp = []
        for method in all_methods:
            per_data_rmse_temp.extend(data_rmse[method][dataset])
        max_rmse_per_dataset[dataset] = max(per_data_rmse_temp)

    for method in all_methods:
        algorithm = ''
        color = ''
        marker = ''

        if method == 'gp':
            algorithm = 'GP'
            topology = '$\mathcal{T}^{0}$'
            color = '#990000'
            marker = "H"
        elif method == 'gsgp':
            algorithm = 'GSGP'
            topology = '$\mathcal{T}^{0}$'
            color = '#990000'
            marker = '*'
        elif method.startswith('slim'):
            if method.upper() not in algorithms:
                continue
            algorithm = methods_alias[method.upper()]
            topology = '$\mathcal{T}^{0}$'
            color = '#990000'
            marker = methods_markers[method.upper()]
        elif method.startswith('c') and '_' in method:
            radius = method.split('_')[1]
            topology = '$\mathcal{T}^{2}_{' + str(radius) + '}$'
            if radius == '2':
                color = '#31AB0C'
            elif radius == '3':
                color = '#283ADF'
            elif radius == '1':
                color = '#C5F30C'

            actual_method = method.split('_')[0][1:]
            if actual_method == 'gp':
                algorithm = 'GP'
                marker = "H"
            elif actual_method == 'gsgp':
                algorithm = 'GSGP'
                marker = "*"
            elif actual_method.startswith('slim'):
                if actual_method.upper() not in algorithms:
                    continue
                algorithm = methods_alias[actual_method.upper()]
                marker = methods_markers[actual_method.upper()]

        all_vals_rmse = []
        all_vals_size = []
        if grid:
            for dataset in all_datasets:
                all_vals_rmse = []
                all_vals_size = []
                all_vals_rmse.extend(data_rmse[method][dataset])
                all_vals_size.extend(data_size[method][dataset])
                if to_normalize:
                    d[dataset]["RMSE"].append(statistics.median([this_error / max_rmse_per_dataset[dataset] for this_error in all_vals_rmse]))
                else:
                    d[dataset]["RMSE"].append(statistics.median(all_vals_rmse))
                d[dataset]["Log10NNodes"].append(statistics.median(all_vals_size))
                d[dataset]['Algorithm'].append(algorithm)
                d[dataset]['Topology'].append(topology)
                d[dataset]['Color'].append(color)
                d[dataset]['Marker'].append(marker)
        else:
            for dataset in all_datasets:
                all_vals_rmse.extend(data_rmse[method][dataset])
                all_vals_size.extend(data_size[method][dataset])      
            if to_normalize:
                d["RMSE"].append(statistics.median([this_error / max_rmse for this_error in all_vals_rmse]))
            else:
                d["RMSE"].append(statistics.median(all_vals_rmse))
            d["Log10NNodes"].append(statistics.median(all_vals_size))
            d['Algorithm'].append(algorithm)
            d['Topology'].append(topology)
            d['Color'].append(color)
            d['Marker'].append(marker)
        
    if grid:
        fastplot.plot(None, f'pareto_median_grid.pdf', mode='callback', callback=lambda plt: my_callback_scatter_grid(plt, d, 'median', methods_alias, dataset_names, dataset_acronyms, 3, 2), style='latex', **PLOT_ARGS)
    else:
        fastplot.plot(None, f'pareto_median.pdf', mode='callback', callback=lambda plt: my_callback_scatter(plt, d, 'median', methods_alias), style='latex', **PLOT_ARGS)


def my_callback_scatter(plt, d, metric, methods_alias):
    fig, ax = plt.subplots(figsize=(7, 7), layout='constrained')

    all_scatter_points = list(zip(d['RMSE'], d['Log10NNodes']))

    for rmse, log10nnodes, color, marker in zip(d['RMSE'], d['Log10NNodes'], d['Color'], d['Marker']):
        ax.scatter(rmse, log10nnodes, c=color, marker=marker, s=100, edgecolor='black', linewidth=0.8)

    pareto_fronts = compute_pareto_fronts(all_scatter_points)
    for front in pareto_fronts:
        front_sorted = front.sort_values(by='RMSE')
        ax.plot(front_sorted['RMSE'], front_sorted['Log10NNodes'], linestyle='-', linewidth=1, color='purple',
                alpha=0.5)

    ax.set_ylim(2.3, 4.5)
    ax.set_yticks([3.0, 4.0])
    # ax.set_xlim(5.0, 12.0)
    ax.tick_params(axis='both', which='both', reset=False, bottom=False, top=False, left=False, right=False)
    ax.set_xlabel(r'\rmse')
    ax.set_ylabel(r'\logNumNodes')
    # ax.set_title(f'Methods Pareto Front ({metric} across all datasets and repetitions)')
    ax.grid(True, axis='both', which='major', color='gray', linestyle='--', linewidth=0.5)


def my_callback_scatter_grid(plt, d, metric, methods_alias, dataset_names, dataset_acronyms, n, m):
    fig, ax = plt.subplots(n, m, figsize=(20, 20), layout='constrained')

    data_i = 0
    for i in range(n):
        for j in range(m):
            all_scatter_points = list(zip(d[dataset_names[data_i]]['RMSE'], d[dataset_names[data_i]]['Log10NNodes']))
            for rmse, log10nnodes, color, marker in zip(d[dataset_names[data_i]]['RMSE'],
                                                        d[dataset_names[data_i]]['Log10NNodes'],
                                                        d[dataset_names[data_i]]['Color'],
                                                        d[dataset_names[data_i]]['Marker']):
                ax[i, j].scatter(rmse, log10nnodes, c=color, marker=marker, s=100, edgecolor='black', linewidth=0.8)

            pareto_fronts = compute_pareto_fronts(all_scatter_points)
            for front in pareto_fronts:
                front_sorted = front.sort_values(by='RMSE')
                ax[i, j].plot(front_sorted['RMSE'], front_sorted['Log10NNodes'], linestyle='-', linewidth=1,
                              color='purple',
                              alpha=0.5)

            ax[i, j].set_ylim(2.0, 5.0)
            ax[i, j].set_yticks([3.0, 4.0])
            # if dataset_names[data_i] == 'concrete':
            #    ax[i, j].set_xlim(right=24)
            ax[i, j].tick_params(axis='both', which='both', reset=False, bottom=False, top=False, left=False,
                                 right=False)

            if i == n - 1:
                ax[i, j].set_xlabel(r'\rmse')
            else:
                ax[i, j].grid(True, axis='both', which='major', color='gray', linestyle='--', linewidth=0.5)
                ax[i, j].tick_params(labelbottom=True)
            if j == 0:
                ax[i, j].set_ylabel(r'\logNumNodes')
            else:
                ax[i, j].grid(True, axis='both', which='major', color='gray', linestyle='--', linewidth=0.5)
                ax[i, j].tick_params(labelleft=False)  # ax[i, j].set_yticklabels([])
            if data_i == len(dataset_names) - 1:
                ax[i, j].tick_params(pad=7)
            ax[i, j].set_title(dataset_acronyms[dataset_names[data_i]])
            ax[i, j].grid(True, axis='both', which='major', color='gray', linestyle='--', linewidth=0.5)

            data_i += 1


def create_boxplot(path: str, dataset_name: str, algorithms: list[str], type_of_result: str, palette: dict[str, str], PLOT_ARGS: dict[str, dict], all_together: bool = False, all_datasets: list[str] = []):
    with open(path, 'r') as f:
        data = json.load(f)
    
    data = data[type_of_result]
    all_methods = list(data.keys())
    metric: str = ' '.join([word[0].upper() + word[1:] for word in type_of_result.replace('_', ' ').split(' ')])
    if metric == 'Training Time':
        metric = 'TT'

    methods_alias = {'GP': 'GP', 'GSGP': 'GSGP',
                     'SLIM+SIG1': r'$\text{SLIM}^{+}_{\text{SIG1}}$', 'SLIM+SIG2': r'$\text{SLIM}^{+}_{\text{SIG2}}$', 'SLIM+ABS': r'$\text{SLIM}^{+}_{\text{ABS}}$',
                     'SLIM*SIG1': r'$\text{SLIM}^{*}_{\text{SIG1}}$', 'SLIM*SIG2': r'$\text{SLIM}^{*}_{\text{SIG2}}$', 'SLIM*ABS': r'$\text{SLIM}^{*}_{\text{ABS}}$'}
    

    d = {metric: [], "Algorithm": [], "Topology": [], "Color": []}

    for method in all_methods:

        if method == 'gp':
            algorithm = 'GP'
            topology = '$\mathcal{T}^{0}$'
            color = '#990000'
        elif method == 'gsgp':
            algorithm = 'GSGP'
            topology = '$\mathcal{T}^{0}$'
            color = '#990000'
        elif method.startswith('slim'):
            if method.upper() not in algorithms:
                continue
            algorithm = methods_alias[method.upper()]
            topology = '$\mathcal{T}^{0}$'
            color = '#990000'
        elif method.startswith('c') and '_' in method:
            radius = method.split('_')[1]
            topology = '$\mathcal{T}^{2}_{' + str(radius) + '}$'
            if radius == '2':
                color = '#31AB0C'
            elif radius == '3':
                color = '#283ADF'
            elif radius == '1':
                color = '#C5F30C'

            actual_method = method.split('_')[0][1:]
            if actual_method == 'gp':
                algorithm = 'GP'
            elif actual_method == 'gsgp':
                algorithm = 'GSGP'
            elif actual_method.startswith('slim'):
                if actual_method.upper() not in algorithms:
                    continue
                algorithm = methods_alias[actual_method.upper()]
        
        if not all_together:
            for value in data[method][dataset_name]:
                d[metric].append(value)
                d["Algorithm"].append(algorithm)
                d["Topology"].append(topology)
                d["Color"].append(color)
        else:
            for dataset_ in all_datasets:
                for value in data[method][dataset_]:
                    d[metric].append(value)
                    d["Algorithm"].append(algorithm)
                    d["Topology"].append(topology)
                    d["Color"].append(color)

    fastplot.plot(None, f'boxplot.pdf', mode='callback', callback=lambda plt: my_callback_boxplot(plt, d, metric, palette), style='latex', **PLOT_ARGS)


def my_callback_boxplot(plt, d, metric, palette):
    fig, ax = plt.subplots(figsize=(8, 8), layout='constrained')
    ax.tick_params(axis='both', which='both', reset=False, bottom=False, top=False, left=False, right=False)
    ax.grid(True, axis='both', which='major', color='gray', linestyle='--', linewidth=0.5)
    sns.boxplot(pd.DataFrame(d), x='Algorithm', y=metric, hue='Topology', palette=palette, legend=False, log_scale=10, fliersize=0.0, showfliers=False, ax=ax)


def lineplot_grid(path: str, type_of_result: str, dataset_names: list[str], dataset_acronyms: dict[str, str], num_gen: int, num_seeds: int, aggregate: bool, with_dashed_line: bool, palette: dict[str, str], PLOT_ARGS: dict[str, dict], test: bool = False):
    with open(path, 'r') as f:
        data = json.load(f)
    
    data = data[type_of_result]
    if type_of_result == 'moran':
        metric = r'\moranI'
    elif type_of_result == 'training_time':
        metric = 'TT'
    elif type_of_result == 'log_10_num_nodes':
        metric = r'\logNumNodes'
    elif type_of_result == 'best_overall_test_fitness':
        metric = r'\rmse'
    
    all_methods = list(data.keys())
    all_datasets = list(data[all_methods[0]].keys())
    methods_alias = {'GP': 'GP', 'GSGP': 'GSGP',
                     'SLIMSIG1': r'$\text{SLIM}_{\text{SIG1}}$', 'SLIMSIG2': r'$\text{SLIM}_{\text{SIG2}}$', 'SLIMABS': r'$\text{SLIM}_{\text{ABS}}$', 
                     'SLIM+SIG1': r'$\text{SLIM}^{+}_{\text{SIG1}}$', 'SLIM+SIG2': r'$\text{SLIM}^{+}_{\text{SIG2}}$', 'SLIM+ABS': r'$\text{SLIM}^{+}_{\text{ABS}}$',
                     'SLIM*SIG1': r'$\text{SLIM}^{*}_{\text{SIG1}}$', 'SLIM*SIG2': r'$\text{SLIM}^{*}_{\text{SIG2}}$', 'SLIM*ABS': r'$\text{SLIM}^{*}_{\text{ABS}}$'}

    d = {"Median": [], "Q1": [], "Q3": [], "Algorithm": [], "Topology": [], "Color": [], "Dataset": [], "Generation": []}

    for method in all_methods:
        if method == 'gp':
            algorithm = 'GP'
            topology = '$\mathcal{T}^{0}$'
            color = '#990000'
        elif method == 'gsgp':
            algorithm = 'GSGP'
            topology = '$\mathcal{T}^{0}$'
            color = '#990000'
        elif method.startswith('slim'):
            algorithm = methods_alias[method.upper()]
            topology = '$\mathcal{T}^{0}$'
            color = '#990000'
        elif method.startswith('c') and '_' in method:
            radius = method.split('_')[1]
            topology = '$\mathcal{T}^{2}_{' + str(radius) + '}$'
            if radius == '2':
                color = '#31AB0C'
            elif radius == '3':
                color = '#283ADF'
            elif radius == '1':
                color = '#C5F30C'

            actual_method = method.split('_')[0][1:]
            if actual_method == 'gp':
                algorithm = 'GP'
            elif actual_method == 'gsgp':
                algorithm = 'GSGP'
            elif actual_method.startswith('slim'):
                algorithm = methods_alias[actual_method.upper()]

        if test:

            if not aggregate:
                for dataset in all_datasets:
                    for gen in range(num_gen):
                        d['Median'].append(1)
                        d['Q1'].append(1)
                        d['Q3'].append(1)
                        d['Generation'].append(gen)
                        d['Algorithm'].append(algorithm)
                        d['Color'].append(color)
                        d['Topology'].append(topology)
                        d['Dataset'].append(dataset_acronyms[dataset])
            else:
                for gen in range(num_gen):
                    d['Median'].append(1)
                    d['Q1'].append(1)
                    d['Q3'].append(1)
                    d['Generation'].append(gen)
                    d['Algorithm'].append(algorithm)
                    d['Color'].append(color)
                    d['Topology'].append(topology)
                    d['Dataset'].append('')

        else:

            if not aggregate:
                for dataset in all_datasets:
                    all_values = data[method][dataset]
                    for gen in range(num_gen):
                        temp = []
                        for seed in range(num_seeds):
                            temp.append(all_values[seed][gen])
                        median = statistics.median(temp)
                        q1 = float(np.percentile(temp, 25))
                        q3 = float(np.percentile(temp, 75))
                        d['Median'].append(median)
                        d['Q1'].append(q1)
                        d['Q3'].append(q3)
                        d['Generation'].append(gen)
                        d['Algorithm'].append(algorithm)
                        d['Color'].append(color)
                        d['Topology'].append(topology)
                        d['Dataset'].append(dataset_acronyms[dataset])
            else:
                for gen in range(num_gen):
                    temp = []
                    for seed in range(num_seeds):
                        for dataset in all_datasets:
                            temp.append(data[method][dataset][seed][gen])
                    median = statistics.median(temp)
                    q1 = float(np.percentile(temp, 25))
                    q3 = float(np.percentile(temp, 75))
                    d['Median'].append(median)
                    d['Q1'].append(q1)
                    d['Q3'].append(q3)
                    d['Generation'].append(gen)
                    d['Algorithm'].append(algorithm)
                    d['Color'].append(color)
                    d['Topology'].append(topology)
                    d['Dataset'].append('')

    fastplot.plot(None, f'lineplot_grid_{type_of_result}.pdf', mode='callback',
                  callback=lambda plt: my_callback_lineplot_grid(plt, d, metric, num_gen=num_gen,
                                                                 methods=['SLIM+SIG2', 'SLIM*SIG2', 'SLIM+SIG1', 'SLIM*SIG1', 'SLIM+ABS', 'SLIM*ABS'],
                                                                 methods_alias=methods_alias,
                                                                 dataset_names=dataset_names,
                                                                 dataset_acronyms=dataset_acronyms,
                                                                 palette=palette, aggregate=aggregate,
                                                                 with_dashed_line=with_dashed_line, test=test), style='latex', **PLOT_ARGS)


def my_callback_lineplot_grid(plt, d, metric, num_gen, methods, methods_alias, dataset_names, dataset_acronyms, palette,
                              aggregate, with_dashed_line, test):
    if with_dashed_line:
        n, m = len(dataset_names), 2 + len(methods) // 2
    else:
        n, m = 2, 4
        figsize = (8, 4)
    figsize = (8, 8)
    if aggregate:
        n = 1
        if not with_dashed_line:
            n = 2
        dataset_names = [''] * 1000
        dataset_acronyms = {'': ''}
        figsize = (8, 4)
    d = pd.DataFrame(d)
    fig, ax = plt.subplots(n, m, figsize=figsize, layout='constrained', squeeze=False)
    x = list(range(num_gen))

    if with_dashed_line:
        methods = ['GP', 'GSGP'] + [('SLIM+ABS', 'SLIM*ABS'), ('SLIM+SIG1', 'SLIM*SIG1'), ('SLIM+SIG2', 'SLIM*SIG2')]
    else:
        methods = ['GP', 'SLIM*ABS', 'SLIM*SIG1', 'SLIM*SIG2', 'GSGP', 'SLIM+ABS', 'SLIM+SIG1', 'SLIM+SIG2']

    met_i = 0
    for i in range(n):
        for j in range(m):
            dataset = dataset_names[i]
            data_acronym = dataset_acronyms[dataset]

            algs = [methods[j]]
            if not with_dashed_line:
                algs = [methods[met_i]]
            if isinstance(algs[0], tuple):
                alg_plus, alg_times = algs[0][0], algs[0][1]
                algs = [alg_plus, alg_times]

            if not test:
                for alg in algs:
                    alg_alias = methods_alias[alg]
                    curr_d = d[(d["Dataset"] == data_acronym) & (d["Algorithm"] == alg_alias)]
                    for topology in ['$\mathcal{T}^{0}$', '$\mathcal{T}^{2}_{2}$', '$\mathcal{T}^{2}_{3}$']:
                        color = palette[topology]
                        all_med = []
                        all_q1 = []
                        all_q3 = []
                        for gen in range(num_gen):
                            all_med.append(float(
                                curr_d[(curr_d["Topology"] == topology) & (curr_d["Generation"] == gen)][
                                    "Median"].squeeze()))
                            all_q1.append(float(
                                curr_d[(curr_d["Topology"] == topology) & (curr_d["Generation"] == gen)][
                                    "Q1"].squeeze()))
                            all_q3.append(float(
                                curr_d[(curr_d["Topology"] == topology) & (curr_d["Generation"] == gen)][
                                    "Q3"].squeeze()))
                        ax[i, j].plot(x, all_med, label=f'{data_acronym}_{alg_alias}_{topology}', color=color,
                                      linestyle='--' if '*' in alg and with_dashed_line else '-',
                                      linewidth=1.4 if '*' in alg and with_dashed_line else 1.0, markersize=10)
                        ax[i, j].fill_between(x, all_q1, all_q3, color=color, alpha=0.1)

            ax[i, j].set_xlim(0, num_gen)
            ax[i, j].set_xticks([0, num_gen // 2, num_gen])

            if metric == r'\moranI':
                ax[i, j].set_ylim(-0.1, 0.3)
                ax[i, j].set_yticks([0.0, 0.2])
            elif metric == r'\logNumNodes':
                ax[i, j].set_ylim(0.5, 4.5)
                ax[i, j].set_yticks([1.0, 2.0, 3.0, 4.0])
            elif metric == 'TT':
                ax[i, j].set_ylim(-2.0, 20.0)
                ax[i, j].set_yticks([0.0, 10.0, 20.0])
            elif metric == r'\rmse':
                if dataset == 'airfoil':
                    ax[i, j].set_ylim(-6, 56)
                    ax[i, j].set_yticks([0, 25, 50])
                elif dataset == 'concrete':
                    ax[i, j].set_ylim(-4, 34)
                    ax[i, j].set_yticks([0, 15, 30])
                elif dataset == 'slump':
                    ax[i, j].set_ylim(-3, 23)
                    ax[i, j].set_yticks([0, 10, 20])
                elif dataset == 'yacht':
                    ax[i, j].set_ylim(-3, 23)
                    ax[i, j].set_yticks([0, 10, 20])
                elif dataset == 'parkinson':
                    ax[i, j].set_ylim(8.4, 13.6)
                    ax[i, j].set_yticks([9, 11, 13])
                elif dataset == 'qsaraquatic':
                    ax[i, j].set_ylim(1.0, 2.0)
                    ax[i, j].set_yticks([1.1, 1.5, 1.9])

            ax[i, j].tick_params(axis='both', which='both', reset=False, bottom=False, top=False, left=False,
                                 right=False)

            if i == n - 1:
                ax[i, j].set_xlabel('Generation')
                if i == 0:
                    ax[i, j].set_title(
                        methods_alias[algs[0].replace('+', '').replace('*', '')] if with_dashed_line else methods_alias[
                            algs[0]])
            else:
                ax[i, j].grid(True, axis='both', which='major', color='gray', linestyle='--', linewidth=0.5)
                ax[i, j].tick_params(labelbottom=False)
                ax[i, j].set_xticklabels([])
                if i == 0:
                    ax[i, j].set_title(
                        methods_alias[algs[0].replace('+', '').replace('*', '')] if with_dashed_line else methods_alias[
                            algs[0]])

            if j == 0:
                if metric != r'\rmse':
                    ax[i, j].set_ylabel(metric)
                else:
                    ax[i, j].set_ylabel(metric, labelpad=6 if dataset in ('qsaraquatic') else 10)
            else:
                ax[i, j].grid(True, axis='both', which='major', color='gray', linestyle='--', linewidth=0.5)
                ax[i, j].tick_params(labelleft=False)
                ax[i, j].set_yticklabels([])
                if j == m - 1:
                    # axttt = ax[i, j].twinx()
                    ax[i, j].set_ylabel(data_acronym, rotation=270, labelpad=14)
                    ax[i, j].yaxis.set_label_position("right")
                    ax[i, j].tick_params(labelleft=False)
                    ax[i, j].set_yticklabels([])
                    # ax[i, j].yaxis.tick_right()

            if i == n - 1 and j == m - 1:
                ax[i, j].tick_params(pad=7)

            ax[i, j].grid(True, axis='both', which='major', color='gray', linestyle='--', linewidth=0.5)
            if not with_dashed_line:
                ax[i, j].set_title(
                    methods_alias[algs[0].replace('+', '').replace('*', '')] if with_dashed_line else methods_alias[
                        algs[0]])
            met_i += 1


def main():
    path: str = f'../../cslim-DATA/GPEM/'

    preamble = r'''
                \usepackage{amsmath}
                \usepackage{libertine}
                \usepackage{xspace}

                \newcommand{\rmse}{RMSE\xspace}
                \newcommand{\moranI}{$I$\xspace}
                \newcommand{\numNodes}{$\ell$\xspace}
                \newcommand{\logNumNodes}{$\log_{10} (\ell)$\xspace}
                \newcommand{\toroid}[2]{$\mathcal{T}^{#1}_{#2}$\xspace}
                
                \newcommand{\CxMut}{$\mathcal{E}_{\text{cx,mut}}$\xspace}
                \newcommand{\Cx}{$\mathcal{E}_{\text{cx}}$\xspace}
                \newcommand{\Mut}{$\mathcal{E}_{\text{mut}}$\xspace}
                \newcommand{\explpipe}{exploration pipeline\xspace}
                
                \newcommand{\airf}{ARF\xspace}
                \newcommand{\conc}{CNC\xspace}
                \newcommand{\slum}{SLM\xspace}
                \newcommand{\toxi}{TXC\xspace}
                \newcommand{\yach}{YCH\xspace}
                \newcommand{\park}{PRK\xspace}
                \newcommand{\ista}{IST\xspace}
                \newcommand{\qsar}{QSR\xspace}
                
                \newcommand{\slimplusabs}{$\text{SLIM}^{+}_{\text{ABS}}$\xspace}
                \newcommand{\slimplussigone}{$\text{SLIM}^{+}_{\text{SIG1}}$\xspace}
                \newcommand{\slimplussigtwo}{$\text{SLIM}^{+}_{\text{SIG2}}$\xspace}
                \newcommand{\slimmulabs}{$\text{SLIM}^{*}_{\text{ABS}}$\xspace}
                \newcommand{\slimmulsigone}{$\text{SLIM}^{*}_{\text{SIG1}}$\xspace}
                \newcommand{\slimmulsigtwo}{$\text{SLIM}^{*}_{\text{SIG2}}$\xspace}
                
                \newcommand{\slimplus}{$\text{SLIM}^{+}$\xspace}
                \newcommand{\slimmul}{$\text{SLIM}^{*}$\xspace}
                \newcommand{\slimabs}{$\text{SLIM}_{\text{ABS}}$\xspace}
                \newcommand{\slimsigone}{$\text{SLIM}_{\text{SIG1}}$\xspace}
                \newcommand{\slimsigtwo}{$\text{SLIM}_{\text{SIG2}}$\xspace}
                '''

    PLOT_ARGS = {'rcParams': {'text.latex.preamble': preamble, 'pdf.fonttype': 42, 'ps.fonttype': 42}}

    palette = {'$\mathcal{T}^{2}_{' + str(1) + '}$': '#C5F30C',
               '$\mathcal{T}^{2}_{' + str(2) + '}$': '#31AB0C',
               '$\mathcal{T}^{2}_{' + str(3) + '}$': '#283ADF',
               '$\mathcal{T}^{0}$': '#990000'}

    dataset_names: list[str] = ['airfoil', 'concrete', 'slump', 'parkinson', 'yacht', 'qsaraquatic']
    slim_versions: list[str] = ['SLIM+ABS', 'SLIM+SIG1', 'SLIM+SIG2']

    dataset_acronyms = {'airfoil': r'\airf', 'concrete': r'\conc', 'parkinson': r'\park',
                        'slump': r'\slum', 'yacht': r'\yach', 'qsaraquatic': r'\qsar'}

    # create_boxplot(path=path + f'all_values.json', dataset_name='parkinson', algorithms=slim_versions, type_of_result='training_time', all_together=True, all_datasets=dataset_names, palette=palette, PLOT_ARGS=PLOT_ARGS)
    methods_pareto_front(path + f'all_values.json', grid=False, to_normalize=True, algorithms=slim_versions, dataset_names=dataset_names, dataset_acronyms=dataset_acronyms, PLOT_ARGS=PLOT_ARGS)
    #lineplot_grid(path=path + f'all_values_for_each_gen.json', test=False, type_of_result='log_10_num_nodes', dataset_names=dataset_names, dataset_acronyms=dataset_acronyms, num_gen=1000, num_seeds=30, aggregate=True, with_dashed_line=False, palette=palette, PLOT_ARGS=PLOT_ARGS)


if __name__ == '__main__':
    main()
