import json
import statistics
import fastplot
import seaborn as sns
import numpy as np
from typing import Optional
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

def create_legend(label_color_dict, PLOT_ARGS, title=None):
    fastplot.plot(None, f'legend.pdf', mode='callback',
                  callback=lambda plt: create_legend_callback(plt, label_color_dict, title),
                  style='latex', **PLOT_ARGS)


def create_legend_callback(plt, label_color_dict, title=None):
    """
    Create a legend from a dictionary of labels and colors.

    Parameters:
    - label_color_dict: dict
        Dictionary where keys are labels and values are colors.
    - ax: matplotlib.axes.Axes, optional
        The axis to which the legend will be added. If None, uses the current axis.
    - loc: str, optional
        Location of the legend (default: 'upper right').
    - title: str, optional
        Title of the legend.
    """
    plt.tight_layout(pad=0)
    fig, ax = plt.subplots(figsize=(6, 1))  # Small figure with only the legend
    fig.set_size_inches(2.6, 0.3)
    ax.axis("off")  # Hide axes

    # Create legend handles
    handles = [plt.Line2D([0], [0], color=color, lw=4, label=label) for label, color in label_color_dict.items()]

    # Create and center the legend
    ax.legend(
        handles=handles,
        loc="center",
        title=title,
        ncol=len(label_color_dict),  # Arrange items in one row
        frameon=False  # Remove box around legend
    )


def create_horizontal_linestyle_legend(label_linestyle_dict, PLOT_ARGS, title=None):
    fastplot.plot(None, f'legend.pdf', mode='callback',
                  callback=lambda plt: create_horizontal_linestyle_legend_callback(plt, label_linestyle_dict, title),
                  style='latex', **PLOT_ARGS)


def create_horizontal_linestyle_legend_callback(plt, label_linestyle_dict, title=None):
    """
    Creates an image containing only a centered horizontal legend with different line styles.

    Parameters:
    - label_linestyle_dict: dict
        Dictionary where keys are labels and values are line styles.
    - title: str, optional
        Title of the legend.
    - filename: str, optional
        Name of the file to save the legend (default: 'legend.pdf').
    """
    plt.tight_layout(pad=0)
    fig, ax = plt.subplots(figsize=(len(label_linestyle_dict) * 1.5, 0.5))  # Adjust figure size dynamically
    fig.set_size_inches(2.8, 0.2)
    ax.axis("off")  # Hide axes

    # Create legend handles with different line styles
    handles = [plt.Line2D([0], [0], linestyle=linestyle, color='black', lw=2, label=label, markersize=8)
               for label, linestyle in label_linestyle_dict.items()]

    # Create and center the legend
    ax.legend(
        handles=handles,
        loc="center",
        title=title,
        ncol=len(label_linestyle_dict),  # Arrange items in one row
        frameon=False,  # Remove box around legend
        handlelength=1.7  # Reduce line length in legend
    )


def create_marker_legend(label_marker_dict, PLOT_ARGS, title=None):
    fastplot.plot(None, f'legend.pdf', mode='callback',
                  callback=lambda plt: create_marker_legend_callback(plt, label_marker_dict, title),
                  style='latex', **PLOT_ARGS)


def create_marker_legend_callback(plt, label_marker_dict, title=None):
    """
    Creates an image containing only a vertical legend with markers and minimal surrounding whitespace.

    Parameters:
    - label_marker_dict: dict
        Dictionary where keys are labels and values are marker styles.
    - title: str, optional
        Title of the legend.
    """
    plt.tight_layout(pad=0)
    fig, ax = plt.subplots(figsize=(2, len(label_marker_dict) * 0.5))  # Adjust figure size dynamically
    fig.set_size_inches(1.4, 2.6)
    ax.axis("off")  # Hide axes

    # Create legend handles with markers
    handles = [plt.Line2D([0], [0], marker=marker, color='black', linestyle='None', markersize=8, label=label)
               for label, marker in label_marker_dict.items()]

    # Create and center the legend
    legend = ax.legend(
        handles=handles,
        loc="center",
        title=title,
        frameon=False  # Remove box around legend
    )


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
        palette: dict[str, str],
        expl_pipe_alias: dict[str, str],
        methods_alias: dict[str, str],
        algorithms: list[str],
        PLOT_ARGS: dict[str, dict],
        dataset_names: list[str],
        dataset_acronyms: dict[str, str],
        dataset: Optional[str] = None,
        grid: bool = False,
        to_normalize: bool = False,
        zoom: bool = False
) -> None:
    with open(path, 'r') as f:
        data = json.load(f)
    
    data_rmse = data['best_overall_test_fitness']
    data_size = data['log_10_num_nodes']

    d = {"RMSE": [], "Log10NNodes": [], "Algorithm": [], "Color": [], "Marker": [], "Topology": []}
  
    all_methods = list(data_rmse['cxmut'].keys())
    all_datasets = list(data_rmse['cxmut'][all_methods[0]].keys())

    # EUROGP 2025
    #methods_markers = {
    #    'GP': 'H', 'GSGP': '*',
    #    'SLIM+SIG1': '>', 'SLIM+SIG2': '^', 'SLIM+ABS': 'P',
    #    'SLIM*SIG1': '<', 'SLIM*SIG2': 'v', 'SLIM*ABS': 'X'
    #}

    # GPEM 2025
    methods_markers = {
       'GP-cxmut': 'o', 'GSGP-cxmut': '*', 'GSGP-cx': 'd', 'GSGP-mut': 'D', 'SLIM+SIG2-cx': 's',
       'SLIM+SIG1-cxmut': 'v', 'SLIM+SIG2-cxmut': '^', 'SLIM+ABS-cxmut': 'X',
       'SLIM+SIG1-mut': '<', 'SLIM+SIG2-mut': '>', 'SLIM+ABS-mut': 'P',
    }

    if grid:
        d = {dataset_name: {"RMSE": [], "Log10NNodes": [], "Algorithm": [], "Color": [], "Marker": [], "Topology": []} for dataset_name in all_datasets}
    else:
        if dataset is not None:
            all_datasets = [dataset]
    
    all_rmse_temp = []
    for expl_pipe in expl_pipe_alias:
        for method in all_methods:
            for dataset in all_datasets:
                all_rmse_temp.extend(data_rmse[expl_pipe][method][dataset])

    min_rmse, max_rmse = min(all_rmse_temp), max(all_rmse_temp)

    max_rmse_per_dataset = {}
    for dataset in all_datasets:
        per_data_rmse_temp = []
        for expl_pipe in expl_pipe_alias:
            for method in all_methods:
                per_data_rmse_temp.extend(data_rmse[expl_pipe][method][dataset])
        max_rmse_per_dataset[dataset] = max(per_data_rmse_temp)

    for expl_pipe in expl_pipe_alias:
        for method in all_methods:
            temp_1, temp_2 = [], []
            for dataset in all_datasets:
                temp_1.extend(data_rmse[expl_pipe][method][dataset])
                temp_2.extend(data_size[expl_pipe][method][dataset])
            if len(temp_1) == 0 and len(temp_2) == 0:
                continue
            if expl_pipe == 'cx' and 'slim' in method:
                continue

            algorithm = ''
            color = ''
            marker = ''

            if method in ('gp', 'gsgp'):
                algorithm = methods_alias[method.upper() + '-' + expl_pipe]
                topology = r'\notoroid'
                color = palette[topology]
                marker = methods_markers[method.upper() + '-' + expl_pipe]
            elif method.startswith('slim'):
                if method.upper() not in algorithms:
                    continue
                algorithm = methods_alias[method.upper() + '-' + expl_pipe]
                topology = r'\notoroid'
                color = palette[topology]
                marker = methods_markers[method.upper() + '-' + expl_pipe]
            elif method.startswith('c') and '_' in method:
                radius = method.split('_')[1]
                topology = r'\toroid{2}{' + str(radius) + '}'
                color = palette[topology]

                actual_method = method.split('_')[0][1:]
                if actual_method in ('gp', 'gsgp'):
                    algorithm = methods_alias[actual_method.upper() + '-' + expl_pipe]
                    marker = methods_markers[actual_method.upper() + '-' + expl_pipe]
                elif actual_method.startswith('slim'):
                    if actual_method.upper() not in algorithms:
                        continue
                    algorithm = methods_alias[actual_method.upper() + '-' + expl_pipe]
                    marker = methods_markers[actual_method.upper() + '-' + expl_pipe]

            all_vals_rmse = []
            all_vals_size = []
            if grid:
                for dataset in all_datasets:
                    all_vals_rmse = []
                    all_vals_size = []
                    all_vals_rmse.extend(data_rmse[expl_pipe][method][dataset])
                    all_vals_size.extend(data_size[expl_pipe][method][dataset])
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
                    all_vals_rmse.extend(data_rmse[expl_pipe][method][dataset])
                    all_vals_size.extend(data_size[expl_pipe][method][dataset])
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
        fastplot.plot(None, f'pareto_median_grid.pdf', mode='callback', callback=lambda plt: my_callback_scatter_grid(plt, d, dataset_names, dataset_acronyms, 3, 2), style='latex', **PLOT_ARGS)
    else:
        fastplot.plot(None, f'pareto_median_{"zoom" if zoom else ""}.pdf', mode='callback', callback=lambda plt: my_callback_scatter(plt, d, zoom), style='latex', **PLOT_ARGS)


def my_callback_scatter(plt, d, zoom):
    fig, ax = plt.subplots(figsize=(10, 10), layout='constrained')

    all_scatter_points = list(zip(d['RMSE'], d['Log10NNodes']))
    if zoom:
        all_scatter_points = [(rmse, log10nnodes) for rmse, log10nnodes in all_scatter_points if 2.5 < log10nnodes < 4.5]

    for rmse, log10nnodes, color, marker in zip(d['RMSE'], d['Log10NNodes'], d['Color'], d['Marker']):
        ax.scatter(rmse, log10nnodes, c=color, marker=marker, s=100, edgecolor='black', linewidth=0.8)

    pareto_fronts = compute_pareto_fronts(all_scatter_points)
    for front in pareto_fronts:
        front_sorted = front.sort_values(by='RMSE')
        ax.plot(front_sorted['RMSE'], front_sorted['Log10NNodes'], linestyle='-', linewidth=1, color='purple',
                alpha=0.5)

    if zoom:
        ax.set_ylim(2.5, 4.5)
    else:
        ax.set_yscale('log')
    #ax.set_yticks([2, 4, 6, 8])
    #ax.set_xlim(5, 12)
    #ax.set_xticks([7, 10])
    ax.tick_params(axis='both', which='both', reset=False, bottom=False, top=False, left=False, right=False)
    ax.set_xlabel(r'\rmse', fontsize=15)
    ax.set_ylabel(r'\logNumNodes', fontsize=15)
    ax.grid(True, axis='both', which='major', color='gray', linestyle='--', linewidth=0.5)


def my_callback_scatter_grid(plt, d, dataset_names, dataset_acronyms, n, m):
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
                     'SLIM+SIG1': r'\slimplussigone', 'SLIM+SIG2': r'\slimplussigtwo', 'SLIM+ABS': r'\slimplusabs',
                     'SLIM*SIG1': r'\slimmulsigone', 'SLIM*SIG2': r'\slimmulsigtwo', 'SLIM*ABS': r'\slimmulabs'}
    

    d = {metric: [], "Algorithm": [], "Topology": [], "Color": []}

    for method in all_methods:

        if method == 'gp':
            algorithm = 'GP'
            topology = r'\notoroid'
            color = palette[topology]
        elif method == 'gsgp':
            algorithm = 'GSGP'
            topology = r'\notoroid'
            color = palette[topology]
        elif method.startswith('slim'):
            if method.upper() not in algorithms:
                continue
            algorithm = methods_alias[method.upper()]
            topology = r'\notoroid'
            color = palette[topology]
        elif method.startswith('c') and '_' in method:
            radius = method.split('_')[1]
            topology = r'\toroid{2}{' + str(radius) + '}'
            color = palette[topology]

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


def lineplot_grid(
        path: str,
        type_of_result: str,
        dataset_names: list[str],
        algorithms: list[str],
        dataset_acronyms: dict[str, str],
        num_gen: int,
        num_seeds: int,
        aggregate: bool,
        expl_pipe_alias: dict[str, str],
        methods_alias: dict[str, str],
        palette: dict[str, str],
        linestyles: dict[str, str],
        linewidths: dict[str, float],
        PLOT_ARGS: dict[str, dict],
        test: bool = False
):
    with open(path, 'r') as f:
        data = json.load(f)
    
    data = data[type_of_result]
    if type_of_result == 'moran':
        metric = r'\moranI'
    elif type_of_result == 'training_time':
        metric = r'TT'
    elif type_of_result == 'log_10_num_nodes':
        metric = r'\logNumNodes'
    elif type_of_result == 'best_overall_test_fitness':
        metric = r'\rmse'
    else:
        raise ValueError(f'Unrecognized type_of_result {type_of_result} in lineplot_grid.')
    
    all_methods = list(data['cxmut'].keys())
    all_datasets = list(data['cxmut'][all_methods[0]].keys())
    d = {"Median": [], "Q1": [], "Q3": [], "Algorithm": [],
         "ExplPipe": [], "LineStyle": [], "LineWidth": [],
         "Topology": [], "Color": [], "Dataset": [], "Generation": []}

    for expl_pipe in expl_pipe_alias:
        for method in all_methods:
            replicate_slim_cx = False
            temp_1 = []
            for dataset in all_datasets:
                temp_1.extend(data[expl_pipe][method][dataset])
            if len(temp_1) == 0:
                if expl_pipe == 'cx' and 'slim' in method:
                    replicate_slim_cx = True
                else:
                    continue

            e_pipe_alias = expl_pipe_alias[expl_pipe]
            algorithm = ''
            color = ''
            topology = ''
            linestyle = linestyles[e_pipe_alias]
            linewidth = linewidths[e_pipe_alias]

            if method in ('gp', 'gsgp'):
                algorithm = methods_alias[method.upper()]
                topology = r'\notoroid'
                color = palette[topology]
            elif method.startswith('slim'):
                if method.upper() not in algorithms:
                    continue
                algorithm = methods_alias[method.upper()]
                topology = r'\notoroid'
                color = palette[topology]
            elif method.startswith('c') and '_' in method:
                radius = method.split('_')[1]
                topology = r'\toroid{2}{' + str(radius) + '}'
                color = palette[topology]

                actual_method = method.split('_')[0][1:]
                if actual_method in ('gp', 'gsgp'):
                    algorithm = methods_alias[actual_method.upper()]
                elif actual_method.startswith('slim'):
                    if actual_method.upper() not in algorithms:
                        continue
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
                            d['ExplPipe'].append(e_pipe_alias)
                            d['LineWidth'].append(linewidth)
                            d['LineStyle'].append(linestyle)
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
                        d['ExplPipe'].append(e_pipe_alias)
                        d['LineWidth'].append(linewidth)
                        d['LineStyle'].append(linestyle)

            else:

                if not aggregate:
                    for dataset in all_datasets:
                        if replicate_slim_cx:
                            all_values = data[expl_pipe][method.replace('*sig1', '+sig2').replace('*abs', '+sig2').replace('+sig1', '+sig2').replace('+abs', '+sig2')][dataset]
                        else:
                            all_values = data[expl_pipe][method][dataset]
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
                            d['ExplPipe'].append(e_pipe_alias)
                            d['LineWidth'].append(linewidth)
                            d['LineStyle'].append(linestyle)
                else:
                    for gen in range(num_gen):
                        temp = []
                        for seed in range(num_seeds):
                            for dataset in all_datasets:
                                if replicate_slim_cx:
                                    temp.append(data[expl_pipe][method.replace('*sig1', '+sig2').replace('*abs', '+sig2').replace('+sig1', '+sig2').replace('+abs', '+sig2')][dataset][seed][gen])
                                else:
                                    temp.append(data[expl_pipe][method][dataset][seed][gen])
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
                        d['ExplPipe'].append(e_pipe_alias)
                        d['LineWidth'].append(linewidth)
                        d['LineStyle'].append(linestyle)

    fastplot.plot(None, f'lineplot_grid_{type_of_result}.pdf', mode='callback',
                  callback=lambda plt: my_callback_lineplot_grid(plt, d, metric, num_gen=num_gen,
                                                                 methods=algorithms,
                                                                 methods_alias=methods_alias,
                                                                 dataset_names=dataset_names,
                                                                 dataset_acronyms=dataset_acronyms,
                                                                 palette=palette, aggregate=aggregate,
                                                                 test=test), style='latex', **PLOT_ARGS)


def my_callback_lineplot_grid(plt, d, metric, num_gen,
                              methods, methods_alias, dataset_names, dataset_acronyms,
                              palette, aggregate, test):
    d = pd.DataFrame(d)
    n, m = len(dataset_names), 2 + len(methods)
    figsize = (8, 8)
    if aggregate: # TODO: NOT HANDLED YET
        n = len(set(list(d['LineStyle'])))
        dataset_names = [''] * 1000
        dataset_acronyms = {'': ''}
        figsize = (8, 4)
    fig, ax = plt.subplots(n, m, figsize=figsize, layout='constrained', squeeze=False)
    x = list(range(num_gen))

    all_methods = ['GP', 'GSGP'] + methods

    met_i = 0
    for i in range(n):
        for j in range(m):
            dataset = dataset_names[i]
            data_acronym = dataset_acronyms[dataset]
            alg = all_methods[j]

            if not test:
                alg_alias = methods_alias[alg]
                curr_d = d[(d["Dataset"] == data_acronym) & (d["Algorithm"] == alg_alias)]
                for topology in [r'\notoroid', r'\toroid{2}{2}', r'\toroid{2}{3}']:
                    color = palette[topology]
                    for e_pipe_alias in [r'\CxMut', r'\Cx', r'\Mut']:
                        if alg == 'GP' and e_pipe_alias != r'\CxMut':
                            continue
                        if e_pipe_alias == r'\Cx' and 'SLIM' in alg:
                            continue
                        all_med = []
                        all_q1 = []
                        all_q3 = []
                        for gen in range(num_gen):
                            very_curr_d = curr_d[(curr_d["Topology"] == topology) & (curr_d["Generation"] == gen) & (curr_d["ExplPipe"] == e_pipe_alias)]
                            all_med.append(float(very_curr_d["Median"].squeeze()))
                            all_q1.append(float(very_curr_d["Q1"].squeeze()))
                            all_q3.append(float(very_curr_d["Q3"].squeeze()))
                        ax[i, j].plot(x, all_med, label=f'{data_acronym}_{alg_alias}_{topology}_{e_pipe_alias}', color=color,
                                      linestyle=str(very_curr_d["LineStyle"].squeeze()),
                                      linewidth=float(very_curr_d["LineWidth"].squeeze()), markersize=10)
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
                else:
                    raise ValueError(f'Unknown dataset {dataset}.')
            else:
                raise ValueError(f'Unknown metric {metric}.')

            ax[i, j].tick_params(axis='both', which='both', reset=False, bottom=False, top=False, left=False,
                                 right=False)

            if i == n - 1:
                ax[i, j].set_xlabel('Generation')
                if i == 0:
                    ax[i, j].set_title(methods_alias[alg])
            else:
                ax[i, j].grid(True, axis='both', which='major', color='gray', linestyle='--', linewidth=0.5)
                ax[i, j].tick_params(labelbottom=False)
                ax[i, j].set_xticklabels([])
                if i == 0:
                    ax[i, j].set_title(methods_alias[alg])

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
                \newcommand{\notoroid}{$\mathcal{T}^{0}$\xspace}
                
                \newcommand{\CxMut}{$\mathcal{E}_{\text{cx,mut}}$\xspace}
                \newcommand{\Cx}{$\mathcal{E}_{\text{cx}}$\xspace}
                \newcommand{\Mut}{$\mathcal{E}_{\text{mut}}$\xspace}
                \newcommand{\explpipe}{exploration pipeline\xspace}
                
                \newcommand{\airf}{ARF\xspace}
                \newcommand{\conc}{CNC\xspace}
                \newcommand{\slum}{SLM\xspace}
                \newcommand{\yach}{YCH\xspace}
                \newcommand{\park}{PRK\xspace}
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

    palette = {r'\notoroid': '#990000',
               r'\toroid{2}{2}': '#31AB0C',
               r'\toroid{2}{3}': '#283ADF',
               }

    expl_pipe_linestyle = {r'\CxMut': '-', r'\Cx': '--', r'\Mut': ':'}
    expl_pipe_linewidth = {r'\CxMut': 0.7, r'\Cx': 1.5, r'\Mut': 2.3}

    dataset_names: list[str] = ['airfoil', 'concrete', 'slump', 'parkinson', 'yacht', 'qsaraquatic']
    slim_versions: list[str] = ['SLIM+ABS', 'SLIM+SIG1', 'SLIM+SIG2']

    dataset_acronyms = {'airfoil': r'\airf', 'concrete': r'\conc', 'parkinson': r'\park',
                        'slump': r'\slum', 'yacht': r'\yach', 'qsaraquatic': r'\qsar'}

    methods_markers = {
        r'\CxMut-GP': 'o', r'\CxMut-GSGP': '*', r'\Cx-GSGP': 'd', r'\Mut-GSGP': 'D',
        r'\CxMut-\slimplussigone': 'v', r'\CxMut-\slimplussigtwo': '^', r'\CxMut-\slimplusabs': 'X',
        r'\Mut-\slimplussigone': '<', r'\Mut-\slimplussigtwo': '>', r'\Mut-\slimplusabs': 'P',
    }

    methods_markers_less = {
        r'\CxMut-GP': 'o', r'\Cx-GSGP': 'd', r'\Mut-GSGP': 'D',
        r'\CxMut-\slimplussigone': 'v', r'\CxMut-\slimplussigtwo': '^', r'\CxMut-\slimplusabs': 'X',
        r'\Mut-\slimplussigone': '<', r'\Mut-\slimplussigtwo': '>', r'\Mut-\slimplusabs': 'P',
    }

    expl_pipe_alias = {'cx': r'\Cx', 'mut': r'\Mut', 'cxmut': r'\CxMut'}
    methods_alias = {'GP': 'GP', 'GSGP': 'GSGP',
                     'SLIM+': r'\slimplus', 'SLIM*': r'\slimmul',
                     'SLIM+SIG1': r'\slimplussigone', 'SLIM+SIG2': r'\slimplussigtwo', 'SLIM+ABS': r'\slimplusabs',
                     'SLIM*SIG1': r'\slimmulsigone', 'SLIM*SIG2': r'\slimmulsigtwo', 'SLIM*ABS': r'\slimmulabs'}
    for radius in [1, 2, 3]:
        for alias in list(methods_alias.keys()):
            methods_alias['c' + alias + '_' + str(radius)] = methods_alias[alias] + '-' + r'\toroid{2}{' + str(
                radius) + '}'
    for expl_pipe in expl_pipe_alias:
        for alias in list(methods_alias.keys()):
            methods_alias[alias + '-' + expl_pipe] = expl_pipe_alias[expl_pipe] + '-' + methods_alias[alias]

    #create_legend(palette, PLOT_ARGS=PLOT_ARGS)
    #create_horizontal_linestyle_legend(expl_pipe_linestyle, PLOT_ARGS=PLOT_ARGS)
    #create_marker_legend(methods_markers, PLOT_ARGS=PLOT_ARGS)

    #create_boxplot(path=path + f'all_values.json', palette=palette, dataset_name='parkinson', algorithms=slim_versions, type_of_result='training_time', all_together=True, all_datasets=dataset_names, palette=palette, PLOT_ARGS=PLOT_ARGS)

    for zoom in [False, True]:
        methods_pareto_front(path + f'all_values.json', expl_pipe_alias=expl_pipe_alias, methods_alias=methods_alias, palette=palette, zoom=zoom, grid=False, to_normalize=True, algorithms=slim_versions, dataset_names=dataset_names, dataset_acronyms=dataset_acronyms, PLOT_ARGS=PLOT_ARGS)
    #for type_of_result in ['best_overall_test_fitness', 'log_10_num_nodes', 'moran']:
    #    lineplot_grid(path=path + f'all_values_for_each_gen.json', test=False, type_of_result=type_of_result, algorithms=slim_versions, dataset_names=dataset_names, dataset_acronyms=dataset_acronyms, expl_pipe_alias=expl_pipe_alias, methods_alias=methods_alias, num_gen=1000, num_seeds=30, aggregate=False, palette=palette, linestyles=expl_pipe_linestyle, linewidths=expl_pipe_linewidth, PLOT_ARGS=PLOT_ARGS)


if __name__ == '__main__':
    main()
