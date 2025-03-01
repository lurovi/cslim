import os
import fastplot
import pandas as pd
import matplotlib.colorbar as colorbar
import matplotlib.colors as mcolors
import numpy as np
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
        torus_dim: int,
        radius: int,
        cmp_rate: float,
        pop_shape: tuple[int, ...],
        seed: int,
) -> list[float]:
    index_col: int = 18
    
    if method.strip().lower() == 'gsgp':
        index_col -= 1

    if method.strip().lower() == 'gp':
        index_col += 1

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
    file: str = os.path.join(folder, f'seed{seed}_run.csv')
    try:
        f = pd.read_csv(file)
    except:
        f = None

    if f is None:
        raise ValueError(f"{file} does not exists.")

    last_uuid = f.iloc[f.shape[0] - 1, 1]
    f = f.loc[f.iloc[:, 1] == last_uuid]
    
    try:
        f = f.iloc[iter_cut, index_col]
    except Exception as e:
        print(file)
        print(e)
        print()
        f = "-1000.0"

    return [float(fff) for fff in f.split(" ")]


def my_callback_heatmap(plt, all_matrixes, all_iters, methods_names, ylabels):
    n, m = len(all_matrixes), len(all_iters)
    fig, ax = plt.subplots(n, m, figsize=(8, 8), layout='constrained', squeeze=False)

    all_errs = []
    for met in all_matrixes:
        all_errs.extend(np.array(all_matrixes[met]).flatten().tolist())
    min_ = min(all_errs)
    max_ = np.percentile(all_errs, 90)
    print("Min: ", min_)
    print("Max: ", max_)

    met_i = 0
    for i in range(n):
        method_name = methods_names[met_i]
        iter_i = 0

        for j in range(m):
            pop = np.array(all_matrixes[method_name][iter_i]).reshape((10, 10))

            _ = ax[i, j].imshow(pop, cmap='cividis', vmin=min_, vmax=max_)
            ax[i, j].tick_params(axis='both', which='both', reset=False, bottom=False, top=False, left=False, right=False)
            ax[i, j].tick_params(labelbottom=False)
            ax[i, j].tick_params(labelleft=False)

            if i == 0:
                ax[i, j].set_title(f"Gen. {all_iters[iter_i]}" if all_iters[iter_i] <= 10 else f"Gen. {all_iters[iter_i] + 1}")

            iter_i += 1

        ax[i, 0].set_ylabel(ylabels[met_i])
        met_i += 1

def heatmap(all_matrixes, all_iters, methods_names, ylabels):
    PLOT_ARGS = {'rcParams': {'text.latex.preamble': r'\usepackage{amsmath}'}}
    fastplot.plot(None, f'heatmap.pdf', mode='callback', callback=lambda plt: my_callback_heatmap(plt, all_matrixes=all_matrixes, all_iters=all_iters, methods_names=methods_names, ylabels=ylabels), style='latex', **PLOT_ARGS)
    
def my_callback_colorbar(plt, vmin, vmax):
    fig, ax = plt.subplots(figsize=(1, 7))
    fig.subplots_adjust(bottom=0.5)

    cmap = plt.get_cmap('cividis')
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    cb = colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation='vertical')
    cb.set_label("RMSE")

def make_colorbar(vmin, vmax):
    PLOT_ARGS = {'rcParams': {'text.latex.preamble': r'\usepackage{amsmath}'}}
    fastplot.plot(None, f'colorbar.pdf', mode='callback', callback=lambda plt: my_callback_colorbar(plt, vmin, vmax), style='latex', **PLOT_ARGS)


if __name__ == '__main__':
    p_inflate: float = 0.3
    p_inflate_as_str: str = str(p_inflate).replace(".", "d")
    path: str = f'../cslim-DATA/results_p_inflate_{p_inflate_as_str}/'

    # ================================
    # GP vs GSGP
    # ================================

    dataset: str = "airfoil"
    seed_index: int = 4
    vmin = 11
    vmax = 125

    #dataset: str = "concrete"
    #seed_index: int = 20
    #vmin = 6
    #vmax = 40

    #dataset: str = "slump"
    #seed_index: int = 14
    #vmin = 0
    #vmax = 37

    #dataset: str = "parkinson"
    #seed_index: int = 6
    #vmin = 9
    #vmax = 31

    #dataset: str = "yacht"
    #seed_index: int = 20
    #vmin = 3
    #vmax = 19

    #dataset: str = "qsaraquatic"
    #seed_index: int = 6
    #vmin = 0
    #vmax = 4

    # ================================
    # SLIM*SIG1 vs SLIM+SIG1
    # ================================

    #dataset: str = "airfoil"
    #seed_index: int = 7
    #vmin = 14
    #vmax = 75

    #dataset: str = "concrete"
    #seed_index: int = 4
    #vmin = 6
    #vmax = 36

    #dataset: str = "slump"
    #seed_index: int = 2
    #vmin = 2
    #vmax = 31

    #dataset: str = "parkinson"
    #seed_index: int = 2
    #vmin = 9
    #vmax = 18

    #dataset: str = "yacht"
    #seed_index: int = 11
    #vmin = 1
    #vmax = 16

    #dataset: str = "qsaraquatic"
    #seed_index: int = 2
    #vmin = 1
    #vmax = 3

    pop_size: int = 100
    n_iter: int = 1000
    n_elites: int = 1
    pressure: int = 4

    torus_dim: int = 2
    pop_shape: tuple[int, ...] = (int(pop_size ** 0.5), int(pop_size ** 0.5))
    radius: int = 2
    cmp_rate: float = 1.0

    #methods: list[str] = ['gp', 'gsgp']
    #ylabels: list[str] = ['GP-$\mathcal{T}^{0}$', 'GP-' + '$\mathcal{T}^{2}_{' + str(radius) + '}$', 'GSGP-$\mathcal{T}^{0}$', 'GSGP-' + '$\mathcal{T}^{2}_{' + str(radius) + '}$']
    #method_names: list[str] = ['gp', 'cgp', 'gsgp', 'cgsgp']
    methods: list[str] = ['slim*sig1', 'slim+sig1']
    ylabels: list[str] = [r'$\text{SLIM}^{*}_{\text{SIG1}}$-$\mathcal{T}^{0}$', r'$\text{SLIM}^{*}_{\text{SIG1}}$-' + r'$\mathcal{T}^{2}_{' + str(radius) + r'}$', r'$\text{SLIM}^{+}_{\text{SIG1}}$-$\mathcal{T}^{0}$', r'$\text{SLIM}^{+}_{\text{SIG1}}$-' + r'$\mathcal{T}^{2}_{' + str(radius) + r'}$']
    method_names: list[str] = ['slim*sig1', 'cslim*sig1', 'slim+sig1', 'cslim+sig1']

    with open('random_seeds.txt', 'r') as f:
        # THE ACTUAL SEED TO BE USED IS LOCATED AT POSITION SEED - 1 SINCE SEED IS AN INDEX THAT STARTS FROM 1
        all_actual_seeds: list[int] = [int(curr_actual_seed_as_str) for curr_actual_seed_as_str in f.readlines()]

    all_matrixes: dict[str, list[list[float]]] = {met: [] for met in methods}
    for met in methods:
        all_matrixes['c' + met] = []
    all_iters: list[int] = [1, 5, 10, 100 - 1, 1000 - 1]

    methods_c = ['c' + met for met in methods]
    methods = methods + methods_c
    for met in methods:
        for iter_cut in all_iters:
            if not met.startswith('c'):
                v = load_value_from_run_csv(
                    base_path=path,
                    method=met if not met.startswith('slim') else 'gsgp' + met,
                    dataset_name=dataset,
                    pop_size=pop_size,
                    n_iter=n_iter,
                    iter_cut=iter_cut,
                    n_elites=n_elites,
                    pressure=pressure,
                    torus_dim=0,
                    radius=0,
                    cmp_rate=0.0,
                    pop_shape=(pop_size,),
                    seed=all_actual_seeds[seed_index - 1],
                )
            else:
                v = load_value_from_run_csv(
                    base_path=path,
                    method=met[1:] if not met[1:].startswith('slim') else 'gsgp' + met[1:],
                    dataset_name=dataset,
                    pop_size=pop_size,
                    n_iter=n_iter,
                    iter_cut=iter_cut,
                    n_elites=n_elites,
                    pressure=0,
                    torus_dim=torus_dim,
                    radius=radius,
                    cmp_rate=cmp_rate,
                    pop_shape=pop_shape,
                    seed=all_actual_seeds[seed_index - 1],
                )           
            all_matrixes[met].append(v)
    
    make_colorbar(vmin, vmax)
    heatmap(all_matrixes=all_matrixes, all_iters=all_iters, methods_names=method_names, ylabels=ylabels)