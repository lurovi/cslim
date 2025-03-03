import csv
import math
import os.path
from copy import copy
from uuid import UUID
import pandas as pd
import os
import re


def is_valid_filename(filename: str) -> bool:
    """Check if a given filename is valid for the current OS."""

    # Check if the filename is empty or too long
    if not filename or len(filename) > 255:
        return False

    # Forbidden characters
    forbidden_chars = r'[<>:"/\\|?*]'

    # Check for forbidden characters
    if re.search(forbidden_chars, filename):
        return False

    # Reserved filenames (case-insensitive check)
    reserved_names = {
        "CON", "PRN", "AUX", "NUL",
        "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
        "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9"
    }
    if filename.split('.')[0].upper() in reserved_names:
        return False

    return True


def compute_path_run_log_and_settings(
        base_path: str,
        method: str,
        dataset_name: str,
        pop_size: int,
        n_iter: int,
        n_elites: int,
        pressure: int,
        p_crossover: float,
        p_inflate: float,
        slim_crossover: str,
        torus_dim: int,
        radius: int,
        cmp_rate: float,
        pop_shape: tuple[int, ...]
) -> str:
    s: str = base_path.strip()

    s = os.path.join(
        s,
        f'{slim_crossover}_pxo{str(round(p_crossover, 3)).replace(".", "d")}pinf{str(round(p_inflate, 3)).replace(".", "d")}',
        f'{method.lower().strip()}_{dataset_name.lower().strip()}',
        f'pop{pop_size}gen{n_iter}elites{n_elites}shape{"x".join([str(n) for n in pop_shape])}',
        f'pressure{pressure}torus{torus_dim}radius{radius}cmprate{str(round(cmp_rate, 3)).replace(".", "d")}',
        ''
    )

    return s


def log_settings(path: str, settings_dict: list, unique_run_id: UUID) -> None:
    """
    Log the settings to a CSV file.

    Args:
        path (str): Path to the CSV file.
        settings_dict (dict): Dictionary of settings.
        unique_run_id (str): Unique identifier for the run.

    Returns:
        None
    """
    if not os.path.isdir('/'.join(path.split('/')[:-1])):
        os.makedirs('/'.join(path.split('/')[:-1]), exist_ok=True)
    settings_dict = merge_settings(*settings_dict)
    del settings_dict["TERMINALS"]

    infos = [unique_run_id, settings_dict]

    with open(path, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(infos)


def merge_settings(sd1: dict, sd2: dict, sd3: dict, sd4: dict) -> dict:
    """
    Merge multiple settings dictionaries into one.

    Args:
        sd1 (dict): First settings dictionary.
        sd2 (dict): Second settings dictionary.
        sd3 (dict): Third settings dictionary.
        sd4 (dict): Fourth settings dictionary.

    Returns:
        dict: Merged settings dictionary.
    """
    return {**sd1, **sd2, **sd3, **sd4}


def logger(
    path: str,
    generation: int,
    pop_val_fitness: float,
    timing: float,
    nodes: int,
    additional_infos: list = None,
    run_info: list = None,
    seed: int = 0,
    scale_nodes_with_log10: bool = False
) -> None:
    """
    Logs information into a CSV file.

    Args:
        path (str): Path containing the log.
        generation (int): Current generation number.
        pop_val_fitness (float): Population's validation fitness value.
        timing (float): Time taken for the process.
        nodes (int): Count of nodes in the population.
        additional_infos (list, optional): Population's test fitness value(s) and diversity measurements. Defaults to None.
        run_info (list, optional): Information about the run. Defaults to None.
        seed (int, optional): The seed used in random, numpy, and torch libraries. Defaults to 0.
        scale_nodes_with_log10 (bool, optional): If True, a log10 is applied to each number of nodes to avoid getting large numbers and possibly overflow (default is False).

    Returns:
        None
    """
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, f"seed{seed}_run.csv"), "a", newline="") as file:
        writer = csv.writer(file)
        infos = copy(run_info) if run_info is not None else []
        nodes_ = round(math.log10(nodes), 6) if scale_nodes_with_log10 else nodes
        infos.extend([seed, generation, float(pop_val_fitness), timing, nodes_])

        if additional_infos is not None:
            infos.extend(additional_infos)

        writer.writerow(infos)


def drop_experiment_from_logger(experiment_id: str | int, log_path: str) -> None:
    """
    Remove an experiment from the logger CSV file. If the given experiment_id is -1, the last saved experiment is removed.

    Args:
        experiment_id (str or int): The experiment id to be removed. If -1, the most recent experiment is removed.
        log_path (str): Path to the file containing the logging information.

    Returns:
        None
    """
    logger_data = pd.read_csv(log_path)

    # If we choose to remove the last stored experiment
    if experiment_id == -1:
        # Find the experiment id of the last row in the CSV file
        experiment_id = logger_data.iloc[-1, 1]

    # Exclude the logger data with the chosen id
    to_keep = logger_data[logger_data.iloc[:, 1] != experiment_id]
    # Save the new excluded dataset
    to_keep.to_csv(log_path, index=False, header=None)
