import random
import time

import numpy as np
import torch
from slim.algorithms.GP.representations.population import Population
from slim.algorithms.GP.representations.tree import Tree
from slim.cellular.support import create_neighbors_topology_factory, compute_all_possible_neighborhoods, weights_matrix_for_morans_I, simple_selection_process
from slim.utils.diversity import one_matrix_zero_diagonal, niche_entropy, gsgp_pop_div_from_vectors, global_moran_I
from slim.utils.logger import logger
from slim.utils.utils import verbose_reporter


class GP:
    def __init__(
        self,
        pi_init,
        initializer,
        selector,
        mutator,
        crossover,
        find_elit_func,
        p_m=0.2,
        p_xo=0.8,
        pop_size=100,
        pop_shape=(100,),
        torus_dim=0,
        radius=0,
        cmp_rate=0.0,
        pressure=2,
        seed=0,
        settings_dict=None,
    ):
        """
        Initialize the Genetic Programming algorithm.

        Args:
            pi_init (dict): Dictionary with all the parameters needed for evaluation.
            initializer (function): Function to initialize the population.
            selector (function): Function to select individuals for crossover/mutation.
            mutator (function): Function to mutate individuals.
            crossover (function): Function to perform crossover between individuals.
            find_elit_func (function): Function to find elite individuals.
            p_m (float): Probability of mutation.
            p_xo (float): Probability of crossover.
            pop_size (int): Size of the population.
            pop_shape (tuple): Shape of the grid containing the population in cellular selection (makes no sense if no cellular selection is performed).
            torus_dim (int): Dimension of the torus in cellular selection (0 if no cellular selection is performed).
            radius (int): Radius of the torus in cellular selection (makes no sense if no cellular selection is performed).
            cmp_rate (float): Competitor rate in cellular selection (makes no sense if no cellular selection is performed).
            pressure (int): The tournament size.
            seed (int): Seed for random number generation.
            settings_dict (dict): Additional settings dictionary.
        """
        self.pi_init = pi_init
        self.selector = selector
        self.p_m = p_m
        self.crossover = crossover
        self.mutator = mutator
        self.p_xo = p_xo
        self.initializer = initializer
        self.pop_size = pop_size
        self.pop_shape = pop_shape
        self.torus_dim = torus_dim
        self.radius = radius
        self.cmp_rate = cmp_rate
        self.pressure = pressure
        self.seed = seed
        self.find_elit_func = find_elit_func
        self.settings_dict = settings_dict

        Tree.FUNCTIONS = pi_init["FUNCTIONS"]
        Tree.TERMINALS = pi_init["TERMINALS"]
        Tree.CONSTANTS = pi_init["CONSTANTS"]

        self.is_cellular_selection = self.torus_dim != 0
        self.neighbors_topology_factory = create_neighbors_topology_factory(pop_size=self.pop_size, pop_shape=self.pop_shape, torus_dim=self.torus_dim, radius=self.radius, pressure=self.pressure)

    def solve(
        self,
        X_train,
        X_test,
        y_train,
        y_test,
        curr_dataset,
        n_iter=20,
        elitism=True,
        log=0,
        verbose=0,
        test_elite=False,
        log_path=None,
        run_info=None,
        max_depth=None,
        max_=False,
        ffunction=None,
        n_elites=1,
        tree_pruner=None,
        depth_calculator=None,
    ):
        """
        Execute the Genetic Programming algorithm.

        Args:
            X_train (torch.Tensor): Training data features.
            X_test (torch.Tensor): Test data features.
            y_train (torch.Tensor): Training data labels.
            y_test (torch.Tensor): Test data labels.
            curr_dataset (str): Current dataset name.
            n_iter (int): Number of iterations.
            elitism (bool): Whether to use elitism.
            log (int): Logging level.
            verbose (int): Verbosity level.
            test_elite (bool): Whether to test elite individuals.
            log_path (str): Path to save logs.
            run_info (list): Information about the current run.
            max_depth (int): Maximum depth of the tree.
            max_ (bool): Whether to maximize the fitness function.
            ffunction (function): Fitness function.
            n_elites (int): Number of elites.
            tree_pruner (function): Function to prune trees.
            depth_calculator (function): Function to calculate tree depth.
        """
        if test_elite and (X_test is None or y_test is None):
            raise Exception('If test_elite is True you need to provide a test dataset')

        torch.manual_seed(31 * self.seed ** 2 + 2056)
        np.random.seed(31 * self.seed ** 2 + 2056)
        random.seed(31 * self.seed ** 2 + 2056)

        all_possible_coordinates, all_neighborhoods_indices = compute_all_possible_neighborhoods(pop_size=self.pop_size, pop_shape=self.pop_shape, is_cellular_selection=self.is_cellular_selection, neighbors_topology_factory=self.neighbors_topology_factory)
        weights_matrix_moran = weights_matrix_for_morans_I(pop_size=self.pop_size, is_cellular_selection=self.is_cellular_selection, all_possible_coordinates=all_possible_coordinates, all_neighborhoods_indices=all_neighborhoods_indices)

        start = time.time()

        # Initialize the population
        population = Population(
            [Tree(tree) for tree in self.initializer(**self.pi_init)]
        )
        population.evaluate(ffunction, X=X_train, y=y_train)

        end = time.time()
        self.elites, self.elite = self.find_elit_func(population, n_elites)

        if test_elite:
            self.elite.evaluate(ffunction, X=X_test, y=y_test, testing=True)

        if log != 0:
            self.log_initial_population(
                population, end - start, log, log_path, run_info, weights_matrix_moran, X_train
            )

        if verbose != 0:
            verbose_reporter(
                curr_dataset.split("load_")[-1],
                0,
                self.elite.fitness,
                self.elite.test_fitness,
                end - start,
                self.elite.node_count,
            )

        for it in range(1, n_iter + 1):
            offs_pop, start = self.evolve_population(
                population,
                ffunction,
                max_depth,
                depth_calculator,
                elitism,
                X_train,
                y_train,
                all_possible_coordinates,
                all_neighborhoods_indices,
            )
            population = offs_pop
            end = time.time()

            self.elites, self.elite = self.find_elit_func(population, n_elites)

            if test_elite:
                self.elite.evaluate(ffunction, X=X_test, y=y_test, testing=True)

            if log != 0:
                self.log_generation(
                    it, population, end - start, log, log_path, run_info, weights_matrix_moran, X_train
                )

            if verbose != 0:
                verbose_reporter(
                    run_info[-1],
                    it,
                    self.elite.fitness,
                    self.elite.test_fitness,
                    end - start,
                    self.elite.node_count,
                )

    def evolve_population(
        self,
        population,
        ffunction,
        max_depth,
        depth_calculator,
        elitism,
        X_train,
        y_train,
        all_possible_coordinates,
        all_neighborhoods_indices,
    ):
        """
        Evolve the population for one generation.

        Args:
            population (Population): Current population.
            ffunction (function): Fitness function.
            max_depth (int): Maximum depth of the tree.
            depth_calculator (function): Function to calculate tree depth.
            elitism (bool): Whether to use elitism.
            X_train (torch.Tensor): Training data features.
            y_train (torch.Tensor): Training data labels.

        Returns:
            Population: Evolved population.
            float: Start time of evolution.
        """
        offs_pop = []
        start = time.time()

        if elitism:
            offs_pop.extend(self.elites)

        indexed_population = [(i, population.population[i]) for i in range(self.pop_size)]
        neighbors_topology = self.neighbors_topology_factory.create(indexed_population, clone=False)
        current_coordinate_index = 0

        while len(offs_pop) < self.pop_size:
            current_coordinate = all_possible_coordinates[current_coordinate_index]
            p1, p2 = simple_selection_process(is_cellular_selection=self.is_cellular_selection, competitor_rate=self.cmp_rate, neighbors_topology=neighbors_topology, all_neighborhoods_indices=all_neighborhoods_indices, coordinate=current_coordinate)

            if random.random() < self.p_xo:
                offs1, offs2 = self.crossover(
                    p1.repr_,
                    p2.repr_,
                    tree1_n_nodes=p1.node_count,
                    tree2_n_nodes=p2.node_count,
                )

                if max_depth is not None:
                    while (
                        depth_calculator(offs1) > max_depth
                        or depth_calculator(offs2) > max_depth
                    ):
                        offs1, offs2 = self.crossover(
                            p1.repr_,
                            p2.repr_,
                            tree1_n_nodes=p1.node_count,
                            tree2_n_nodes=p2.node_count,
                        )

                offspring = [offs1, offs2]
            else:
                offs1 = self.mutator(p1.repr_, num_of_nodes=p1.node_count)

                if max_depth is not None:
                    while depth_calculator(offs1) > max_depth:
                        offs1 = self.mutator(p1.repr_, num_of_nodes=p1.node_count)

                offspring = [offs1]

            offs_pop.extend([Tree(child) for child in offspring])
            current_coordinate_index += 1

        if len(offs_pop) > population.size:
            offs_pop = offs_pop[: population.size]

        offs_pop = Population(offs_pop)
        offs_pop.evaluate(ffunction, X=X_train, y=y_train)
        return offs_pop, start

    def log_initial_population(self, population, elapsed_time, log, log_path, run_info, weights_matrix_moran, X_train):
        """
        Log the initial population.

        Args:
            population (Population): The population to log.
            elapsed_time (float): Time taken for the process.
            log (int): Logging level.
            log_path (str): Path to save logs.
            run_info (list): Information about the current run.

        Returns:
            None
        """
        if log == 2:
            add_info = [
                self.elite.test_fitness,
                self.elite.node_count,
                float(niche_entropy([ind.repr_ for ind in population.population])),
                np.std(population.fit),
                log,
            ]
        elif log == 3:
            add_info = [
                self.elite.test_fitness,
                self.elite.node_count,
                " ".join([str(ind.node_count) for ind in population.population]),
                " ".join([str(f) for f in population.fit]),
                log,
            ]
        elif log == 4:
            add_info = [
                self.elite.test_fitness,
                self.elite.node_count,
                float(niche_entropy([ind.repr_ for ind in population.population])),
                np.std(population.fit),
                " ".join([str(ind.node_count) for ind in population.population]),
                " ".join([str(f) for f in population.fit]),
                log,
            ]
        elif log == 5:
            add_info = [
                self.elite.repr_,
                float(self.elite.fitness),
                float(self.elite.test_fitness),
                int(self.elite.node_count),
                float(niche_entropy([ind.repr_ for ind in population.population])),
                gsgp_pop_div_from_vectors(
                    torch.stack(
                        [
                            (
                                ind.train_semantics
                                if ind.train_semantics.shape != torch.Size([])
                                else ind.train_semantics.repeat(len(X_train))
                            )
                            for ind in population.population
                        ]
                    )
                ),
                global_moran_I(
                    [
                        (
                            ind.train_semantics
                            if ind.train_semantics.shape != torch.Size([])
                            else ind.train_semantics.repeat(len(X_train))
                        )
                        for ind in population.population
                    ],
                    w=one_matrix_zero_diagonal(self.pop_size)
                ),
                global_moran_I(
                    [
                        (
                            ind.train_semantics
                            if ind.train_semantics.shape != torch.Size([])
                            else ind.train_semantics.repeat(len(X_train))
                        )
                        for ind in population.population
                    ],
                    w=weights_matrix_moran
                ),
                np.mean(population.fit),
                np.std(population.fit),
                " ".join([str(ind.node_count) for ind in population.population]),
                " ".join([str(f) for f in population.fit]),
                log,
            ]
        else:
            add_info = [self.elite.test_fitness, self.elite.node_count, log]

        logger(
            log_path,
            0,
            self.elite.fitness,
            elapsed_time,
            float(population.nodes_count),
            additional_infos=add_info,
            run_info=run_info,
            seed=self.seed,
        )

    def log_generation(
        self, generation, population, elapsed_time, log, log_path, run_info, weights_matrix_moran, X_train
    ):
        """
        Log the results for the current generation.

        Args:
            generation (int): Current generation number.
            population (Population): Current population.
            elapsed_time (float): Time taken for the process.
            log (int): Logging level.
            log_path (str): Path to save logs.
            run_info (list): Information about the current run.

        Returns:
            None
        """
        if log == 2:
            add_info = [
                self.elite.test_fitness,
                self.elite.node_count,
                float(niche_entropy([ind.repr_ for ind in population.population])),
                np.std(population.fit),
                log,
            ]
        elif log == 3:
            add_info = [
                self.elite.test_fitness,
                self.elite.node_count,
                " ".join([str(ind.node_count) for ind in population.population]),
                " ".join([str(f) for f in population.fit]),
                log,
            ]
        elif log == 4:
            add_info = [
                self.elite.test_fitness,
                self.elite.node_count,
                float(niche_entropy([ind.repr_ for ind in population.population])),
                np.std(population.fit),
                " ".join([str(ind.node_count) for ind in population.population]),
                " ".join([str(f) for f in population.fit]),
                log,
            ]
        elif log == 5:
            add_info = [
                self.elite.repr_,
                float(self.elite.fitness),
                float(self.elite.test_fitness),
                int(self.elite.node_count),
                float(niche_entropy([ind.repr_ for ind in population.population])),
                gsgp_pop_div_from_vectors(
                    torch.stack(
                        [
                            (
                                ind.train_semantics
                                if ind.train_semantics.shape != torch.Size([])
                                else ind.train_semantics.repeat(len(X_train))
                            )
                            for ind in population.population
                        ]
                    )
                ),
                global_moran_I(
                    [
                        (
                            ind.train_semantics
                            if ind.train_semantics.shape != torch.Size([])
                            else ind.train_semantics.repeat(len(X_train))
                        )
                        for ind in population.population
                    ],
                    w=one_matrix_zero_diagonal(self.pop_size)
                ),
                global_moran_I(
                    [
                        (
                            ind.train_semantics
                            if ind.train_semantics.shape != torch.Size([])
                            else ind.train_semantics.repeat(len(X_train))
                        )
                        for ind in population.population
                    ],
                    w=weights_matrix_moran
                ),
                np.mean(population.fit),
                np.std(population.fit),
                " ".join([str(ind.node_count) for ind in population.population]),
                " ".join([str(f) for f in population.fit]),
                log,
            ]
        else:
            add_info = [self.elite.test_fitness, self.elite.node_count, log]

        logger(
            log_path,
            generation,
            self.elite.fitness,
            elapsed_time,
            float(population.nodes_count),
            additional_infos=add_info,
            run_info=run_info,
            seed=self.seed,
        )
