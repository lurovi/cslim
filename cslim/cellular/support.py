import random
from cslim.cellular.factory.NeighborsTopologyFactory import NeighborsTopologyFactory
from cslim.cellular.factory.TournamentTopologyFactory import TournamentTopologyFactory
from cslim.cellular.factory.RowMajorMatrixFactory import RowMajorMatrixFactory
from cslim.cellular.factory.RowMajorLineFactory import RowMajorLineFactory
from cslim.cellular.factory.RowMajorCubeFactory import RowMajorCubeFactory
from cslim.cellular.NeighborsTopology import NeighborsTopology
from cslim.utils.diversity import zero_matrix, one_matrix_zero_diagonal
import itertools


def create_neighbors_topology_factory(
        pop_size: int,
        pop_shape: tuple[int, ...],
        torus_dim: int,
        radius: int,
        pressure: int
) -> NeighborsTopologyFactory:
    
    if torus_dim == 0:
        return TournamentTopologyFactory(pressure=pressure)
    else:
        if len(pop_shape) != torus_dim:
            raise ValueError(f'{torus_dim} is the torus_dim but the length of pop_shape is {len(pop_shape)}, there is a mismatch, check the provided pop_shape {str(pop_shape)}.')

        tmp: int = 1
        for n in pop_shape:
            tmp *= n
        if tmp != pop_size:
            raise ValueError(f'{pop_size} is the pop_size but the multiplication of all the elements in pop_shape {str(pop_shape)} is {tmp}, which is different from the pop_size.')

        if torus_dim == 2:
            return RowMajorMatrixFactory(n_rows=pop_shape[0], n_cols=pop_shape[1], radius=radius)
        elif torus_dim == 3:
            return RowMajorCubeFactory(n_channels=pop_shape[0], n_rows=pop_shape[1], n_cols=pop_shape[2], radius=radius)
        elif torus_dim == 1:
            return RowMajorLineFactory(radius=radius)
        else:
            raise ValueError(f'{torus_dim} is not a valid torus dimension.')


def compute_all_possible_neighborhoods(
        pop_size: int,
        pop_shape: tuple[int, ...],
        is_cellular_selection: bool,
        neighbors_topology_factory: NeighborsTopologyFactory
) -> tuple[list[tuple[int, ...]], dict[tuple[int, ...], list[tuple[int, ...]]]]:
    if len(pop_shape) > 1 and is_cellular_selection:
        all_possible_coordinates: list[tuple[int, ...]] = [elem for elem in itertools.product(*[list(range(s)) for s in pop_shape])]
    else:
        all_possible_coordinates: list[tuple[int, ...]] = [(i,) for i in range(pop_size)]

    neigh_top_indices: NeighborsTopology = neighbors_topology_factory.create(all_possible_coordinates, clone=False)
    all_neighborhoods_indices: dict[tuple[int, ...], list[tuple[int, ...]]] = {}
    
    if is_cellular_selection:
        for coordinate in all_possible_coordinates:
            curr_neighs: list[tuple[int, ...]] = neigh_top_indices.neighborhood(coordinate, include_current_point=True, clone=False, distinct_coordinates=True)
            all_neighborhoods_indices[coordinate] = curr_neighs
    
    return all_possible_coordinates, all_neighborhoods_indices


def weights_matrix_for_morans_I(
        pop_size: int,
        is_cellular_selection: bool,
        all_possible_coordinates: list[tuple[int, ...]],
        all_neighborhoods_indices: dict[tuple[int, ...], list[tuple[int, ...]]]
) -> list[list[float]]:

    weights_matrix_moran: list[list[float]] = zero_matrix(pop_size)
    if is_cellular_selection:
        for i in range(pop_size):
            coordinate_i: tuple[int, ...] = all_possible_coordinates[i]
            neigh_indices_of_i: list[tuple[int, ...]] = all_neighborhoods_indices[coordinate_i]
            for j in range(pop_size):
                coordinate_j: tuple[int, ...] = all_possible_coordinates[j]
                if i != j and coordinate_j in neigh_indices_of_i:
                    weights_matrix_moran[i][j] = 1.0
    else:
        weights_matrix_moran = one_matrix_zero_diagonal(pop_size)
    
    return weights_matrix_moran


def simple_selection_process(
        is_cellular_selection: bool,
        competitor_rate: float,
        neighbors_topology: NeighborsTopology,
        coordinate: tuple[int, ...],
        all_neighborhoods_indices: dict[tuple[int, ...], list[tuple[int, ...]]]
) -> tuple:
    # A COMPETITOR IS A TUPLE WHERE THE FIRST ELEMENT IS AN INDEX (THE POSITION IN THE POPULATION),
    # THE SECOND ELEMENT IS THE INDIVIDUAL ITSELF, WHICH HAS AN ATTRIBUTE FITNESS THAT MUST BE MINIMIZED
    if is_cellular_selection:
        competitors: list = [neighbors_topology.get(idx_tuple, clone=False) for idx_tuple in all_neighborhoods_indices[coordinate]]
        competitors.sort(key=lambda x: x[0], reverse=False)
        
        if competitor_rate == 1.0:
            sampled_competitors: list = competitors
        else:
            sampled_competitors: list = [competitor for competitor in competitors if random.random() < competitor_rate]
        while len(sampled_competitors) < 2:
            sampled_competitors.append(competitors[int(random.random()*len(competitors))])
        sampled_competitors.sort(key=lambda x: x[1].fitness, reverse=False)
        first = sampled_competitors[0][1]
        second = sampled_competitors[1][1]
    else:
        first_tournament: list = neighbors_topology.neighborhood(coordinate, include_current_point=True, clone=False, distinct_coordinates=False)
        first_tournament.sort(key=lambda x: x[1].fitness, reverse=False)
        second_tournament: list = neighbors_topology.neighborhood(coordinate, include_current_point=True, clone=False, distinct_coordinates=False)
        second_tournament.sort(key=lambda x: x[1].fitness, reverse=False) 
        first = first_tournament[0][1]
        second = second_tournament[0][1]

    return first, second
