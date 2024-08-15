import torch
import statistics
from scipy.stats import entropy


def one_matrix_zero_diagonal(N: int) -> list[list[float]]:
    return [ [0.0 if i == j else 1.0 for j in range(N)] for i in range(N)]


def one_matrix(N: int) -> list[list[float]]:
    return [ [1.0 for j in range(N)] for i in range(N)]


def zero_matrix(N: int) -> list[list[float]]:
    return [ [0.0 for j in range(N)] for i in range(N)]


def sum_of_all_elem_in_matrix(v: list[list[float]]) -> float:
    return float(sum([sum(l) for l in v]))


def sum_of_all_elem_in_tensor_matrix(v: torch.Tensor) -> float:
    return float(v.sum().item())


def dot_product(v1: torch.Tensor, v2: torch.Tensor) -> float:
    return float(torch.dot(v1, v2).item())


def self_dot_product(v1: torch.Tensor) -> float:
    return float(torch.dot(v1, v1).item())


def euclidean_distance(v1: torch.Tensor, v2: torch.Tensor) -> float:
    return float((v1 - v2).pow(2).sum().sqrt().item())


def cosine_similarity(v1: torch.Tensor, v2: torch.Tensor) -> float:
    cosi = torch.nn.CosineSimilarity(dim=0)
    return float(cosi(v1, v2).item())


def geometric_center(vectors: list[torch.Tensor]) -> torch.Tensor:
    semantic_matrix = torch.stack(vectors, dim=0).float()
    return torch.mean(semantic_matrix, dim=0)


def global_moran_I(vectors: list[torch.Tensor], w: list[list[float]]) -> float:
    N: int = len(vectors)
    W: float = sum_of_all_elem_in_matrix(w)
    gc: torch.Tensor = geometric_center(vectors)

    numerator: float = sum([ sum([w[i][j] * dot_product(vectors[i] - gc, vectors[j] - gc) for j in range(N)]) for i in range(N)])
    
    denominator: float = sum([self_dot_product(vectors[i] - gc) for i in range(N)])
    denominator = denominator if denominator != 0 else 1e-12

    return ( float(N) / W ) * (numerator / denominator)


def compute_euclidean_diversity_all_distinct_distances(vectors: list[torch.Tensor], measure: str = 'median') -> float:
    distances: list[float] = []

    for i in range(len(vectors) - 1):
        for j in range(i + 1, len(vectors)):
            distances.append(euclidean_distance(vectors[i], vectors[j]))

    if measure == 'mean':
        return statistics.mean(distances)
    elif measure == 'median':
        return statistics.median(distances)
    else:
        raise AttributeError(f'Invalid measure {measure}.')


def niche_entropy(repr_, n_niches=10):
    """
    Calculate the niche entropy of a population.

    Args:
        repr_ (list): List of individuals in the population.
        n_niches (int): Number of niches to divide the population into.

    Returns:
        float: The entropy of the distribution of individuals across niches.
    """
    # https://www.semanticscholar.org/paper/Entropy-Driven-Adaptive-RoscaComputer/ab5c8a8f415f79c5ec6ff6281ed7113736615682
    # https://strathprints.strath.ac.uk/76488/1/Marchetti_etal_Springer_2021_Inclusive_genetic_programming.pdf

    num_nodes = [len(ind) - 1 for ind in repr_]
    min_ = min(num_nodes)
    max_ = max(num_nodes)
    pop_size = len(repr_)
    stride = (max_ - min_) / n_niches

    distributions = []
    for i in range(1, n_niches + 1):
        distribution = (
            sum((i - 1) * stride + min_ <= x < i * stride + min_ for x in num_nodes)
            / pop_size
        )
        if distribution > 0:
            distributions.append(distribution)

    return entropy(distributions)


def gsgp_pop_div_from_vectors(sem_vectors):
    """
    Calculate the diversity of a population from semantic vectors.

    Args:
        sem_vectors (torch.Tensor): Tensor of semantic vectors.

    Returns:
        float: The average pairwise distance between semantic vectors.
    """
    # https://ieeexplore.ieee.org/document/9283096
    return torch.sum(torch.cdist(sem_vectors, sem_vectors)) / (
        sem_vectors.shape[0] ** 2
    )
