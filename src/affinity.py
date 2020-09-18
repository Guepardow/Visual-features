import numpy as np
from scipy.stats import wasserstein_distance


def get_manhattan(x, dict_y):
    """
    Computes Manhattan distance, aka L1-distance, between two vectors. The lower the better.

    :param x: vector of appearances of the query object
    :param dict_y: dictionnary of appearances of the candidates objects
    """
    distances = {key: np.linalg.norm(x-y, ord=1) for key, y in dict_y.items()}
    return distances


def get_euclidean(x, dict_y):
    """
    Computes Euclidean distance, aka L2-distance, between two vectors. The lower the better.

    :param x: vector of appearances of the query object
    :param dict_y: dictionnary of appearances of the candidates objects
    """
    distances = {key: np.linalg.norm(x-y, ord=2) for key, y in dict_y.items()}
    return distances


def get_cosine(x, dict_y):
    """
    Computes the cosine similarity between two vectors. The higher the better.

    :param x: vector of appearances of the query object
    :param dict_y: dictionnary of appearances of the candidates objects
    """
    similarity = {key: x.dot(y)/(np.linalg.norm(x) * np.linalg.norm(y)) for key, y in dict_y.items()}
    return similarity


def get_dotProduct(x, dict_y):
    """
    Computes the dot product between two vectors. The higher the better.

    :param x: vector of appearances of the query object
    :param dict_y: dictionnary of appearances of the candidates objects
    """
    similarity = {key: x.dot(y) for key, y in dict_y.items()}
    return similarity


def get_rank1(x, dict_y):
    """
    Computes rank-1 distance between one vector given a set of other vectors. The higher the better.

    :param x: vector of appearances of the query object
    :param dict_y: dictionnary of appearances of the candidates objects
    """

    # We create a 2D-array with all the features from the second frame
    list_candidats = np.array(list(dict_y.values()))

    # Then, for each coordinates, we give 1 point for the objectID2 which is the closest
    # to objectID1
    distanceL1 = np.absolute(list_candidats - x)
    min_dist = np.min(distanceL1, axis=0)  # minimal distance for each coordinate
    is_min = distanceL1 == min_dist  # where the minimum is reached

    n_min = np.sum(is_min, axis=0)  # number of minimum for each coordinate

    points = np.sum(distanceL1[:, n_min == 1] == min_dist[n_min == 1], axis=1)

    similarity = {key: value for key, value in zip(list(dict_y.keys()), points)}
    return similarity


def get_bhattacharyya(x, dict_y):
    """
    Used for probability distribution. The lower the better.

    :param x: vector of appearances of the query object
    :param dict_y: dictionnary of appearances of the candidates objects
    """

    epsilon = 1e-9
    x_norm = x/np.sum(x)  # normalization to get a probability distribution

    distance = {key: -np.log(epsilon + np.sqrt(x_norm).dot(np.sqrt(y/np.sum(y)))) for key, y in dict_y.items()}
    return distance


def get_wasserstein(x, dict_y):
    """
    Used for probability distribution. The lower the better.
    """

    x_norm = x/np.sum(x)  # normalization to get a distribution
    distance = {key: wasserstein_distance(x_norm, y/np.sum(y)) for key, y in dict_y.items()}
    return distance