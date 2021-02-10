from __future__ import annotations
import numpy as np
import networkx as nx
import sys
from typing import List


def read_float(number: str) -> float:
    """
    Just a helper function to read rational numbers (fractions) into floats, i.e. "1/5" -> 0.2
    """
    try:
        return float(number)
    except ValueError:
        num, denom = number.split('/')
        return float(num) / float(denom)


def read_upper_traingular_row(row: List[str]) -> List[float]:
    return [read_float(n) for n in row.split(",")]


def read_upper_triangular(s: str) -> List[List[float]]:
    return [read_upper_traingular_row(r) for r in s.split(";")]


def fill_comparison_matrix_from_upper_triangular(upper_triangular: List[List[float]]) -> np.array:
    """
        Creates a comparison matrix based on the upper triangular part of the matrix, i.e.
        an upper triangular matrix:

        0 2 3
        0 0 4
        0 0 0

        we should get a full comparison matrix:

        1    2    3
        1/2  1    4
        1/3  1/4  1
    """
    size = len(upper_triangular) + 1
    comparison_matrix = np.eye(size, dtype=float)
    for row in range(size - 1):
        comparison_matrix[row, 1 + row:] = upper_triangular[row]
    for row in range(1, size):
        for col in range(0, row):
            comparison_matrix[row, col] = 1 / comparison_matrix[col, row]

    return comparison_matrix


def read_comparison_matrix(s: str) -> np.array:
    """
    Just a helper function to read a comparison matrix from upper triangular matrix
    written in a simple format, where "," separates cols and ";" rows, e.g.

    "2,3;4" = [[2,3],[4]]

    which represents:

    0 2 3
    0 0 4
    0 0 0
    """
    upper_triangular_matrix = read_upper_triangular(s)
    return fill_comparison_matrix_from_upper_triangular(upper_triangular_matrix)


def principal_eigenvector(c: np.Array) -> List[float]:
    graph = nx.from_numpy_matrix(c.T, create_using=nx.DiGraph)
    return np.array([v for v in nx.eigenvector_centrality_numpy(graph, weight='weight').values()])


def principal_eigenvalue(c: np.Array) -> float:
    pe = principal_eigenvector(c)
    nz_index = pe.nonzero()[0][0]
    return c.dot(pe)[nz_index] / pe[nz_index]


def normalize_vector(w: np.Array) -> np.Array:
    return w / sum(w)


def evm(c: np.array) -> np.Array:
    return normalize_vector(principal_eigenvector(c))


def gmm(c: np.Array) -> np.Array:
    var = []
    for i in range(len(c)):
        mean_val = c[i].prod() ** (1.0 / len(c[i]))
        var.append(mean_val)
    normalized = normalize_vector(var)
    return normalized


def saaty_index(c: np.Array) -> float:
    l_max = principal_eigenvalue(c)
    alt = c.shape
    return_val = (l_max - alt[0]) / (alt[0] - 1)
    return return_val


def koczkodaj_index(c: np.Array) -> float:
    indexes = []
    for i in range(c.shape[0]):
        for j in range(c.shape[0]):
            for k in range(c.shape[0]):
                append_value = min(abs(1 - c[i, j] / (c[i, k] * c[k, j])), abs(1 - (c[i, k] * c[k, j]) / c[i, j]))
                indexes.append(append_value)

    return max(indexes)


if __name__ == "__main__":
    '''
    Reads the upper triangular from the first arg and calculates normalized principal eigenvector
    '''
    comparison_matrix_string = sys.argv[1]
    comparison_matrix = read_comparison_matrix(comparison_matrix_string)
    print(evm(comparison_matrix))
