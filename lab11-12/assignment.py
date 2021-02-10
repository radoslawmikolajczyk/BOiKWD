from __future__ import annotations
import ahp
from dataclasses import dataclass
import numpy as np
from typing import Tuple


@dataclass
class RankingSolution:
    rankings: Tuple[np.Array]
    preference_ranking: np.Array
    global_ranking: np.Array
    choice: int


@dataclass
class ConsistencySolution:
    saaty: float
    koczkodaj: float


def ranking_assignment():
    Cs = (
        ahp.read_comparison_matrix("1/7,1/5;3"),
        ahp.read_comparison_matrix("5,9;4"),
        ahp.read_comparison_matrix("4,1/5;1/9"),
        ahp.read_comparison_matrix("9,4;1/4"),
        ahp.read_comparison_matrix("1,1;1"),
        ahp.read_comparison_matrix("6,4;1/3"),
        ahp.read_comparison_matrix("9,6;1/3"),
        ahp.read_comparison_matrix("1/2,1/2;1")
    )
    c_p = ahp.read_comparison_matrix("4,7,5,8,6,6,2;5,3,7,6,6,1/3;1/3,5,3,3,1/5;6,3,4,1/2;1/3,1/4,1/7;1/2,1/5;1/5")

    evm_rankings = [ahp.evm(m) for m in Cs]
    evm_p_ranking = ahp.evm(c_p)
    evm_global_ranking = []
    for i in range(len(evm_rankings[0])):
        option_weights = np.array([lr[i] for lr in evm_rankings])
        evm_global_ranking.append(option_weights @ evm_p_ranking)
    evm_choice = np.where(np.array(evm_global_ranking) == max(evm_global_ranking))[0][0]
    evm_solution = RankingSolution(evm_rankings, evm_p_ranking, evm_global_ranking, evm_choice)

    gmm_rankings = [ahp.gmm(m) for m in Cs]
    gmm_p_ranking = ahp.gmm(c_p)
    gmm_global_ranking = []
    for i in range(len(gmm_rankings[0])):
        option_weights = np.array([lr[i] for lr in gmm_rankings])
        gmm_global_ranking.append(option_weights @ gmm_p_ranking)
    gmm_choice = np.where(np.array(gmm_global_ranking) == max(gmm_global_ranking))[0][0]
    gmm_solution = RankingSolution(gmm_rankings, gmm_p_ranking, gmm_global_ranking, gmm_choice)

    return evm_solution, gmm_solution


def consistency_assignment():
    Cs = (
        ahp.read_comparison_matrix("7,3;2"),
        ahp.read_comparison_matrix("1/5,7,1;1/2,2;3"),
        ahp.read_comparison_matrix("2,5,1,7;3,1/2,5;1/5,2;7")
    )

    saaty_indexes = []
    koczkodaj_indexes = []
    for m in Cs:
        saaty_indexes.append(ahp.saaty_index(m))
    for m in Cs:
        koczkodaj_indexes.append(ahp.koczkodaj_index(m))

    return (ConsistencySolution(s, k) for s, k in zip(saaty_indexes, koczkodaj_indexes))
