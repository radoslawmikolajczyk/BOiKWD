from __future__ import annotations
import assignment as a
import numpy as np
import math

check_precision = 0.0001

def soft_assert(passed: bool, msg: str) -> int:
    if passed:
        return 0
    else:
        print(msg)
        return 1 

def almost_eq(x: np.Array, y: np.Array) -> bool:
    return np.isclose(x, y, check_precision).all()

def check_rankings() -> int:
    solutions = a.ranking_assignment()
    expected_rankings = (np.array([0.34634713, 0.36914035, 0.28451251]), np.array([0.34715539, 0.36790115, 0.28494346]) )
    expected_choices = (1, 1)
    return sum([almost_eq(s.global_ranking, e) for s,e in zip(solutions, expected_rankings)]) + sum([s.choice == e for s,e in zip(solutions, expected_choices)])

def check_consistency() -> int:
    solutions = tuple(a.consistency_assignment())
    expected_saaty = (0.13475389612171895, 0.7428823466101196, 0.006100230953349328)
    expected_koczkodaj = (0.7857142857142857, 0.9857142857142858, 0.30000000000000004)
    return sum([math.isclose(s.saaty, e, abs_tol=check_precision) for s,e in zip(solutions, expected_saaty)])\
         + sum([math.isclose(s.koczkodaj, e, abs_tol=check_precision) for s,e in zip(solutions, expected_koczkodaj)])

points = check_rankings() + check_consistency()
print(f"--- partial points: {points}/10")