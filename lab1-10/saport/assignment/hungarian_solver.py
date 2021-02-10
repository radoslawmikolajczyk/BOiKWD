import numpy as np
from .model import Assignment, AssignmentProblem, NormalizedAssignmentProblem
from typing import List, Dict, Tuple, Set


class Solver:
    """
    A hungarian solver for the assignment problem.

    Methods:
    --------
    __init__(problem: AssignmentProblem):
        creates a solver instance for a specific problem
    solve() -> Assignment:
        solves the given assignment problem
    extract_mins(costs: np.Array):
        substracts from columns and rows in the matrix to create 0s in the matrix
    find_max_assignment(costs: np.Array) -> Dict[int,int]:
        finds the biggest possible assinments given 0s in the cost matrix
        result is a dictionary, where index is a worker index, value is the task index
    add_zero_by_crossing_out(costs: np.Array, partial_assignment: Dict[int,int])
        creates another zero(s) in the cost matrix by crossing out lines (rows/cols) with zeros in the cost matrix,
        then substracting/adding the smallest not crossed out value
    create_assignment(raw_assignment: Dict[int, int]) -> Assignment:
        creates an assignment instance based on the given dictionary assignment
    """

    def __init__(self, problem: AssignmentProblem):
        self.problem = NormalizedAssignmentProblem.from_problem(problem)

    def solve(self) -> Assignment:
        costs = np.array(self.problem.costs)

        while True:
            self.extracts_mins(costs)
            max_assignment = self.find_max_assignment(costs)
            if len(max_assignment) == self.problem.size():
                return self.create_assignment(max_assignment)
            self.add_zero_by_crossing_out(costs, max_assignment)

    def extracts_mins(self, costs):
        for i in range(len(costs)):
            costs[i] = costs[i] - min(costs[i])
            costs[:, i] = costs[:, i] - min(costs[:, i])

    def add_zero_by_crossing_out(self, costs: np.array, partial_assignment: Dict[int, int]):
        na_rows = []
        for i in range(len(costs)):
            if i not in partial_assignment:
                na_rows.append(i)
        rows_sbh = [*na_rows]
        cols_cross = []
        for r in na_rows:
            row = costs[r]
            zero_cols = []
            for i in range(len(costs)):
                if row[i] == 0:
                    zero_cols.append(i)
            cols_cross.extend(zero_cols)

            for i in zero_cols:
                col_assign = []
                for key, value in partial_assignment.items():
                    if value == i:
                        col_assign.append((key, value))
                if col_assign:
                    rows_sbh.append(col_assign[0][0])

        min_u = float('inf')
        for i in range(len(costs)):
            for j in range(len(costs[0])):
                if i in rows_sbh and j not in cols_cross:
                    min_u = min(min_u, costs[i][j])
        costs -= min_u
        for i in cols_cross:
            costs[:, i] += min_u
        for i in range(len(costs)):
            if i not in rows_sbh:
                costs[i] += min_u

    def find_max_assignment(self, costs) -> Dict[int, int]:
        assign = {}

        workers = {}
        for i in range(len(costs)):
            zero_count = len(costs[i]) - np.count_nonzero(costs[i])
            if zero_count == 0:
                continue
            workers[i] = zero_count

        tasks = {}
        for i in range(len(costs.T)):
            zero_count = len(costs.T[i]) - np.count_nonzero(costs.T[i])
            if zero_count == 0:
                continue
            tasks[i] = zero_count

        while True:
            seq = []
            optimal_worker = 0
            for item in workers.items():
                if (item[0], item[1]):
                    seq.append(item)
            if not seq:
                optimal_worker = None
            else:
                _, key = min((val, key) for key, val in seq)
                optimal_worker = key

            task_with_zero_cost = []
            for i in range(len(costs[optimal_worker])):
                if costs[optimal_worker][i] == 0:
                    task_with_zero_cost.append(i)

            w_seq = []
            best_fit = 0
            for item in tasks.items():
                if item[0] in task_with_zero_cost:
                    w_seq.append(item)
            if not w_seq:
                best_fit = None
            else:
                _, key = min((val, key) for key, val in w_seq)
                best_fit = key

            del workers[optimal_worker]
            if best_fit is None:
                continue
            del tasks[best_fit]
            assign[optimal_worker] = best_fit
            if len(workers.keys()) == 0 or len(tasks.keys()) == 0:
                break

        return assign

    def create_assignment(self, raw_assignment: Dict[int, int]) -> Assignment:
        assignments = []
        for i in range(len(self.problem.original_problem.costs)):
            assignments.append(-1)

        objective = 0
        for i, j in raw_assignment.items():
            if i >= len(self.problem.original_problem.costs) or j >= len(self.problem.original_problem.costs[0]):
                continue
            assignments[i] = j
            objective += self.problem.original_problem.costs[i, j]
        return Assignment(assignments, objective)
