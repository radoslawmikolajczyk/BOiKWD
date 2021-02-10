import numpy as np
from .model import AssignmentProblem, Assignment, NormalizedAssignmentProblem
from ..simplex.model import Model
from ..simplex.expressions.expression import Expression
from dataclasses import dataclass
from typing import List
from collections import defaultdict


class Solver:
    """
    A simplex solver for the assignment problem.

    Methods:
    --------
    __init__(problem: AssignmentProblem):
        creates a solver instance for a specific problem
    solve() -> Assignment:
        solves the given assignment problem
    """

    def __init__(self, problem: AssignmentProblem):
        self.problem = NormalizedAssignmentProblem.from_problem(problem)

    def solve(self) -> Assignment:
        model = Model("assignment")

        objective_expression = Expression()
        vars_dict = {}
        for i in range(len(self.problem.costs)):
            for j in range(len(self.problem.costs[0])):
                variable = model.create_variable(f"cost{i}{j}")
                model.add_constraint(variable <= 1)
                objective_expression += self.problem.costs[i][j] * variable
                vars_dict.setdefault(f"row{i}", []).append(variable)
                vars_dict.setdefault(f"col{j}", []).append(variable)

        model.minimize(objective_expression)
        for i in range(len(self.problem.costs)):
            model.add_constraint(sum(vars_dict[f"row{i}"], Expression()) == 1)
        for i in range(len(self.problem.costs[0])):
            model.add_constraint(sum(vars_dict[f"col{i}"], Expression()) == 1)

        result = model.solve()

        assign = []
        for i in range(len(self.problem.costs)):
            assign.append(-1)

        for i in range(len(self.problem.original_problem.costs)):
            r_val = []
            for var in vars_dict[f"row{i}"]:
                r_val.append(result.value(var))
            if len(self.problem.original_problem.costs[0]) > r_val.index(1):
                assign[i] = r_val.index(1)

        to_objective = []
        for i, j in enumerate(assign):
            if j >= 0:
                to_objective.append(self.problem.original_problem.costs[i][j])
        objective_expression = sum(to_objective)

        return_value = Assignment(assign, objective_expression)

        return return_value



