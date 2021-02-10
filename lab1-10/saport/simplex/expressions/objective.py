from enum import Enum
from saport.simplex.expressions import expression 

class ObjectiveType(Enum):
    """
        An enum to represent an objective type:
        - MAX = maximize the objective
        - MIN = minimze the objective
    """
    MAX = 1
    MIN = -1

    def __str__(self):
        return {
            ObjectiveType.MAX: 'max',
            ObjectiveType.MIN: 'min'
        }[self]

class Objective: 
    """
        A class to represent an objective in the linear programming expression, e.g. 4x + 5y -> max, etc.

        Attributes
        ----------
        expression : Expression
            polynomial expressions that is being optimized
        type: ObjectiveType
            type of the objective: MIN, MAX
        factor: float
            factor associated with the objective variable used in simplex algorithm

        Methods
        -------
        __init__(expression: Expression, type: ObjectiveType = ObjectiveType.Max) -> Constraint:
            constructs new objective with a specified polynomial and type
        simplify() -> Objective:
            returns new objective with the simplified polynomial
        invert():
            inverts the objective, keeping the "objective variable factor" intact  
        evaluate(assignemnt: list[float]) -> float:
            returns value of the objective for the given assignment
            assignment is just a list of floats corresponding (by index) to the variables in the model 
    """

    def __init__(self, expression, type = ObjectiveType.MAX, factor = 1.0):
        self.expression = expression
        self.type = type
        self.factor = factor

    def invert(self):
        self.type = ObjectiveType(self.type.value * -1)
        self.expression = self.expression * -1
        self.factor = self.factor * -1

    def simplify(self):
        return Objective(self.expression.simplify(), self.type, self.factor)

    def depends_on_variable(self, model, variable):
        return self.expression.factors(model)[variable.index] != 0

    def evaluate(self, assignment):
        return self.expression.evaluate(assignment)
        
    def name(self):
        return f"{'-' if self.factor < 0 else ''}z"

    def __str__(self):
        return f'{self.type}: {self.name()} = {self.expression}'