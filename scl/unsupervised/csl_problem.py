import torch


class ConstrainedStatisticalLearningProblem:
    """Constrained learning problem base class.
    
    Constrained learning problems are defined by inheriting from
    `ConstrainedLearningProblem` and defining its attributes:

    - ``model``: underlying model to train
    - ``data``: data with which to train the model
    - ``objective_function``: objective function 
    - ``constraints`` constraints model must satisfy
    - ``rhs``: right-hand side of average constraints, i.e. the c in f(x) < c
    """

    def __init__(self):
        device = next(self.model.parameters()).device
        
        lambda_init = [0.0 for _ in self.constraints]

        # define dual variables for the constraints
        self.lambdas = [torch.tensor(lambda_0, dtype = torch.float,
                                        requires_grad = False,
                                        device = device) \
                        for (lambda_0, _) in zip(lambda_init, self.constraints)]
            
    def evaluate_lagrangian(self):
        """Evaluate lagrangian"""
        objective_value = self.objective_function()
        slacks = self.evaluate_constraints_slacks()
        dualized_slacks = [lambda_ * slack for lambda_, slack in zip(self.lambdas, slacks)]
        L = objective_value + sum(dualized_slacks)

        return L, objective_value, slacks

    def evaluate_objective(self):
        """Evaluate the objective function."""
        objective_value = self.objective_function()
        return objective_value

    def evaluate_constraints_slacks(self):
        """Evaluate the constraints slacks."""
        slacks_values = [constraint() - c for constraint, c in zip(self.constraints, self.rhs)]
        return slacks_values
