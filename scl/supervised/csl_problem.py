# Base class for constrained learning problems that can also handle per 
# sample constraints (and not only constraints averaged over all samples).

import torch


class ConstrainedStatisticalLearningProblem:
    """Constrained learning problem base class handling average and per sample constraints.
    
    Constrained learning problems are defined by inheriting from
    `ConstrainedLearningProblem` and defining its attributes:

    - model: underlying model to train
    - data: data with which to train the model
    - objective_function: objective function 
    - constraints: constraints model must satisfy
    - rhs: right-hand side of average constraints, i.e. the c in f(x) < c
    - per_sample_constraints: constraints model must satisfy for each sample
    - per_sample_rhs: right-hand side of per sample constraints. list of torch tensors or shape (num_samples,)
    """


    def __init__(self):
        device = next(self.model.parameters()).device

        if not hasattr(self, 'constraints'):
            self.constraints = []
            self.lambdas = []
            self.rhs = []
        else:
            # define dual variables for the (average) constraints
            self.lambdas = [torch.tensor(0, dtype=torch.float, requires_grad=False, device=device)
                            for _ in self.constraints]
        
        if not hasattr(self, 'per_sample_constraints'):
            self.per_sample_constraints = []
            self.mus = []
            self.per_sample_rhs = []
        else:
            # dual variables for per sample constraints, i.e. pointwise constraints
            self.mus = [torch.zeros_like(rhs, dtype=torch.float, requires_grad=False, device=device)
                        for rhs in self.per_sample_rhs]
                
    def evaluate_lagrangian(self):
        """Evaluate lagrangian"""
        objective_value = self.objective_function()
        
        slacks = self.evaluate_constraints_slacks()
        dualized_slacks = [lambda_ * slack for lambda_, slack in zip(self.lambdas, slacks)]

        per_sample_slacks = self.evaluate_per_sample_constraints_slacks()
        per_sample_dualized_slacks = [torch.sum(mu[self.samples_indices] * per_sample_slack) for mu, per_sample_slack in zip(self.mus, per_sample_slacks)]

        L = objective_value + sum(dualized_slacks) + sum(per_sample_dualized_slacks)

        return L, objective_value, slacks, per_sample_slacks

    def evaluate_objective(self):
        """Evaluate the objective function."""
        objective_value = self.objective_function()
        return objective_value

    def evaluate_constraints_slacks(self):
        """Evaluate the constraints slacks."""
        slacks_values = [constraint() - c for constraint, c in zip(self.constraints, self.rhs)]
        return slacks_values
    
    def evaluate_per_sample_constraints_slacks(self):
        """Evaluate the per sample constraints slacks, i.e poitwise constraints."""
        per_sample_slacks_values_list = []
        for per_sample_constraint, c in zip(self.per_sample_constraints, self.per_sample_rhs):
            per_sample_slacks_values = per_sample_constraint() - c[self.samples_indices]
            per_sample_slacks_values_list.append(per_sample_slacks_values)

        return per_sample_slacks_values_list
    