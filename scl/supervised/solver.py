# Solver class for solving constrained statistical learning problems with average constraints as well as per sample constraints.

import torch
import numpy as np
import matplotlib.pyplot as plt
import wandb

from scl.supervised.utils import eval_model_per_sample_constraints


class PrimalDualBaseSolver:
    """Primal dual algorithm for solving constrained statistical learning 
    problems with average constrains as well as per sample constraints."""
    
    def __init__(self, csl_problem, optimizers, primal_lr, dual_lr, epochs, eval_every, train_dataloader, validation_loader, test_dataloader, path_save):
        self.primal_lr = primal_lr
        self.dual_lr = dual_lr
        self.epochs = epochs
        self.eval_every = eval_every
        self.train_dataloader = train_dataloader
        self.validation_loader = validation_loader
        self.test_dataloader = test_dataloader
        self.path_save = path_save
        self.device = csl_problem.device
        self.n_train_samples = len(train_dataloader.dataset)
        self.best_validation_error = np.inf

        # optimizers
        self.primal_optimizer = getattr(torch.optim, optimizers['dual_optimizer'])(csl_problem.model.parameters(), lr=primal_lr)
        self.dual_optimizer = getattr(torch.optim, optimizers['dual_optimizer'])(csl_problem.lambdas + csl_problem.mus, lr=dual_lr)

        # lr schedulers
        if optimizers['use_primal_lr_scheduler']:
            self.primal_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.primal_optimizer, step_size=100, gamma=0.5)
            self.use_primal_lr_scheduler = True
        else:
            self.use_primal_lr_scheduler = False

        if optimizers['use_dual_lr_scheduler']:
            self.dual_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.dual_optimizer, step_size=100, gamma=0.5)
            self.use_dual_lr_scheduler = True
        else:
            self.use_dual_lr_scheduler = False

        # initialize state dict to keep track of quantities of interest
        self.initialize_state_dict(len(csl_problem.lambdas), len(csl_problem.mus))

    def initialize_state_dict(self, num_dual_variables, num_per_sample_constraints):
        self.state_dict = {}

        self.state_dict['dual_variables'] = {f'dual_variable_{i}': [] for i in range(num_dual_variables)}
        self.state_dict['slacks'] = {f'slack_{i}': [] for i in range(num_dual_variables)}

        self.state_dict['per_sample_dual_variables'] = {f'per_sample_dual_variables_{i}': [] for i in range(num_per_sample_constraints)}
        self.state_dict['per_sample_slacks'] = {f'per_sample_slacks_{i}': [] for i in range(num_per_sample_constraints)}

        self.state_dict['primal_value'] = []
        self.state_dict['Lagrangian'] = []
        self.state_dict['approximate_duality_gap'] = []
        self.state_dict['aproximate_relative_duality_gap'] = []
        self.state_dict['Relative_L2_test_error'] = []
        self.state_dict['Relative_L2_validation_error'] = []

    def primal_dual_update(self, csl_problem):
        """Update primal and dual variables"""
        raise NotImplementedError

    def solve(self, csl_problem):
        """Solve constrained learning problem"""

        for e in range(self.epochs):
            csl_problem.model.train()

            # estimates for quantities of interest
            log_dict = {
                'L': 0, 
                'primal_value': 0, 
                'slacks': [torch.tensor(0, dtype=torch.float, device=self.device) for _ in csl_problem.rhs], 
                'per_sample_slacks': [torch.zeros_like(per_sample_rhs, dtype=torch.float, device=self.device) for per_sample_rhs in csl_problem.per_sample_rhs]}   
            
            for samples_idx, x, y in self.train_dataloader:
                samples_idx, x, y = samples_idx.to(self.device), x.to(self.device), y.to(self.device)
                log_dict = self.primal_dual_update(csl_problem, samples_idx, x, y, log_dict)

            # update learning rate
            if self.use_primal_lr_scheduler:
                self.primal_lr_scheduler.step()
            if self.use_dual_lr_scheduler:
                self.dual_lr_scheduler.step()

            # log
            self.diagnostics(csl_problem, log_dict, e)
    
    def diagnostics(self, csl_problem, log_dict, epoch):
        """Log qunatities of intrest"""

        L = log_dict['L'].item()
        primal_value = log_dict['primal_value'].item()
        slacks = [slack.item() for slack in log_dict['slacks']]
        per_sample_slacks = [per_sample_slack.detach().to('cpu') for per_sample_slack in log_dict['per_sample_slacks']]

        approx_duality_gap = (primal_value - L)
        approx_relative_duality_gap = approx_duality_gap / primal_value

        dual_variables = [lambda_.item() for lambda_ in csl_problem.lambdas]
        per_sample_dual_variables = [mu.detach().to('cpu') for mu in csl_problem.mus]

        # dict for logging to wandb
        to_log_wandb = {}

        # early stopping based on validation set
        validation_error = eval_model_per_sample_constraints(csl_problem.model, self.validation_loader, self.device)
        self.state_dict['Relative_L2_validation_error'].append(validation_error)
        if validation_error < self.best_validation_error:
            self.best_validation_error = validation_error
            self.best_epoch = epoch
            self.test_error_best_epoch = eval_model_per_sample_constraints(csl_problem.model, self.test_dataloader, self.device)
            best_model = csl_problem.model
            torch.save(best_model.state_dict(), f'{self.path_save}/best_model.pt') 
            
        to_log_wandb['Relative L2 validation error/Validation error'] = validation_error
        to_log_wandb['Relative L2 validation error/Epoch'] = epoch

        print('')
        print(f'Epoch {epoch} - Relative L2 validation error: {validation_error}')
        print(f'Best relative L2 validation error: {self.best_validation_error} at epoch {self.best_epoch}')
        print(f'Relative L2 test error at best epoch: {self.test_error_best_epoch}')

        if epoch % self.eval_every == 0:
            test_error = eval_model_per_sample_constraints(csl_problem.model, self.test_dataloader, self.device)
            self.state_dict['Relative_L2_test_error'].append(test_error)
            to_log_wandb['Relative L2 test error/Relative L2 test error'] = test_error
            to_log_wandb['Relative L2 test error/Epoch'] = epoch

            print(f'Epoch {epoch} -  Relative L2 test error: {test_error}')

        # log
        for i in range(len(dual_variables)):
            self.state_dict['dual_variables'][f'dual_variable_{i}'].append(dual_variables[i])
            self.state_dict['slacks'][f'slack_{i}'].append(slacks[i])
            
            to_log_wandb[f'Dual variables/Dual variable {i}'] = dual_variables[i]
            to_log_wandb[f'Slacks/Slack {i}'] = slacks[i]

        for i in range(len(per_sample_dual_variables)):
            self.state_dict['per_sample_dual_variables'][f'per_sample_dual_variables_{i}'].append(per_sample_dual_variables[i])
            self.state_dict['per_sample_slacks'][f'per_sample_slacks_{i}'].append(per_sample_slacks[i])

        self.state_dict['approximate_duality_gap'].append(approx_duality_gap)
        self.state_dict['aproximate_relative_duality_gap'].append(approx_relative_duality_gap)
        self.state_dict['primal_value'].append(primal_value)
        self.state_dict['Lagrangian'].append(L)

        to_log_wandb['Duality gap/Approximate duality gap'] = approx_duality_gap
        to_log_wandb['Duality gap/Approximate relative duality gap'] = approx_relative_duality_gap
        to_log_wandb['Other diagnostics/Primal value'] = primal_value
        to_log_wandb['Other diagnostics/Lagrangian'] = L

        to_log_wandb['epoch'] = epoch

        # log wandb
        wandb.log(to_log_wandb)

    def plot(self, path_save):
        """Plot diagnostics"""
        num_dual_variables = len(self.state_dict['dual_variables'])
        epochs = np.arange(self.epochs) + 1 
        epochs_eval = np.arange(0, self.epochs, self.eval_every) + 1

        # plot dual variables
        for i in range(num_dual_variables):
            plt.figure()
            plt.plot(epochs, self.state_dict['dual_variables'][f'dual_variable_{i}'], color='blue')
            plt.title(f'Evoluation of dual variable {i}')
            plt.xlabel('Epochs')
            plt.ylabel(f'Dual variable {i}')
            plt.grid(True)
            plt.savefig(f'{path_save}/dual_variable_{i}_.pdf', format='pdf', bbox_inches='tight')
            plt.close()

        # plot slacks
        for i in range(num_dual_variables):
            plt.figure()
            plt.plot(epochs, self.state_dict['slacks'][f'slack_{i}'], color='blue')
            plt.title(f'Evoluation of slack {i}')
            plt.xlabel('Epochs')
            plt.ylabel(f'Slack {i}')
            plt.grid(True)
            plt.savefig(f'{path_save}/slack_{i}.pdf', format='pdf', bbox_inches='tight')
            plt.close()

        # plot all dual variables in one plot (if more than one dual variable is present)
        if num_dual_variables > 1:
            plt.figure()
            for i in range(num_dual_variables):
                plt.plot(epochs, self.state_dict['dual_variables'][f'dual_variable_{i}'], label=f'Dual variable {i}')
            plt.title('Evoluation of dual variables')
            plt.xlabel('Epochs')
            plt.ylabel('Dual variables')
            plt.legend(frameon=True, loc='best', fontsize=12)
            plt.grid(True)
            plt.savefig(f'{path_save}/dual_variables.pdf', format='pdf', bbox_inches='tight')
            plt.close()

        # plot approximate duality gap
        plt.figure()
        plt.plot(epochs, self.state_dict['approximate_duality_gap'], color='blue')
        plt.title('Approximate duality gap')
        plt.xlabel('Epochs')
        plt.ylabel('Approximate duality gap')
        plt.grid(True)
        plt.savefig(f'{path_save}/approximate_duality_gap.pdf', format='pdf', bbox_inches='tight')
        plt.close()

        # plot approximate relative duality gap
        plt.figure()
        plt.plot(epochs, self.state_dict['aproximate_relative_duality_gap'], color='blue')
        plt.title('Approximate relative duality gap')
        plt.xlabel('Epochs')
        plt.ylabel('Approximate relative duality gap')
        plt.grid(True)
        plt.savefig(f'{path_save}/approximate_relative_duality_gap.pdf', format='pdf', bbox_inches='tight')
        plt.close()

        # plot primal and lagrangian value in same plots (from which the duality gap can be computed)
        plt.figure()
        plt.plot(epochs, self.state_dict['primal_value'], label='Primal value', color='red')
        plt.plot(epochs, self.state_dict['Lagrangian'], label='Lagrangian', color='blue')
        plt.title('Primal and Lagrangian value')
        plt.xlabel('Epochs')
        plt.ylabel('Value')
        plt.legend(frameon=True, loc='best', fontsize=12)
        plt.grid(True)
        plt.savefig(f'{path_save}/primal_lagrangian.pdf', format='pdf', bbox_inches='tight')
        plt.close()

        # plo relative L2 test error
        plt.figure()
        plt.plot(epochs_eval, self.state_dict['Relative_L2_test_error'], color='blue')
        plt.title('Relative L2 test Error')
        plt.xlabel('Epochs')
        plt.ylabel('Relative L2 Error')
        plt.grid(True)
        plt.savefig(f'{path_save}/relative_L2_test_error.pdf', format='pdf', bbox_inches='tight')
        plt.close()

        # plot relative L2 validation error
        plt.figure()
        plt.plot(epochs_eval, self.state_dict['Relative_L2_validation_error'], color='blue')
        plt.title('Relative L2 Validation Error')
        plt.xlabel('Epochs')
        plt.ylabel('Relative L2 Error')
        plt.grid(True)
        plt.savefig(f'{path_save}/relative_L2_validation_error.pdf', format='pdf', bbox_inches='tight')


class SimultaneousPrimalDual(PrimalDualBaseSolver):
    """Updates the primal and dual variales simultaneously (instead of primal first).
    This saves one call to evaluate the constraints which can be expensive."""

    def __init__(self, csl_problem, optimizers, primal_lr, dual_lr, epochs, eval_every, train_dataloader, validation_loader, test_dataloader, path_save):
        super().__init__(csl_problem, optimizers, primal_lr, dual_lr, epochs, eval_every, train_dataloader, validation_loader, test_dataloader, path_save)

    def primal_dual_update(self, csl_problem, samples_idx, x, y, log_dict):

        self.primal_optimizer.zero_grad()
        self.dual_optimizer.zero_grad()

        csl_problem.forward(x)
        csl_problem.store_targets_and_samples_idx(samples_idx, y)

        # primal update
        L, objective_value, slacks, per_sample_slacks = csl_problem.evaluate_lagrangian()
        L.backward()

        # dual update
        # for average constraints
        for i, slack in enumerate(slacks):
            csl_problem.lambdas[i].grad = -slack

        # for per sample constraints
        for i, per_sample_slack in enumerate(per_sample_slacks):
            expanded_per_sample_slack = torch.zeros_like(csl_problem.mus[i])
            expanded_per_sample_slack[samples_idx] = per_sample_slack
            csl_problem.mus[i].grad = -expanded_per_sample_slack
        
        self.primal_optimizer.step()
        self.dual_optimizer.step()

        # project dual variables onto the non-negative orthant
        for i in range(len(csl_problem.lambdas)):
            csl_problem.lambdas[i][csl_problem.lambdas[i] < 0] = 0
        for i in range(len(csl_problem.mus)):
            csl_problem.mus[i][csl_problem.mus[i] < 0] = 0

        # discard stored prediction and target
        csl_problem.reset()

        batch_size = x.shape[0]
        log_dict['L'] += batch_size * L / self.n_train_samples
        log_dict['primal_value'] += batch_size * objective_value / self.n_train_samples
        for i, slack in enumerate(slacks):
            log_dict['slacks'][i] += batch_size * slack / self.n_train_samples
        for i, per_sample_slack in enumerate(per_sample_slacks):
            log_dict['per_sample_slacks'][i][samples_idx] = per_sample_slack

        return log_dict
    