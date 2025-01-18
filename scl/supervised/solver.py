import torch
import numpy as np
import matplotlib.pyplot as plt
import wandb

from supervised.utils import eval_model


class PrimalDualBaseSolver:
    """Primal dual algorithm for solving constrained statistical learning problems."""
    
    def __init__(self, csl_problem, optimizers, primal_lr, dual_lr, epochs, eval_every, train_dataloader, test_dataloader):
        self.primal_lr = primal_lr
        self.dual_lr = dual_lr
        self.epochs = epochs
        self.eval_every = eval_every
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.device = csl_problem.device

        # optimizers
        self.primal_optimizer = getattr(torch.optim, optimizers['dual_optimizer'])(csl_problem.model.parameters(), lr=primal_lr)
        self.dual_optimizer = getattr(torch.optim, optimizers['dual_optimizer'])(csl_problem.lambdas, lr=dual_lr)

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
        self.initialize_state_dict(len(csl_problem.lambdas))

    def initialize_state_dict(self, num_dual_variables):
        self.state_dict = {}
        self.state_dict['dual_variables'] = {f'dual_variable_{i}': [] for i in range(num_dual_variables)}
        self.state_dict['slacks'] = {f'slack_{i}': [] for i in range(num_dual_variables)}
        self.state_dict['primal_value'] = []
        self.state_dict['Lagrangian'] = []
        self.state_dict['approximate_duality_gap'] = []
        self.state_dict['aproximate_relative_duality_gap'] = []
        self.state_dict['Relative_L2_test_error'] = []

    def primal_dual_update(self, csl_problem):
        """Update primal and dual variables"""
        raise NotImplementedError
    
    def solve(self, csl_problem):
        """Solve constrained learning problem"""

        for e in range(self.epochs):
            csl_problem.model.train()
            n_batches = len(self.train_dataloader)
            for i, (x, y) in enumerate(self.train_dataloader):
                x, y = x.to(self.device), y.to(self.device)
                logging_dict = self.primal_dual_update(csl_problem, x, y)

                # log qunataties of interest
                if i == (n_batches - 1):
                    evaluate_model = True
                else:
                    evaluate_model = False
                self.diagnostics(csl_problem, logging_dict, e, evaluate_model)

            # update learning rate
            if self.use_primal_lr_scheduler:
                self.primal_lr_scheduler.step()
            if self.use_dual_lr_scheduler:
                self.dual_lr_scheduler.step()

    def diagnostics(self, csl_problem, logging_dict, epoch, evaluate_model):
        """Log qunatities of intrest"""

        L = logging_dict['L']
        primal_value = logging_dict['primal_value']
        slacks = logging_dict['slacks']

        approx_duality_gap = (primal_value - L).item()
        approx_relative_duality_gap = approx_duality_gap / primal_value.item()

        dual_variables = [lambda_.item() for lambda_ in csl_problem.lambdas]
        slacks = [slack.item() for slack in slacks]

        # dict for logging to wandb
        to_log_wandb = {}

        if epoch % self.eval_every == 0 and evaluate_model:
            test_error = eval_model(csl_problem.model, self.test_dataloader, self.device)
            self.state_dict['Relative_L2_test_error'].append(test_error)
            to_log_wandb['Relative L2 test error/Relative L2 test error'] = test_error
            to_log_wandb['Relative L2 test error/Epoch'] = epoch

            print(f'Epoch {epoch} -  Relative L2 error: {test_error}')

        # log
        for i in range(len(dual_variables)):
            self.state_dict['dual_variables'][f'dual_variable_{i}'].append(dual_variables[i])
            self.state_dict['slacks'][f'slack_{i}'].append(slacks[i])
            to_log_wandb[f'Dual variables/Dual variable {i}'] = dual_variables[i]
            to_log_wandb[f'Slacks/Slack {i}'] = slacks[i]

        self.state_dict['approximate_duality_gap'].append(approx_duality_gap)
        self.state_dict['aproximate_relative_duality_gap'].append(approx_relative_duality_gap)
        self.state_dict['primal_value'].append(primal_value.item())
        self.state_dict['Lagrangian'].append(L.item())
        to_log_wandb['Duality gap/Approximate duality gap'] = approx_duality_gap
        to_log_wandb['Duality gap/Approximate relative duality gap'] = approx_relative_duality_gap
        to_log_wandb['Other diagnostics/Primal value'] = primal_value.item()
        to_log_wandb['Other diagnostics/Lagrangian'] = L.item()

        to_log_wandb['epoch'] = epoch

        # log wandb
        wandb.log(to_log_wandb)

        csl_problem.model.train()

    def plot(self, path_save):
        """Plot diagnostics"""
        num_dual_variables = len(self.state_dict['dual_variables'])
        num_batches = len(self.train_dataloader)
        epochs = np.repeat(np.arange(self.epochs) + 1, num_batches)     # done since we log each batch
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

        # plot all dual variables in one plot
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


class PrimalThenDual(PrimalDualBaseSolver):
    """Updates the primal vairables first and then the dual variables"""

    def __init__(self, csl_problem, optimizers, primal_lr, dual_lr, epochs, eval_every, train_dataloader, test_dataloader):
        super().__init__(csl_problem, optimizers, primal_lr, dual_lr, epochs, eval_every, train_dataloader, test_dataloader)
        
    def primal_dual_update(self, csl_problem, x, y):
        """Update primal and dual variables for one batch."""

        # make prediction and store it together with target as attribute of csl_problem
        # Since objective and consrtaints both use the prediction, this is done so that only one forward pass is required
        csl_problem.forward(x)
        csl_problem.store_targets(y)

        # primal update
        self.primal_optimizer.zero_grad()
        L, objective_value, _ = csl_problem.evaluate_lagrangian()
        L.backward()
        self.primal_optimizer.step()

        # dual update
        # NOTE: The Lagrangian is a concave function of the dual variables so we know the gradient and don't need to compute it
        self.dual_optimizer.zero_grad()
        slacks = csl_problem.evaluate_constraints_slacks()
        for i, slack in enumerate(slacks):
            csl_problem.lambdas[i].grad = -slack
        self.dual_optimizer.step()

        # project dual variables onto the non-negative orthant
        # this is required by the definition of the dual variable
        for i in range(len(csl_problem.lambdas)):
            csl_problem.lambdas[i][csl_problem.lambdas[i] < 0] = 0

        # discard stored prediction and target
        csl_problem.reset()

        logging_dict = {'L': L, 'primal_value': objective_value, 'slacks': slacks}
        return logging_dict


class SimultaneousPrimalDual(PrimalDualBaseSolver):
    """Updates the primal and dual variales simultaneously (instead of primal first).
    This saves one call to evaluate the constraints which can be expensive."""

    def __init__(self, csl_problem, optimizers, primal_lr, dual_lr, epochs, eval_every, train_dataloader, test_dataloader):
        super().__init__(csl_problem, optimizers, primal_lr, dual_lr, epochs, eval_every, train_dataloader, test_dataloader)

    def primal_dual_update(self, csl_problem, x, y):

        self.primal_optimizer.zero_grad()
        self.dual_optimizer.zero_grad()

        csl_problem.forward(x)
        csl_problem.store_targets(y)

        # primal update
        L, objective_value, slacks = csl_problem.evaluate_lagrangian()
        L.backward()

        # dual update
        for i, slack in enumerate(slacks):
            csl_problem.lambdas[i].grad = -slack
        
        self.primal_optimizer.step()
        self.dual_optimizer.step()

        # project dual variables onto the non-negative orthant
        for i in range(len(csl_problem.lambdas)):
            csl_problem.lambdas[i][csl_problem.lambdas[i] < 0] = 0

        # discard stored prediction and target
        csl_problem.reset()

        logging_dict = {'L': L, 'primal_value': objective_value, 'slacks': slacks}
        return logging_dict
