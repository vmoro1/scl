# pde_param refers to the parameter(s) of the PDE, e.g., beta for convection or (nu,rho) for reaction diffusion

import torch
import numpy as np
import matplotlib.pyplot as plt


class PrimalDualBase:
    """Primal dual algorithm for solving constrained statistical learning problems."""
    def __init__(self, csl_problem, optimizers, primal_lr, dual_lr, epochs, eval_every, data_dict, solving_specific_BVP=False, path_save=None):
        self.primal_lr = primal_lr
        self.dual_lr = dual_lr
        self.epochs = epochs
        self.eval_every = eval_every

        # data for evaluation
        self.data_dict = data_dict

        # optimizers
        self.primal_optimizer = getattr(torch.optim, optimizers['dual_optimizer'])(csl_problem.model.parameters(), lr=primal_lr)
        self.dual_optimizer = getattr(torch.optim, optimizers['dual_optimizer'])(csl_problem.lambdas, lr=dual_lr)

        # lr schedulers
        if optimizers['use_primal_lr_scheduler']:
            self.primal_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.primal_optimizer, step_size=5000, gamma=0.9)
            self.use_primal_lr_scheduler = True
        else:
            self.use_primal_lr_scheduler = False

        if optimizers['use_dual_lr_scheduler']:
            self.dual_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.dual_optimizer, step_size=5000, gamma=0.9)
            self.use_dual_lr_scheduler = True
        else:
            self.use_dual_lr_scheduler = False

        # initialize state dict to keep track of quantities of interest
        self.initialize_state_dict(len(csl_problem.lambdas))

        # attributes for early stopping in diagnostics
        self.solving_specific_BVP = solving_specific_BVP
        self.best_epoch = None
        self.best_primal_value = np.inf
        self.path_save = path_save


    def initialize_state_dict(self, num_dual_variables):
        self.state_dict = {}
        self.state_dict['dual_variables'] = {f'dual_variable_{i}': [] for i in range(num_dual_variables)}
        self.state_dict['slacks'] = {f'slack_{i}': [] for i in range(num_dual_variables)}
        self.state_dict['primal_value'] = []
        self.state_dict['Lagrangian'] = []
        self.state_dict['approximate_duality_gap'] = []
        self.state_dict['aproximate_relative_duality_gap'] = []

        self.state_dict['Relative Error'] = {f'Relative Error_pde_param_{pde_param}': [] for pde_param in self.data_dict.keys()}
        self.state_dict['Absolute Error'] = {f'Absolute Error_pde_param_{pde_param}': [] for pde_param in self.data_dict.keys()}
        self.state_dict['linf Error'] = {f'linf Error_pde_param_{pde_param}': [] for pde_param in self.data_dict.keys()}

    def primal_dual_update(self, csl_problem):
        """Update primal and dual variables"""
        raise NotImplementedError

    def solve(self, csl_problem):
        """Solve constrained learning problem"""

        for e in range(self.epochs):
            csl_problem.model.train()

            logging_dict = self.primal_dual_update(csl_problem)

            # update learning rate
            if self.use_primal_lr_scheduler:
                self.primal_lr_scheduler.step()
            if self.use_dual_lr_scheduler:
                self.dual_lr_scheduler.step()

            # log quantities of interest
            self.diagnostics(csl_problem, logging_dict, e)

    def diagnostics(self, csl_problem, logging_dict, epoch):
        """Log qunatities of intrest"""

        L = logging_dict['L']
        primal_value = logging_dict['primal_value']
        slacks = logging_dict['slacks']

        approx_duality_gap = (primal_value - L).item()
        approx_relative_duality_gap = approx_duality_gap / primal_value.item()

        dual_variables = [lambda_.item() for lambda_ in csl_problem.lambdas]
        slacks = [slack.item() for slack in slacks]

        # only do for solving specific BVP not parametric solution (only one entry in dict/one pde_param)
        if self.solving_specific_BVP:   
            # find best checkpoint based on feasibility (withing 1.1 eps but can be changed) plus small objective
            # slacks already incorporate eps so it's only less than 0.1 eps
            feasible = all(slack < 0.1 * eps for slack, eps in zip(slacks, csl_problem.rhs))
            if primal_value.item() < self.best_primal_value and feasible:
                self.best_epoch = epoch
                self.best_primal_value = primal_value.item()
                best_model = csl_problem.model
                torch.save(best_model.state_dict(), f'{self.path_save}/best_model.pt') 

                pde_param = list(self.data_dict.keys())[0]
                X_test = self.data_dict[pde_param]['X_test']
                u_pred = csl_problem.predict(X_test, pde_param)
                try:
                    u_exact = self.data_dict[pde_param]['u_exact']
                except:
                    u_exact = self.data_dict[pde_param]['u_exact_1d']

                self.best_rel_l2_error = np.linalg.norm(u_exact-u_pred, 2)/np.linalg.norm(u_exact, 2)

        # evaluation of model
        if epoch % self.eval_every == 0:
            for pde_param in self.data_dict.keys():
                X_test = self.data_dict[pde_param]['X_test']

                # data from convection/rd is named differently than the rest, i.e., u_exact is 1D 
                # for convection/rd but for others it's u_exact_1d that's the 1D array
                # TODO: fix this
                try:
                    u_exact = self.data_dict[pde_param]['u_exact']
                    u_pred = csl_problem.predict(X_test, pde_param)
                    error_u_relative = np.linalg.norm(u_exact-u_pred, 2)/np.linalg.norm(u_exact, 2)
                    error_u_abs = np.mean(np.abs(u_exact - u_pred))
                    error_u_linf = np.linalg.norm(u_exact - u_pred, np.inf)/np.linalg.norm(u_exact, np.inf)
                except:
                    u_exact_1d_pde_param = self.data_dict[pde_param]['u_exact_1d']
                    u_pred_pde_param = csl_problem.predict(X_test, pde_param)
                    error_u_relative = np.linalg.norm(u_exact_1d_pde_param-u_pred_pde_param, 2)/np.linalg.norm(u_exact_1d_pde_param, 2)
                    error_u_abs = np.mean(np.abs(u_exact_1d_pde_param - u_pred_pde_param))
                    error_u_linf = np.linalg.norm(u_exact_1d_pde_param - u_pred_pde_param, np.inf)/np.linalg.norm(u_exact_1d_pde_param, np.inf)

                self.state_dict['Relative Error'][f'Relative Error_pde_param_{pde_param}'].append(error_u_relative)
                self.state_dict['Absolute Error'][f'Absolute Error_pde_param_{pde_param}'].append(error_u_abs)
                self.state_dict['linf Error'][f'linf Error_pde_param_{pde_param}'].append(error_u_linf)

        # log
        for i in range(len(dual_variables)):
            self.state_dict['dual_variables'][f'dual_variable_{i}'].append(dual_variables[i])
            self.state_dict['slacks'][f'slack_{i}'].append(slacks[i])
        self.state_dict['approximate_duality_gap'].append(approx_duality_gap)
        self.state_dict['aproximate_relative_duality_gap'].append(approx_relative_duality_gap)
        self.state_dict['primal_value'].append(primal_value.item())
        self.state_dict['Lagrangian'].append(L.item())

    def plot(self, path_save):
        """Plot diagnostics"""
        num_dual_variables = len(self.state_dict['dual_variables'])
        epochs = np.arange(self.epochs) + 1

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

        epochs_eval = np.arange(0, self.epochs, self.eval_every) + 1
        for pde_param in self.data_dict.keys():
            # plot Relative Error
            plt.figure()
            plt.plot(epochs_eval, self.state_dict['Relative Error'][f'Relative Error_pde_param_{pde_param}'], color='blue')
            plt.title(f'Relative Error for pde_param={pde_param}')
            plt.xlabel('Epochs')
            plt.ylabel('Relative Error')
            plt.grid(True)
            plt.savefig(f'{path_save}/relative_error_pde_param_{pde_param}.pdf', format='pdf', bbox_inches='tight')
            plt.close()

            # plot Absolute Error
            plt.figure()
            plt.plot(epochs_eval, self.state_dict['Absolute Error'][f'Absolute Error_pde_param_{pde_param}'], color='blue')
            plt.title(f'Absolute Error for pde_param={pde_param}')
            plt.xlabel('Epochs')
            plt.ylabel('Absolute Error')
            plt.grid(True)
            plt.savefig(f'{path_save}/absolute_error_pde_param_{pde_param}.pdf', format='pdf', bbox_inches='tight')
            plt.close()

            # plot linf Error
            plt.figure()
            plt.plot(epochs_eval, self.state_dict['linf Error'][f'linf Error_pde_param_{pde_param}'], color='blue')
            plt.title(f'linf Error for pde_param={pde_param}')
            plt.xlabel('Epochs')
            plt.ylabel('linf Error')
            plt.grid(True)
            plt.savefig(f'{path_save}/linf_error_pde_param_{pde_param}.pdf', format='pdf', bbox_inches='tight')
            plt.close()

        # plot all errors in one plot
        plt.figure()
        for pde_param in self.data_dict.keys():
            plt.plot(epochs_eval, self.state_dict['Relative Error'][f'Relative Error_pde_param_{pde_param}'], label=f'Relative Error pde_param {pde_param}')
        plt.title('Relative Error')
        plt.xlabel('Epochs')
        plt.ylabel('Relative Error')
        plt.legend(frameon=True, loc='best', fontsize=12)
        plt.grid(True)
        plt.savefig(f'{path_save}/relative_error.pdf', format='pdf', bbox_inches='tight')
        plt.close()

        plt.figure()
        for pde_param in self.data_dict.keys():
            plt.plot(epochs_eval, self.state_dict['Absolute Error'][f'Absolute Error_pde_param_{pde_param}'], label=f'Absolute Error pde_param {pde_param}')
        plt.title('Absolute Error')
        plt.xlabel('Epochs')
        plt.ylabel('Absolute Error')
        plt.legend(frameon=True, loc='best', fontsize=12)
        plt.grid(True)
        plt.savefig(f'{path_save}/absolute_error.pdf', format='pdf', bbox_inches='tight')
        plt.close()

        plt.figure()
        for pde_param in self.data_dict.keys():
            plt.plot(epochs_eval, self.state_dict['linf Error'][f'linf Error_pde_param_{pde_param}'], label=f'linf Error pde_param {pde_param}')
        plt.title('linf Error')
        plt.xlabel('Epochs')
        plt.ylabel('linf Error')
        plt.legend(frameon=True, loc='best', fontsize=12)
        plt.grid(True)
        plt.savefig(f'{path_save}/linf_error.pdf', format='pdf', bbox_inches='tight')
        plt.close()



class PrimalThenDual(PrimalDualBase):
    """Updates the primal vairables first and then the dual variables"""
    
    def __init__(self, csl_problem, optimizers, primal_lr, dual_lr, epochs, eval_every, X_test, u_exact, solving_specific_BVP=False, path_save=None):
        super().__init__(csl_problem, optimizers, primal_lr, dual_lr, epochs, eval_every, X_test, u_exact, solving_specific_BVP, path_save)

    def primal_dual_update(self, csl_problem):
        """Update primal and dual variables"""

        # primal update
        L, primal, slacks = self._primal(csl_problem)

        logging_dict = {'L': L, 'primal_value': primal, 'slacks': slacks}

        # dual update
        # NOTE: the Lagrangian is a concave function of the dual variables so we know the gradient and don't need to compute it
        self.dual_optimizer.zero_grad()
        slacks = csl_problem.evaluate_constraints_slacks()
        for i, slack in enumerate(slacks):
            csl_problem.lambdas[i].grad = -slack
        self.dual_optimizer.step()

        # project dual variables onto the non-negative orthant
        # this is required by the definition of the dual variable
        for i in range(len(csl_problem.lambdas)):
            csl_problem.lambdas[i][csl_problem.lambdas[i] < 0] = 0

        return logging_dict

    def _primal(self, csl_problem):
        self.primal_optimizer.zero_grad()
        L, objective_value, slacks = csl_problem.evaluate_lagrangian()
        L.backward()
        self.primal_optimizer.step()
        return L, objective_value, slacks



class SimultaneousPrimalDual(PrimalDualBase):
    """Updates the primal and dual variales simultaneously (instead of primal first).
    This saves one call to evaluate the constraints which can be expensive for e.g. advesarial losses"""

    def __init__(self, csl_problem, optimizers, primal_lr, dual_lr, epochs, eval_every, data_dict, solving_specific_BVP=False, path_save=None):
        super().__init__(csl_problem, optimizers, primal_lr, dual_lr, epochs, eval_every, data_dict, solving_specific_BVP, path_save)

    def primal_dual_update(self, csl_problem):

        self.primal_optimizer.zero_grad()
        self.dual_optimizer.zero_grad()

        # primal update
        L, primal, slacks = csl_problem.evaluate_lagrangian()
        L.backward()

        logging_dict = {'L': L, 'primal_value': primal, 'slacks': slacks}

        # dual update
        for i, slack in enumerate(slacks):
            csl_problem.lambdas[i].grad = -slack
        
        self.primal_optimizer.step()
        self.dual_optimizer.step()

        # project dual variables onto the non-negative orthant
        for i in range(len(csl_problem.lambdas)):
            csl_problem.lambdas[i][csl_problem.lambdas[i] < 0] = 0

        return logging_dict
