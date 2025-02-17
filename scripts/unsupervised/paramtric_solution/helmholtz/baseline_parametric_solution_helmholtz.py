# Baseline parametric solution (PI / Cho et al., 2024) for Helmholtz equation.
# Loss terms (data + PDE) are aggregated for different (a1,a2) tuples in the loss function.
# Want to find NN f(x,t,a1,a2) that approximates solution everywhere for all (a1,a2) in the domain
# NOTE: pde_param refers to the tuple of PDE parameters (a1,a2) that defines the reaction-diffusion PDE.


import sys
import os
import argparse
from datetime import datetime
import itertools
import math
import pickle

sys.path.append('.') 

import numpy as np
import torch

from scl.unsupervised.model import MLP

from scl.unsupervised.generate_data.generate_data_helmholtz import *
from scl.unsupervised.utils import *
from scl.unsupervised.visualize_solution import *

parser = argparse.ArgumentParser(description='PINN')

parser.add_argument('--seed', type=int, default=0, help='Random initialization.')
parser.add_argument('--num_collocation_pts', type=int, default=1000, help='Number of collocation points.')
parser.add_argument('--fixed_collocation_pts', action=argparse.BooleanOptionalAction, default=False, help='Whether to use fixed collocation points or sample new ones at each epoch.')
parser.add_argument('--epochs', type=int, default=200000, help='Number of epochs to train for.')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--use_lr_scheduler', action=argparse.BooleanOptionalAction, default=True, help='Whether to use a learning rate scheduler.')
parser.add_argument('--data_loss_weight', type=float, default=100.0, help='Weight for the data loss.')
parser.add_argument('--layers', type=str, default='50,50,50,50,1', help='Dimensions of layers of the NN (except the first layer).')

parser.add_argument('--nx', type=int, default=256, help='Number of points in the x grid.')
parser.add_argument('--ny', type=int, default=256, help='Number of points in the y grid.')
parser.add_argument('--a1_train', nargs='+', default=[1.0, 1.5, 2.0], help='a1 used for training.')
parser.add_argument('--a2_train', nargs='+', default=[1.0, 1.5, 2.0], help='a2 used for training.')
parser.add_argument('--a1_test', nargs='+', default=[1.0, 1.25, 1.75], help='a1 for testing after training.')
parser.add_argument('--a2_test', nargs='+', default=[1.0, 1.25, 1.75], help='a2 for testing after training.')
parser.add_argument('--k', type=float, default=1, help='k value for the Helmholtz PDE.')

parser.add_argument('--visualize', action=argparse.BooleanOptionalAction, default=True, help='Visualize the solution and prediction of the model.')
parser.add_argument('--plot_diagnostics', action=argparse.BooleanOptionalAction, default=True, help='Plot diagnostics of the model.')
parser.add_argument('--save_model', action=argparse.BooleanOptionalAction, default=True)
parser.add_argument('--eval_every', type=int, default=100, help='Evaluate the model every n epochs.') 


class BaselineParametricSolution:
    def __init__(
            self,
            args, 
            model, 
            X_train_pde_list,
            bc_list,
            bc_u_list,
            device):
        
        self.num_collocation_pts = args.num_collocation_pts
        self.fixed_collocation_pts = args.fixed_collocation_pts
        self.model = model
        self.device = device     

        a1_train = [float(a1) for a1 in args.a1_train]
        a2_train = [float(a2) for a2 in args.a2_train]
        self.k = args.k

        # computes the cartesian product (all ordered combinations) that we train on
        self.pde_params = torch.tensor(list(itertools.product(a1_train, a2_train))).float().to(device)
        
        self.n_combinations = len(self.pde_params)

        # data (stack data for different pde parameters so that we can process in batches)
        X_train_pde_stacked = np.stack(X_train_pde_list, axis=0)
        self.x_pde = torch.tensor(X_train_pde_stacked[:, :, 0:1]).requires_grad_(True).float().to(device)
        self.y_pde = torch.tensor(X_train_pde_stacked[:, :, 1:2]).requires_grad_(True).float().to(device)

        bc_stacked = np.stack(bc_list, axis=0)
        self.x_bc = torch.tensor(bc_stacked[:, :, 0:1]).float().to(device)
        self.y_bc = torch.tensor(bc_stacked[:, :, 1:2]).float().to(device)

        bc_u_stacked = np.stack(bc_u_list, axis=0)
        self.u_bc = torch.tensor(bc_u_stacked).float().to(device)

    def forward(self, x, y, pde_param):
        """Make a forward pass throught the model. The model takes (x,t) as input and predicts the value of the funcion, u."""
        N = x.shape[0]                                                # number of points
        pde_param = pde_param.reshape(1, -1).repeat(N, 1)             # shape of pde_param: (2,) --> (N, 2)
        u = self.model(torch.cat([x, y, pde_param], dim=1))
        return u
    
    def forward_batch(self, x, y):
        """Make a forward pass throught the model. The model takes (x,t) as input and predicts the value of the funcion, u."""
        N = x.shape[1]      # number of points
        pde_params = self.pde_params.unsqueeze(1).repeat(1, N, 1)
        u = self.model(torch.cat([x, y, pde_params], dim=-1))
        return u
    
    def pde_residual(self, x, y):
        """ Autograd for calculating the PDE residual, i.e., if the model satisfies the PDE"""
        u = self.forward_batch(x, y)

        u_x = torch.autograd.grad(
            u, x,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        u_y = torch.autograd.grad(
            u, y,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]

        u_xx = torch.autograd.grad(
            u_x, x,
            grad_outputs=torch.ones_like(u_x),
            retain_graph=True,
            create_graph=True
            )[0]
        
        u_yy = torch.autograd.grad(
            u_y, y,
            grad_outputs=torch.ones_like(u_y),
            retain_graph=True,
            create_graph=True
            )[0]

        a1 = self.pde_params[:, 0].view(-1, 1, 1)
        a2 = self.pde_params[:, 1].view(-1, 1, 1)
        q = self.get_forcing(x, y, a1, a2)

        residual = u_xx + u_yy + (self.k ** 2) * u - q
        
        return residual
    
    def get_forcing(self, x, y, a1, a2):
        """Get the forcing term for the 2D Helmholtz PDE."""
        q = - ((a1 * math.pi) ** 2) * torch.sin(a1 * math.pi * x)*torch.sin(a2 * math.pi * y) \
                - ((a2 * math.pi) ** 2) * torch.sin(a1 * math.pi * x) * torch.sin(a2 * math.pi * y) \
                + (self.k ** 2) * torch.sin(a1 * math.pi * x) * torch.sin(a2 * math.pi * y)
        return q
    
    def data_loss(self):
        """Computes the data loss. The data loss is computed for points on the 
        boundary."""
        u_pred_bc = self.forward_batch(self.x_bc, self.y_bc)
        loss_bc = torch.mean((u_pred_bc - self.u_bc) ** 2, dim=-2)
        loss_bc = sum(loss_bc)
        return loss_bc
    
    def pde_loss_fixed(self):
        """Computes the PDE loss, i.e. how well the model satisfies the underlying PDE
        for fixed collocation points."""
        residual_pde = self.pde_residual(self.x_pde, self.y_pde)

        # average over all points and sum over all pde parameters
        loss_pde = torch.mean(residual_pde ** 2, dim=-2)
        loss_pde = sum(loss_pde)
        return loss_pde
    
    def pde_loss_stochastic(self):
        """Computes the PDE loss, i.e. how well the model satisfies the underlying PDE
        for stochastic collocation points."""

        # sample collocation points for x, y in the range (-1, 1)
        x_pde = torch.rand(self.num_collocation_pts) * 2 - 1
        y_pde = torch.rand(self.num_collocation_pts) * 2 - 1

        x_pde = x_pde.reshape(1, -1, 1).repeat(self.n_combinations, 1, 1).float().to(self.device)
        y_pde = y_pde.reshape(1, -1, 1).repeat(self.n_combinations, 1, 1).float().to(self.device)

        x_pde.requires_grad_(True)
        y_pde.requires_grad_(True)

        residual_pde = self.pde_residual(x_pde, y_pde)

        # average over all collocation points and sum over all pde parameters
        loss_pde = torch.mean(residual_pde ** 2, dim=-2)
        loss_pde = sum(loss_pde)
        return loss_pde
    
    def pde_loss(self):
        if self.fixed_collocation_pts:
            return self.pde_loss_fixed()
        else:
            return self.pde_loss_stochastic()
    
    def predict(self, X, pde_param):
        """Make predictions. Used during evaluation."""
        x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(self.device)
        y = torch.tensor(X[:, 1:2], requires_grad=True).float().to(self.device)
        pde_param = torch.tensor(pde_param).float().to(self.device)

        self.model.eval()
        u = self.forward(x, y, pde_param)
        u = u.detach().cpu().numpy()

        return u
    

def main():
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    a1_train = [float(a1) for a1 in args.a1_train]
    a2_train = [float(a2) for a2 in args.a2_train]
    a1_test = [float(a1) for a1 in args.a1_test]
    a2_test = [float(a2) for a2 in args.a2_test]

    # layers of NN
    layers = [int(item) for item in args.layers.split(',')]

    # computes the cartesian product (all ordered combinations)
    pde_params_train = list(itertools.product(a1_train, a2_train))  
    pde_params_test = list(itertools.product(a1_test, a2_test)) 

    # get data by solving the PDE
    data_train = {}
    X_train_pde_list = []
    bc_list = []
    bc_u_list = []
    for a1, a2 in pde_params_train:
        X_train_pde, bc, bc_u, X_test, u_exact, x, y, u_exact_1d = generate_helmholtz_data(args, a1, a2)
        data_pde_param = {'X_train_pde': X_train_pde, 'bc': bc, 'bc_u': bc_u, 'X_test': X_test, 'u_exact': u_exact, 'x': x, 'y': y, 'u_exact_1d': u_exact_1d}
        data_train[(a1, a2)] = data_pde_param

        X_train_pde_list.append(X_train_pde)
        bc_list.append(bc)
        bc_u_list.append(bc_u)

    data_test = {}
    for a1, a2 in pde_params_test:
        X_train_pde, bc, bc_u, X_test, u_exact, x, y, u_exact_1d = generate_helmholtz_data(args, a1, a2)
        data_pde_param = {'X_train_pde': X_train_pde, 'bc': bc, 'bc_u': bc_u, 'X_test': X_test, 'u_exact': u_exact, 'x': x, 'y': y, 'u_exact_1d': u_exact_1d}
        data_test[(a1, a2)] = data_pde_param

    # add first layer with correct number of input nodes
    layers.insert(0, X_test.shape[-1] + 2)     # +2 for (a1, a2)

    # set set for reproducibility
    set_seed(args.seed)  

    model = MLP(layers, 'tanh').to(device)
    pinn = BaselineParametricSolution(
        args,
        model,
        X_train_pde_list,
        bc_list,
        bc_u_list,
        device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # lr schedulers
    if args.use_lr_scheduler:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.9)

    rel_error = {f'Relative Error PDE Parameter: {pde_param}': [] for pde_param in pde_params_test}
    abs_error = {f'Absolute Error PDE Parameter: {pde_param}': [] for pde_param in pde_params_test}
    linf_error = {f'L_inf Error PDE Parameter: {pde_param}': [] for pde_param in pde_params_test}

    model.train()
    for e in range(args.epochs):
        optimizer.zero_grad()
        loss = args.data_loss_weight * pinn.data_loss() + pinn.pde_loss()
        loss.backward()
        optimizer.step()

        if args.use_lr_scheduler:
            lr_scheduler.step()

        # eval
        if e % args.eval_every == 0:
            for pde_param in pde_params_test:
                X_test = data_test[pde_param]['X_test']
                u_exact_1d = data_test[pde_param]['u_exact_1d']

                u_pred = pinn.predict(X_test, pde_param)
                error_u_relative = np.linalg.norm(u_exact_1d - u_pred, 2) / np.linalg.norm(u_exact_1d, 2)
                error_u_abs = np.mean(np.abs(u_exact_1d - u_pred))
                error_u_linf = np.linalg.norm(u_exact_1d - u_pred, np.inf) / np.linalg.norm(u_exact_1d, np.inf)

                rel_error[f'Relative Error PDE Parameter: {pde_param}'].append(error_u_relative)
                abs_error[f'Absolute Error PDE Parameter: {pde_param}'].append(error_u_abs)
                linf_error[f'L_inf Error PDE Parameter: {pde_param}'].append(error_u_linf)

    # make predictions
    preds = {}
    for pde_param in data_test.keys():
        X_test = data_test[pde_param]['X_test']
        u_pred = pinn.predict(X_test, pde_param)
        preds[pde_param] = u_pred

    # visualize PINN solution and plot diagnostics
    now = datetime.now()
    date_time_string = now.strftime("%Y-%m-%d %H:%M:%S")
    suffix = f'baseline_parametric_solution_helmholtz'
    dir_name = date_time_string  + '_' + suffix

    path_save = f'saved/helmholtz/{dir_name}'

    if args.save_model:
        if not os.path.exists(path_save):
            os.makedirs(path_save)
        torch.save(pinn.model.state_dict(), f"{path_save}/model.pt")
        torch.save(optimizer.state_dict(), f'{path_save}/optimizer.pt')

    if args.visualize:
        if not os.path.exists(path_save):
            os.makedirs(path_save)

        for pde_param in data_test.keys():
            u_pred_pde_param = preds[pde_param]
            u_pred_pde_param = u_pred_pde_param.reshape(len(y), len(x))
            u_exact_pde_param = data_test[pde_param]['u_exact']
            u_exact_1d_pde_param = data_test[pde_param]['u_exact_1d']

            plot_exact_u(u_exact_pde_param, x, y, path_save, label_x='x', label_y='y', suffix=f'_{pde_param}', flip_axis_plotting=False)
            plot_u_diff(u_exact_pde_param, u_pred_pde_param, x, y, path_save, label_x='x', label_y='y', suffix=f'_{pde_param}', flip_axis_plotting=False)
            plot_u_pred(u_exact_1d_pde_param, u_pred_pde_param, x, y, path_save, label_x='x', label_y='y', suffix=f'_{pde_param}', flip_axis_plotting=False)
    
    # save arguments
    with open(f'{path_save}/args.txt', 'a') as file:
        for key, value in vars(args).items():
            file.write(f'{key}: {value}\n')

    # test model and print metrics
    for pde_param in data_test.keys():
        u_exact_1d_pde_param = data_test[pde_param]['u_exact_1d']
        u_pred_pde_param = preds[pde_param]
        u_pred_pde_param = u_pred_pde_param.reshape(-1,1)

        error_u_relative = np.linalg.norm(u_exact_1d_pde_param - u_pred_pde_param, 2) / np.linalg.norm(u_exact_1d_pde_param, 2)
        error_u_abs = np.mean(np.abs(u_exact_1d_pde_param - u_pred_pde_param))
        error_u_linf = np.linalg.norm(u_exact_1d_pde_param - u_pred_pde_param, np.inf) / np.linalg.norm(u_exact_1d_pde_param, np.inf)

        print(f'pde_param: {pde_param}')
        print('Test metrics:')
        print('Relative error: %e' % (error_u_relative))
        print('Absolute error: %e' % (error_u_abs))
        print('L_inf error: %e' % (error_u_linf))
        print('')
        
        with open(f'{path_save}/args.txt', 'a') as file:
            file.write(f'pde_param: {pde_param}\n')
            file.write('Test metrics:\n')
            file.write(f'Relative error: {error_u_relative}\n')
            file.write(f'Absolute error: {error_u_abs}\n')
            file.write(f'L_inf error: {error_u_linf}\n')

    # plot errors
    plot_errors_parametric_solution(rel_error, abs_error, linf_error, args.epochs, args.eval_every, pde_params_test, path_save)

    # # save relative error
    # np.save(f'{path_save}/relative_error.npy', rel_error)

    # saver errors
    with open(f'{path_save}/rel_error.pkl', 'wb') as f:
        pickle.dump(rel_error, f)
    with open(f'{path_save}/abs_error.pkl', 'wb') as f:
        pickle.dump(abs_error, f)
    with open(f'{path_save}/linf_error.pkl', 'wb') as f:
        pickle.dump(linf_error, f)


if __name__ == '__main__':
    import time
    start = time.time()
    main()
    print('Time taken:', time.time() - start)
    print('Done!')
