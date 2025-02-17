# Baseline PINN for the Eikonal equaiton

import sys
import os
import argparse
from datetime import datetime

sys.path.append('.') 

import numpy as np
import torch
import torch.nn.functional as F

from scl.unsupervised.model import MLP
from scl.unsupervised.utils import *
from scl.unsupervised.visualize_solution import *
from scl.unsupervised.generate_data.load_data_eikonal import load_eikonal_data

parser = argparse.ArgumentParser(description='PINN baseline')

parser.add_argument('--seed', type=int, default=0, help='Random initialization.')
parser.add_argument('--num_collocation_pts', type=int, default=1000, help='Number of collocation points.')
parser.add_argument('--fixed_collocation_pts', action=argparse.BooleanOptionalAction, default=False, help='Whether to use fixed collocation points or sample new ones at each epoch.')
parser.add_argument('--epochs', type=int, default=100000, help='Number of epochs to train for.')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
parser.add_argument('--use_lr_scheduler', action=argparse.BooleanOptionalAction, default=True, help='Whether to use a learning rate scheduler.')
parser.add_argument('--ic_loss_weight', type=float, default=500.0, help='Weight for the data loss.')
parser.add_argument('--bc_loss_weight', type=float, default=10.0, help='Weight for the data loss.')
parser.add_argument('--layers', type=str, default='128,128,128,128,1', help='Dimensions of layers of the NN (except the first layer).')

parser.add_argument('--visualize', action=argparse.BooleanOptionalAction, default=True, help='Visualize the solution and prediction of the model.')
parser.add_argument('--save_model', action=argparse.BooleanOptionalAction, default=True)
parser.add_argument('--eval_every', type=int, default=1, help='Evaluate the model every n epochs.')
parser.add_argument('--run_location', type=str, default='local', help='Where the script is executed', choices=['local', 'horeka', 'bro_cluster'])


class PINN:
    def __init__(
            self,
            args, 
            model, 
            X_train_pde,
            X_bc,
            X_ic,
            device):

        self.model = model
        self.fixed_collocation_pts = args.fixed_collocation_pts
        self.num_collocation_pts = args.num_collocation_pts
        self.device = device

        # data
        self.x_pde = torch.tensor(X_train_pde[:, 0:1], requires_grad=True).float().to(device)
        self.y_pde = torch.tensor(X_train_pde[:, 1:2], requires_grad=True).float().to(device)
        self.x_ic = torch.tensor(X_ic[:, 0:1]).float().to(device)
        self.y_ic = torch.tensor(X_ic[:, 1:2]).float().to(device)
        self.x_bc = torch.tensor(X_bc[:, 0:1]).float().to(device)
        self.y_bc = torch.tensor(X_bc[:, 1:2]).float().to(device)

    def forward(self, x, y):
        """Make a forward pass throught the model. The model takes (x,t) as input and predicts the value of the funcion, u."""
        u = self.model(torch.cat([x, y], dim=1))
        return u

    def pde_residual(self, x, y):
        """ Autograd for calculating the residual for different PDE systems of the model. I.e. if the model satisfies the PDE"""
        u = self.forward(x, y)

        u_x = torch.autograd.grad(
            u, 
            x,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        u_y = torch.autograd.grad(
            u, 
            y,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        
        residual = u_x ** 2 + u_y ** 2 - 1
        return residual
    
    def data_loss(self):
        """Computes the data loss. The data loss is computed for points on the 
        boundary."""
        loss = self.bc_loss() + self.ic_loss()
        return loss
    
    def ic_loss(self):
        u_pred = self.forward(self.x_ic, self.y_ic)
        loss_ic = (u_pred ** 2).mean()  # Signed Distance Field (SDF) = 0 on zero contour set
        return loss_ic

    def bc_loss(self):
        u_pred = self.forward(self.x_bc, self.y_bc)
        loss_bc = (F.relu(-u_pred)).mean()  # Signed Distance Field (SDF) should be positive on boundary
        return loss_bc
    
    def pde_loss_fixed(self):
        """Computes the PDE loss, i.e. how well the model satisfies the underlying PDE
        for fixed collocation points."""
        residual_pde = self.pde_residual(self.x_pde, self.t_pde)
        loss_pde = torch.mean(residual_pde ** 2)
        return loss_pde
    
    def pde_loss_stochastic(self):
        """Computes the PDE loss, i.e. how well the model satisfies the underlying PDE
        for stochastic collocation points."""

        # sample collocation points for x, y in the range (-1, 1)
        x_pde = torch.rand(self.num_collocation_pts) * 2 - 1
        y_pde = torch.rand(self.num_collocation_pts) * 2 - 1

        x_pde = x_pde.reshape(-1, 1).float().to(self.device)
        y_pde = y_pde.reshape(-1, 1).float().to(self.device)

        x_pde.requires_grad_(True)
        y_pde.requires_grad_(True)

        residual_pde = self.pde_residual(x_pde, y_pde)
        loss_pde = torch.mean(residual_pde ** 2)
        return loss_pde
    
    def pde_loss(self):
        if self.fixed_collocation_pts:
            return self.pde_loss_fixed()
        else:
            return self.pde_loss_stochastic()
    
    def predict(self, X):
        """Make predictions. Used during evaluation."""
        x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(self.device)
        y = torch.tensor(X[:, 1:2], requires_grad=True).float().to(self.device)

        self.model.eval()
        u = self.forward(x, y)
        u = u.detach().cpu().numpy()

        return u
    
    def predict_pde_residual(self, X):
        x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(self.device)
        y = torch.tensor(X[:, 1:2], requires_grad=True).float().to(self.device)
        
        self.model.eval()
        pde_residual = self.pde_residual(x, y)
        pde_residual = pde_residual.detach().cpu().numpy()

        return pde_residual
    

def main():
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # layers of NN
    layers = [int(item) for item in args.layers.split(',')]

    # get data
    X_train_pde, X_bc, X_ic, X_test, u_exact, x, y, u_exact_1d = load_eikonal_data(args)
    
    # add first layer with correct number of input nodes
    layers.insert(0, X_test.shape[-1])

    # set set for reproducibility
    set_seed(args.seed)  
    
    model = MLP(layers, 'tanh').to(device)
    pinn = PINN(
        args, 
        model, 
        X_train_pde,
        X_bc,
        X_ic,
        device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.use_lr_scheduler:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.9)

    rel_error = []
    abs_error = []
    linf_error = []
    loss_list = []
    
    # for early stopping
    best_loss = np.inf
    best_model = None

    # for saving
    now = datetime.now()
    date_time_string = now.strftime("%Y-%m-%d %H:%M:%S")
    suffix = f'baseline_pinn_eikonal'
    dir_name = date_time_string  + '_' + suffix

    path_save = f'saved/eikonal/{dir_name}'

    if not os.path.exists(path_save):
        os.makedirs(path_save)

    model.train()
    for e in range(args.epochs):
        optimizer.zero_grad()
        loss = args.ic_loss_weight * pinn.ic_loss() + args.bc_loss_weight * pinn.bc_loss() + pinn.pde_loss()
        loss.backward()
        optimizer.step()

        if args.use_lr_scheduler:
            lr_scheduler.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_model = pinn.model
            epoch = e
            torch.save(best_model.state_dict(), f'{path_save}/best_model.pt')
            
            u_pred = pinn.predict(X_test)
            best_rel_l2_error = np.linalg.norm(u_exact_1d-u_pred, 2)/np.linalg.norm(u_exact, 2)
        
        loss_list.append(loss.item())

        # eval
        if e % args.eval_every == 0:
            u_pred = pinn.predict(X_test)
            error_u_relative = np.linalg.norm(u_exact_1d-u_pred, 2)/np.linalg.norm(u_exact, 2)
            error_u_abs = np.mean(np.abs(u_exact_1d - u_pred))
            error_u_linf = np.linalg.norm(u_exact_1d - u_pred, np.inf)/np.linalg.norm(u_exact, np.inf)

            rel_error.append(error_u_relative)
            abs_error.append(error_u_abs)
            linf_error.append(error_u_linf)

    # test model and print metrics
    u_pred = pinn.predict(X_test)

    error_u_relative = np.linalg.norm(u_exact_1d - u_pred, 2) / np.linalg.norm(u_exact, 2)
    error_u_abs = np.mean(np.abs(u_exact_1d - u_pred))
    error_u_linf = np.linalg.norm(u_exact_1d - u_pred, np.inf) / np.linalg.norm(u_exact, np.inf)

    print('Test metrics at final epoch:')
    print('Relative error: %e' % (error_u_relative))
    print('Absolute error: %e' % (error_u_abs))
    print('L_inf error: %e' % (error_u_linf))
    print('')

    idx_best = int(epoch / args.eval_every)
    print('Test metrics for epoch with the lowest loss:')
    print('Relative error: %e' % (rel_error[idx_best]))
    print('Absolute error: %e' % (abs_error[idx_best]))
    print('L_inf error: %e' % (linf_error[idx_best]))
    print('')

    # visualize PINN solution
    if args.save_model:
        torch.save(pinn.model.state_dict(), f"{path_save}/model.pt")
        torch.save(optimizer.state_dict(), f'{path_save}/optimizer.pt')

    if args.visualize:
        if not os.path.exists(path_save):
            os.makedirs(path_save)
        u_pred = u_pred.reshape(len(y), len(x))
        plot_exact_u(u_exact, x, y, path_save, label_x='x', label_y='y', flip_axis_plotting=False, cmap='coolwarm')
        plot_u_diff(u_exact, u_pred, x, y, path_save, label_x='x', label_y='y', flip_axis_plotting=False)
        plot_u_pred(u_exact, u_pred, x, y, path_save, label_x='x', label_y='y', flip_axis_plotting=False, cmap='coolwarm')

        pde_residual = pinn.predict_pde_residual(X_test)
        pde_residual = pde_residual.reshape(len(y), len(x))
        visualize(u_exact, u_pred, pde_residual, path_save)

    # save arguments
    with open(f'{path_save}/args.txt', 'w') as file:
        for key, value in vars(args).items():
            file.write(f'{key}: {value}\n')
        
        # save error metrics
        file.write(f'Relative error: {error_u_relative}\n')
        file.write(f'Absolute error: {error_u_abs}\n')
        file.write(f'linf error: {error_u_linf}\n')

        file.write(f'Best relative error: {best_rel_l2_error}\n')
        file.write(f'Best epoch: {epoch}\n')
        file.write(f'Best relative error: {rel_error[idx_best]}\n')
        file.write(f'Best absolute error: {abs_error[idx_best]}\n')
        file.write(f'Best L_inf error: {linf_error[idx_best]}\n')

    # plot errors
    plot_errors(rel_error, abs_error, linf_error, args.epochs, args.eval_every, path_save)

    # save errors
    np.save(f'{path_save}/relative_error.npy', rel_error)
    np.save(f'{path_save}/absolute_error.npy', abs_error)
    np.save(f'{path_save}/linf_error.npy', linf_error)

    # save loss
    np.save(f'{path_save}/loss.npy', loss_list)

    # save u_pred
    np.save(f'{path_save}/u_pred.npy', u_pred.T)
    
    # best model
    print('Best model at epoch:', epoch)
    print('Best relative L2 error:', best_rel_l2_error)
    

if __name__ == '__main__':
    import time
    start = time.time()
    main()
    print('Time taken:', time.time() - start)
    print('Done!')
