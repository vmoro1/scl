# SCL(O) for Navier Stokes equation. This is done by solving the constrained learning problem 
# where each sample is a constraint (i.e., we have per sample constraints) instead of averaging them.


import sys
import os
import argparse
from datetime import datetime
import pickle

import torch.utils

sys.path.append('.')
sys.path.append('/slurm-storage/vigmor/scl')
sys.path.append('/home/gridsan/vmoro/scl')

import torch
import wandb

from scl.supervised.csl_problem import ConstrainedStatisticalLearningProblem
from scl.supervised.solver import SimultaneousPrimalDual
from scl.supervised.dataset import DatasetPerSampleConstraints
from scl.supervised.neuraloperator.neuralop.models import FNO
from scl.supervised.utils import *

parser = argparse.ArgumentParser(description='SCL(O)')

parser.add_argument('--seed', type=int, default=0, help='Random initialization.')
parser.add_argument('--epochs', type=int, default=500, help='Number of epochs to train for')
parser.add_argument('--batch_size', type=int, default=20, help='Batch size for training.')
parser.add_argument('--lr_primal', type=float, default=1e-3, help='Learning rate for primal variables (NN parameters).')
parser.add_argument('--lr_dual', type=float, default=1e-4, help='Learning rate for dual variables (lambdas).')
parser.add_argument('--eps', nargs='+', default=[1e-2], help='Tolerances for the constraints.')
parser.add_argument('--per_sample_eps', nargs='+', default=[1e-2], help='Tolerances for the per sample constraints. All samples for a per sample constraint share the same tolerance.')
parser.add_argument('--use_primal_lr_scheduler', action=argparse.BooleanOptionalAction, default=True, help='Whether to use a learning rate scheduler for the primal variables.')
parser.add_argument('--use_dual_lr_scheduler', action=argparse.BooleanOptionalAction, default=True, help='Whether to use a learning rate scheduler for the dual variables.')

parser.add_argument('--viscosity', type=float, choices=[1e-3, 1e-4, 1e-5], default=1e-3, help='Viscosity of the fluid.')
parser.add_argument('--n_train', type=int, default=800, help='Number of training samples.')
parser.add_argument('--n_validation', type=int, default=200, help='Number of validation samples.')
parser.add_argument('--n_test', type=int, default=200, help='Number of test samples.')

parser.add_argument('--n_modes', type=int, default=8, help='Number of Fourier modes to use.')
parser.add_argument('--hidden_channels', type=int, default=64, help='Width of the FNO(i.e. number of channels)')  
parser.add_argument('--projection_channels', type=int, default=128, help='Number of hidden channels of the projection back to the output')
parser.add_argument('--n_layers', type=int, default=8, help='Number of Fourier layers to use.')

parser.add_argument('--save_model', default=True)
parser.add_argument('--eval_every', type=int, default=1, help='Evaluate the model every n epochs.')
parser.add_argument('--visualize', default=False, help='Visualize the solution and prediction of the model.')
parser.add_argument('--plot_diagnostics', default=True, help='Plot diagnostics of the model.')
parser.add_argument('--wandb_project_name', type=str, default='scl_supervised')   
parser.add_argument('--wandb_run_name', type=str, default='scl_o_navier_stokes_per_sample_constraints')
parser.add_argument('--run_location', choices=['locally', 'supercloud', 'horeka', 'brocluster'], default='locally', help='Choose where the script is executed.')


class SCL_O(ConstrainedStatisticalLearningProblem):
    def __init__(self, model, n_train_samples, args):
        self.model = model
        self.device = next(model.parameters()).device
        self.relative_L2_loss_fn = LpLoss()
        self.per_sample_relative_L2_loss_fn = LpLoss(size_average=False, reduction=False)

        eps = [float(eps) for eps in args.eps]

        # define objective, average constraints, per sample constraints and the the 
        # right hand side (tolerance) for all constraints (i.e. the c in f_i(x) < c)
        self.objective_function = lambda: torch.tensor(1.0, dtype=float).to(self.device)
        self.per_sample_constraints = [self.per_sample_data_loss]
        self.per_sample_rhs = [torch.full((n_train_samples,), float(per_sample_tol), device=self.device) for per_sample_tol in args.per_sample_eps]

        super().__init__()

    def forward(self, x):
        self.y_pred = self.model(x).squeeze(1)
        self.x = x
        self.prediction_stored = True
        self.input_stored = True

    def store_targets_and_samples_idx(self, sample_idx, y):
        self.y = y
        self.target_stored = True
        self.samples_indices = sample_idx
        self.samples_indices_stored = True

    def reset(self):
        self.prediction_computed = False
        self.y_pred = None
        self.input_stored = False
        self.x = None
        self.target_stored = False
        self.y = None
        self.samples_indices_stored = False
        self.samples_indices = None

    def data_loss(self):
        """Computes the data loss (relative L2 loss)"""
        data_loss = self.relative_L2_loss_fn(self.y_pred, self.y)
        return data_loss
    
    def per_sample_data_loss(self):
        """Computes the per sample loss for the data, i.e. the relative L2 loss."""
        per_sample_data_loss = self.per_sample_relative_L2_loss_fn(self.y_pred, self.y)
        return per_sample_data_loss


def main():
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(args.seed)

    if args.run_location == 'locally':
        path_base = ""
    elif args.run_location == 'supercloud':
        path_base = "/home/gridsan/vmoro/scl/"
    elif args.run_location == 'horeka':
        path_base = "/home/hk-project-test-p0021798/st_ac144859/scl/"
    elif args.run_location == 'brocluster':
        path_base = "/slurm-storage/vigmor/scl/"
    else:
        raise ValueError('Invalid run location')

    date_time = datetime.now()
    date_time_string = date_time.strftime("%Y-%m-%d %H:%M:%S")
    name_date = f"{args.wandb_run_name}_{date_time_string}"

    # wandb logger
    path_wandb = path_base + "saved/wandb_logs"
    if not os.path.exists(path_wandb):
        os.makedirs(path_wandb)
    config = vars(args)
    os.environ["WANDB_API_KEY"] = '2ba73edbdcfc28a2f25e4fb15eb9f9d6b5af890f'
    os.environ["WANDB_MODE"] = 'offline'
    os.environ["WANDB_DIR"] = path_wandb
    
    wandb.init(
        project=args.wandb_project_name, 
        name=f"{args.wandb_run_name}_{name_date}",
        entity="viggomoro",
        config=config
        )
    
    # # TODO: remember to change this
    # args.n_train = 4
    # args.n_validation = 2
    # args.n_test = 2
    # args.batch_size = 2
    # args.epochs = 2

    # load data - the first time step is used to predict all other time steps (including the first one)
    if args.viscosity == 1e-3:
        n_spatial = 64      # spatial grid size
        n_temporal = 50     # temporal grid size
        
        data_path = path_base + 'data/supervised/navier_stokes/ns_V1e-3_N5000_T50.mat'
        data_reader = MatReader(data_path)
        data = data_reader.read_field('u')
        data = data.permute(0, 3, 1, 2)          # new shape: (number of samples, n_temporal, n_spatial, n_spatial)

        # data_path = path_base + 'data/navier_stokes/ns_solution_subset_v1e-3_T50.pt'
        # data = torch.load(data_path)           # shape: (number of samples, n_temporal, n_spatial, n_spatial). Each sample is w(t, x, y) values for discretized grid
        # data = data.permute(0, 3, 1, 2)        # new shape: (number of samples, n_temporal, n_spatial, n_spatial)
    elif args.viscosity == 1e-4:
        n_spatial = 64      # spatial grid size
        n_temporal = 30     # temporal grid size    # NOTE: cane be increased to 50 (or reduced further)

        data_path = path_base + 'data/supervised/navier_stokes/ns_V1e-4_N10000_T30.mat'
        data_reader = MatReader(data_path)
        data = data_reader.read_field('u')
        data = data.permute(0, 3, 1, 2)
        data = data[:, :n_temporal, :, :]       # new shape: (number of samples, n_temporal, n_spatial, n_spatial)
    elif args.viscosity == 1e-5:
        n_spatial = 64      # spatial grid size
        n_temporal = 20     # temporal grid size

        data_path = path_base + 'data/supervised/navier_stokes/NavierStokes_V1e-5_N1200_T20.mat'
        data_reader = MatReader(data_path)
        data = data_reader.read_field('u')
        data = data.permute(0, 3, 1, 2)         # new shape: (number of samples, n_temporal, n_spatial, n_spatial)
    else:
        raise ValueError('Invalid viscosity')

    x_data = data[:, 0:1, :, :]               # shape: (number of samples, 1, n_spatial, n_spatial). Each sample is w(0, x, y) values for discretized grid
    y_data = data[:, :, :, :]                 # shape: (number of samples, n_temporal, n_spatial, n_spatial). Each sample is w(t, x, y) values for discretized grid

    x_data = x_data.repeat([1, n_temporal, 1, 1])     # shape: (number of samples, n_temporal, n_spatial, n_spatial)

    # add positional encoding (i.e. x, y and t coordinates) to the input data
    grid_x_1d = torch.tensor(np.linspace(0, 1, n_spatial + 1)[:-1], dtype=torch.float)
    grid_y_1d = torch.tensor(np.linspace(0, 1, n_spatial + 1)[:-1], dtype=torch.float)
    gridt_1d = torch.tensor(np.linspace(1, n_temporal, n_temporal), dtype=torch.float)

    n_samples = x_data.shape[0]
    grid_x = grid_x_1d.reshape(1, 1, 1, n_spatial, 1).repeat([n_samples, 1, n_temporal, 1, n_spatial])
    grid_y = grid_y_1d.reshape(1, 1, 1, 1, n_spatial).repeat([n_samples, 1, n_temporal, n_spatial, 1])
    gridt = gridt_1d.reshape(1, 1, n_temporal, 1, 1).repeat([n_samples, 1, 1, n_spatial, n_spatial])

    x_data = x_data.unsqueeze(1)                                    # add channel dim. shape: (number of samples, 1, n_temporal, n_spatial, n_spatial)
    x_data = torch.cat([x_data, grid_x, grid_y, gridt], dim=1)      # shape: (number of samples, 4, n_temporal, n_spatial, n_spatial)

    x_train = x_data[:args.n_train]        
    y_train = y_data[:args.n_train]
    x_validation = x_data[args.n_train:args.n_train + args.n_validation]
    y_validation = y_data[args.n_train:args.n_train + args.n_validation]   
    x_test = x_data[-args.n_test:]
    y_test = y_data[-args.n_test:]        

    train_loader = torch.utils.data.DataLoader(DatasetPerSampleConstraints(x_train, y_train), batch_size=args.batch_size, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(DatasetPerSampleConstraints(x_validation, y_validation), batch_size=args.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(DatasetPerSampleConstraints(x_test, y_test), batch_size=args.batch_size, shuffle=False)

    model = FNO(
        n_modes=(args.n_modes, args.n_modes, args.n_modes), 
        hidden_channels=args.hidden_channels, 
        in_channels=4, 
        out_channels=1, 
        lifting_channels=False,
        projection_channels=args.projection_channels, 
        n_layers=args.n_layers)
    
    model = model.to(device)
    
    scl_o = SCL_O(model, args.n_train, args)

    optimizers = {'primal_optimizer': 'Adam',
                  'use_primal_lr_scheduler': args.use_primal_lr_scheduler,
                'dual_optimizer': 'Adam',
                'use_dual_lr_scheduler': args.use_dual_lr_scheduler}
    # save 
    name = 'scl_o_per_sample_constraints'
    path_save = f'saved/navier_stokes/{name}/{date_time_string}'

    if not os.path.exists(path_save):
        os.makedirs(path_save)

    solver = SimultaneousPrimalDual(
        csl_problem=scl_o,
        optimizers=optimizers,
        primal_lr=args.lr_primal,
        dual_lr=args.lr_dual,
        epochs=args.epochs,
        eval_every=args.eval_every,
        train_dataloader=train_loader,
        validation_loader=validation_loader,
        test_dataloader=test_loader,
        path_save=path_save
    )
    solver.solve(scl_o)

    # relative L2 error on test
    final_test_error = eval_model_per_sample_constraints(model, test_loader, device)
    test_errors = solver.state_dict['Relative_L2_test_error']

    print()
    print('-------Training complete-------')
    print('Final relative L2 error on test set: ', final_test_error)
    print('Best relative L2 error on test set: ', min(test_errors))
    print('Epoch with lowest relative L2 error on test set: ', test_errors.index(min(test_errors)) * args.eval_every)
    print()
    print(f'Epoch with lowest validation error: {solver.best_epoch}')
    print(f'Lowest validation error: {solver.best_validation_error}')
    print(f'Test error for epoch with lowest validation error: {solver.test_error_best_epoch}')
    print()


    # print final diagnostics
    print('Final diagnostics:')
    num_dual_variables = len(scl_o.lambdas)
    for i in range(num_dual_variables):
        print(f'Dual variable {i}: ', solver.state_dict['dual_variables'][f'dual_variable_{i}'][-1])
        print(f'Slack_{i}: ', solver.state_dict['slacks'][f'slack_{i}'][-1])
    print('Approximate duality gap: ', solver.state_dict['approximate_duality_gap'][-1])
    print('Approximate relative duality gap: ', solver.state_dict['aproximate_relative_duality_gap'][-1])
    print('Lagrangian: ', solver.state_dict['Lagrangian'][-1])
    print('Primal value: ', solver.state_dict['primal_value'][-1])
    print()

    if args.plot_diagnostics:
        solver.plot(path_save)

    if args.save_model:
        torch.save(model.state_dict(), path_save + '/model.pt')

    # save arguments
    with open(f'{path_save}/args.txt', 'w') as file:
        for key, value in vars(args).items():
            file.write(f'{key}: {value}\n')
        
        # save error metrics
        file.write(f'Best relative L2 error: {min(test_errors)}\n')
        file.write(f'Final relative L2 error: {final_test_error}\n')
        file.write(f'Epoch with lowest relative L2 error on test set: {test_errors.index(min(test_errors)) * args.eval_every}\n')
        file.write(f'Epoch with lowest validation error: {solver.best_epoch}\n')
        file.write(f'Lowest validation error: {solver.best_validation_error}\n')
        file.write(f'Test error for epoch with lowest validation error: {solver.test_error_best_epoch}\n')
        file.write('\n')

    # save state dict
    with open(f'{path_save}/state_dict.pkl', 'wb') as f:
        pickle.dump(solver.state_dict, f)

    # close logger
    wandb.finish()

if __name__ == '__main__':
    import time
    start = time.time()
    main()
    print('Time taken:', time.time() - start)
    print('Done!')
