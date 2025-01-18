# FNO baseline for Diffusion-Sorption

import sys
import os
import argparse
from datetime import datetime

sys.path.append('.')
sys.path.append('/home/gridsan/vmoro/CSL/csl_neuraloperator')
sys.path.append('/home/hk-project-test-p0021798/st_ac144859/csl_neuraloperator')

import torch
import wandb

from csl_neuraloperator.utils import *
from csl_neuraloperator.neuraloperator.neuralop.models import FNO
from csl_neuraloperator.pde_losses import DiffusionSorptionPDE_Loss

parser = argparse.ArgumentParser(description='FNO for Diffusion-Sorption')

parser.add_argument('--seed', type=int, default=0, help='Random initialization.')
parser.add_argument('--epochs', type=int, default=500, help='Number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=20, help='Batch size for training.')
parser.add_argument('--use_lr_scheduler', action=argparse.BooleanOptionalAction, default=True, help='Whether to use a learning rate scheduler.')

parser.add_argument('--pde_loss_method', type=str, default='finite_difference', help='Method used to compute the PDE loss/physics-informed loss (if the pde loss is used).')
parser.add_argument('--data_loss_weight', type=float, default=1.0, help='Weight for the data loss.')
parser.add_argument('--pde_loss_weight', type=float, default=0.0, help='Weight for the PDE loss.')
parser.add_argument('--ic_loss_weight', type=float, default=0.0, help='Weight for the initial condition loss.')
parser.add_argument('--bc_loss_weight', type=float, default=0.0, help='Weight for the boundary condition loss.')

parser.add_argument('--n_train', type=int, default=8000, help='Number of training samples.')
parser.add_argument('--n_test', type=int, default=2000, help='Number of test samples.')

parser.add_argument('--n_modes', type=int, default=8, help='Number of Fourier modes to use.')
parser.add_argument('--hidden_channels', type=int, default=64, help='Width of the FNO(i.e. number of channels)')
parser.add_argument('--projection_channels', type=int, default=128, help='Number of hidden channels of the projection back to the output')
parser.add_argument('--n_layers', type=int, default=5, help='Number of Fourier layers to use.')

parser.add_argument('--save_model', default=True)
parser.add_argument('--eval_every', type=int, default=1, help='Evaluate the model every n epochs.')
parser.add_argument('--visualize', default=False, help='Visualize the solution and prediction of the model.')
parser.add_argument('--wandb_project_name', type=str, default='csl_neuraloperator')   
parser.add_argument('--wandb_run_name', type=str, default='fno_baseline_diffsorp')
parser.add_argument('--run_location', choices=['locally', 'supercloud', 'horeka'], default='locally', help='Choose where the script is executed.')


def main():
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(args.seed)

    if args.run_location == 'locally':
        path_base = ""
    elif args.run_location == 'supercloud':
        path_base = "/home/gridsan/vmoro/CSL/csl_neuraloperator/"
    elif args.run_location == 'horeka':
        path_base = "/home/hk-project-test-p0021798/st_ac144859/csl_neuraloperator/"
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

    # load data
    nx = 1024           # spatial grid size
    nt = 101            # temporal grid size

    # TODO: remember to change this
    # n_train = 8000
    # n_test = 2000
    # n_train = 2
    # n_test = 2
    # args.batch_size = 2
    # args.epochs = 1

    data_path = path_base + 'data/diffusion_sorption/'
    data = torch.load(data_path + 'solution.pt')        # shape: (number of samples, nt, nx). Each sample is the solution u(t, x) for discretized grid
    x_grid = torch.load(data_path + 'x_grid.pt')
    t_grid = torch.load(data_path + 't_grid.pt')

    x_data = data[:, 0:1, :]                # shape: (number of samples, 1, nx). Each sample is u(0, x) for discretized grid
    x_data = x_data.repeat([1, nt, 1])      # shape: (number of samples, nt, nx)
    y_data = data[:, :, :]                  # shape: (number of samples, nt, nx). Each sample is u(t, x) for discretized grid

    # add positional encoding (i.e. x and t coordinates) to the input data
    n_samples = x_data.shape[0]
    grid_x = x_grid.reshape(1, 1, 1, nx).repeat([n_samples, 1, nt, 1])
    grid_t = t_grid.reshape(1, 1, nt, 1).repeat([n_samples, 1, 1, nx])

    x_data = x_data.unsqueeze(1)                            # add channel dim, shape: (number of samples, 1, nt, nx)
    x_data = torch.cat([x_data, grid_x, grid_t], dim=1)     # shape: (number of samples, 3, nt, nx)

    x_train = x_data[:args.n_train,:]        
    y_train = y_data[:args.n_train,:]        
    x_test = x_data[-args.n_test:,:]
    y_test = y_data[-args.n_test:,:]

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=args.batch_size, shuffle=False, pin_memory=True)

    model = FNO(
        n_modes=(args.n_modes, args.n_modes), 
        hidden_channels=args.hidden_channels, 
        in_channels=3, 
        out_channels=1, 
        lifting_channels=False,
        projection_channels=args.projection_channels, 
        n_layers=args.n_layers)
    
    model = model.to(device)
    
    relative_L2_loss_fn = LpLoss()
    pde_loss_fn = DiffusionSorptionPDE_Loss(method=args.pde_loss_method)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.use_lr_scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    # train
    test_errors = []
    for e in range(args.epochs):
        model.train()
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            # forward pass
            y_pred = model(x).squeeze(1)

            # loss
            data_loss = pde_loss = ic_loss = bc_loss = 0

            if args.data_loss_weight > 0:
                data_loss = relative_L2_loss_fn(y_pred, y)
            
            if args.pde_loss_weight > 0:
                pde_loss = pde_loss_fn(y_pred)
            
            if args.ic_loss_weight > 0:
                ic_gt = y[:, 0, :]
                ic_pred = y_pred[:, 0, :]
                ic_loss = relative_L2_loss_fn(ic_pred, ic_gt)
            
            if args.bc_loss_weight > 0:
                bc_pred = get_bc_1d(y_pred)
                bc_gt = get_bc_1d(y)
                bc_loss = relative_L2_loss_fn(bc_pred, bc_gt)

            total_loss = args.data_loss_weight * data_loss + args.pde_loss_weight * pde_loss + args.ic_loss_weight * ic_loss + args.bc_loss_weight * bc_loss

            # backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # log
            mini_batch_idx = (e * len(train_loader) + i)
            to_log = {'Loss/training loss': total_loss.item(), 'Loss/Mini batch index': mini_batch_idx, 'Loss/Epoch': e}
            wandb.log(to_log)

        if args.use_lr_scheduler:
            scheduler.step()

        # eval
        if e % args.eval_every == 0:
            test_error = eval_model(model, test_loader, device)
            # test_error = 1.0    # TODO
            test_errors.append(test_error)

            print(f'Epoch {e}, Relative L2 test error: {test_error}')

            # log
            to_log = {'Relative L2 test error/Relative L2 test error': test_error, 'Relative L2 test error/Epoch': e}
            wandb.log(to_log)

    # relative L2 error on test sef after training is done
    final_test_error = eval_model(model, test_loader, device)

    print()
    print('-------Training complete-------')
    print('Final relative L2 error on test set: ', final_test_error)
    print('Best relative L2 error on test set: ', min(test_errors))
    print('Epoch with lowest relative L2 error on test set: ', test_errors.index(min(test_errors)) * args.eval_every)
    print()

    # save
    if args.pde_loss_weight > 0:
        name = 'physics_informed_fno_baseline'
    else:
        name = 'fno_baseline'
    path_save = f'saved/diffusion_sorption/{name}/{date_time_string}/'

    if not os.path.exists(path_save):
        os.makedirs(path_save)
    
    # TODO: visualize solution

    if args.save_model:
        torch.save(model.state_dict(), path_save + 'model.pt')

    # save arguments
    with open(f'{path_save}/args.txt', 'w') as file:
        for key, value in vars(args).items():
            file.write(f'{key}: {value}\n')
        
        # save error metrics
        file.write(f'Best relative L2 error: {min(test_errors)}\n')

    # close logger
    wandb.finish()

if __name__ == '__main__':
    import time
    start = time.time()
    main()
    print('Time taken:', time.time() - start)
    print('Done!')
