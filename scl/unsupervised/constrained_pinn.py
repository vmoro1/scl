import numpy as np
import torch

from .csl_problem import ConstrainedStatisticalLearningProblem


class ConstrainedPINN(ConstrainedStatisticalLearningProblem):
    """Physics-informed neural network for convection/diffusion/reaction PDEs. 
    The problem of approximating a soluting to a PDE is formulated as a constrained 
    statistical learning problem."""
    def __init__(
            self, 
            system, 
            model, 
            eps,
            X_train_initial, 
            u_train_initial, 
            X_train_pde, 
            bc_lb, 
            bc_ub, 
            G, 
            nu, 
            beta, 
            rho, 
            device):

        # pde system (convection, diffusion, reaction, reaction-diffusion (rd))
        self.system = system

        # data
        self.x_initial = torch.tensor(X_train_initial[:, 0:1], requires_grad=True).float().to(device)
        self.t_initial = torch.tensor(X_train_initial[:, 1:2], requires_grad=True).float().to(device)
        self.x_pde = torch.tensor(X_train_pde[:, 0:1], requires_grad=True).float().to(device)
        self.t_pde = torch.tensor(X_train_pde[:, 1:2], requires_grad=True).float().to(device)
        self.x_bc_lb = torch.tensor(bc_lb[:, 0:1], requires_grad=True).float().to(device)
        self.t_bc_lb = torch.tensor(bc_lb[:, 1:2], requires_grad=True).float().to(device)
        self.x_bc_ub = torch.tensor(bc_ub[:, 0:1], requires_grad=True).float().to(device)
        self.t_bc_ub = torch.tensor(bc_ub[:, 1:2], requires_grad=True).float().to(device)
        self.u_initial = torch.tensor(u_train_initial, requires_grad=True).float().to(device)

        # model
        self.model = model

        # PDE parameters
        self.nu = nu
        self.beta = beta
        self.rho = rho

        # forcing term (not used)
        self.G = torch.tensor(G, requires_grad=True).float().to(device)
        self.G = self.G.reshape(-1, 1)

        self.device = device

        # define objective, constraints, and the right hand side of the constraints (i.e. the c in f(x) < c)
        self.objective_function = self.data_loss
        self.constraints = [self.pde_loss]
        self.rhs = [eps]        # epsillon is the tolerance for the average residual of the PDE loss, i.e. how well the model satisfies the PDE

        super().__init__()

    def forward(self, x, t):
        """Make a forward pass throught the model. The model takes (x,t) as input and predicts the value of the funcion, u."""
        u = self.model(torch.cat([x, t], dim=1))
        return u

    def pde_residual(self, x, t):
        """ Autograd for calculating the residual for different PDE systems of the model. I.e. if the model satisfies the PDE"""
        u = self.forward(x, t)

        u_t = torch.autograd.grad(
            u, t,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        u_x = torch.autograd.grad(
            u, x,
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

        if 'convection' in self.system or 'diffusion' in self.system:
            residual = u_t - self.nu*u_xx + self.beta*u_x - self.G
        elif 'rd' in self.system:
            residual = u_t - self.nu*u_xx - self.rho*u + self.rho*u**2
        elif 'reaction' in self.system:
            residual = u_t - self.rho*u + self.rho*u**2
        
        return residual

    def boundary_derivatives(self, u_lb, u_ub, x_bc_lb, x_bc_ub):
        """Takes derivative of model at the boundary. Used to satisfy periodic boundary condition (BC)."""

        u_lb_x = torch.autograd.grad(
            u_lb, x_bc_lb,
            grad_outputs=torch.ones_like(u_lb),
            retain_graph=True,
            create_graph=True
            )[0]

        u_ub_x = torch.autograd.grad(
            u_ub, x_bc_ub,
            grad_outputs=torch.ones_like(u_ub),
            retain_graph=True,
            create_graph=True
            )[0]

        return u_lb_x, u_ub_x
    
    def data_loss(self):
        """Computes the data loss. The data loss is computed for points on the 
        boundary and for the initial condition."""
        # prediction for initial condition, lower boundary, and upper boundary
        u_pred_initial = self.forward(self.x_initial, self.t_initial)
        u_pred_lb = self.forward(self.x_bc_lb, self.t_bc_lb)
        u_pred_ub = self.forward(self.x_bc_ub, self.t_bc_ub)

        loss_initial = torch.mean((self.u_initial - u_pred_initial) ** 2)
        loss_boundary = torch.mean((u_pred_lb - u_pred_ub) ** 2)

        if self.nu != 0:
            u_pred_lb_x, u_pred_ub_x = self.boundary_derivatives(u_pred_lb, u_pred_ub, self.x_bc_lb, self.x_bc_ub)
            loss_boundary += torch.mean((u_pred_lb_x - u_pred_ub_x) ** 2)

        loss = loss_initial + loss_boundary
        return loss
    
    def pde_loss(self):
        """Computes the PDE loss, i.e. how well the model satisfies the underlying PDE."""
        residual_pde = self.pde_residual(self.x_pde, self.t_pde)
        loss_pde = torch.mean(residual_pde ** 2)
        return loss_pde
    
    def predict(self, X):
        """Make predictions. Used during evaluation."""
        x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(self.device)
        t = torch.tensor(X[:, 1:2], requires_grad=True).float().to(self.device)

        self.model.eval()
        u = self.forward(x, t)
        u = u.detach().cpu().numpy()

        return u
