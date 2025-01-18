import numpy as np
import torch
from torch.nn import functional as F
from torch.autograd import grad

from .finite_differances import *


class BurgersPDE_Loss:
    """Physics-informed loss for Burgers equation: u_t = -u^2_x/2 + v*u_xx  = 0"""

    def __init__(self, visc, method='finite_difference', loss=F.mse_loss, domain_length=1.0, reduce=True):
        super().__init__()

        self.visc = visc
        self.method = method
        self.loss = loss
        self.reduce = reduce
        self.domain_length = domain_length
        if not isinstance(self.domain_length, (tuple, list)):
            self.domain_length = [self.domain_length] * 2

    def autograd(self, x, u):
        """x is input to model and u is the output of the model."""

        # compute derivatives and second derivatives
        u_grad = grad(outputs=u.sum(), inputs=x, create_graph=True, retain_graph=True)[0]
        u_t = u_grad[:, 1, :, :]
        u_x = u_grad[:, 2, :, :]
        u_xx = grad(outputs=u_x.sum(), inputs=x, create_graph=True, retain_graph=True)[0][:, 1, :, :]

        u_t = u_t.view(u.shape)
        u_x = u_x.view(u.shape)
        u_xx = u_xx.view(u.shape)

        # note that ux * u == u_x^2 / 2
        right_hand_side = - u_x * u + self.visc * u_xx

        pde_loss = self.loss(u_t, right_hand_side)
        assert not u_t.isnan().any()
        assert not u_x.isnan().any()
        assert not u_xx.isnan().any()

        del u_grad, u_x, u_t, u_xx
        return pde_loss

    def finite_difference(self, u):
        _, nt, nx = u.shape

        # compute the terms in Burger's equation
        # note: here we assume that the input is a regular grid
        dt = self.domain_length[1] / (nt - 1)
        dx = self.domain_length[0] / nx

        # du/dt and du/dx
        dudt, dudx = central_diff_2d(u, [dt, dx], fix_x_bnd=True, fix_y_bnd=True)

        # d^2u/dxx
        dudxx = (torch.roll(u, -1, dims=-1) - 2*u + torch.roll(u, 1, dims=-1))/dx**2
        # fix boundary (use only backward/forward differance as opposed to central difference)
        dudxx[...,0] = (u[...,2] - 2*u[...,1] + u[...,0])/dx**2
        dudxx[...,-1] = (u[...,-1] - 2*u[...,-2] + u[...,-3])/dx**2

        assert u.shape == dudx.shape == dudxx.shape == dudt.shape

        # right hand side. note that ux * u == u_x^2 / 2
        right_hand_side = -dudx * u + self.visc * dudxx

        if self.reduce:
            pde_loss = self.loss(dudt, right_hand_side)
            return pde_loss
        else:
            # don't average loss across samples but only withing samples
            per_sample_pde_loss = self.loss(dudt, right_hand_side, reduction='none')
            per_sample_pde_loss = per_sample_pde_loss.mean(dim=[1,2])   # average over spatial and temporal dimension
            return per_sample_pde_loss

    def fdm_fourier_hybrid(self, u):
        """Fourier differentiation for spatial dim, finite differances for temporal dim"""
        _, nt, nx = u.shape

        # note: here we assume that the input is a regular grid
        dt = self.domain_length[0] / (nt - 1)

        # Wavenumbers in y-direction
        k_max = nx // 2
        k_x = torch.cat((torch.arange(start=0, end=k_max, step=1, device=u.device),
                        torch.arange(start=-k_max, end=0, step=1, device=u.device)), 0).reshape(1, 1, nx)

        # compute u_xx using Fourier differentiation
        u_h = torch.fft.fft(u, dim=2)       # this is F(u), the Fourier transform
        ux_h = 2j * np.pi * k_x * u_h       # this is F(u_x)
        uxx_h = 2j * np.pi * k_x * ux_h     # this is F(u_xx)
        u_x = torch.fft.irfft(ux_h[:, :, :k_max+1], dim=2, n=nx)
        u_xx = torch.fft.irfft(uxx_h[:, :, :k_max+1], dim=2, n=nx)

        # compute u_t using finite difference
        u_t = (u[:, 2:, :] - u[:, :-2, :]) / (2 * dt)
        
        # note that ux * u == u_x^2 / 2
        left_hand_side = u_t + (u_x * u - self.visc * u_xx)[:, 1:-1, :]
        f = torch.zeros(left_hand_side.shape, device=u.device)
        pde_loss = self.loss(left_hand_side, f)
        return pde_loss   

    def __call__(self, u, x=None):
        if self.method == 'finite_difference':
            return self.finite_difference(u)
        elif self.method == 'fdm_fourier_hybrid':
            return self.fdm_fourier_hybrid(u)
        elif self.method == 'autograd':
            return self.autograd(x, u)
        else:
            raise NotImplementedError()
        

class DarcyFlowPDE_Loss:
    def __init__(self, method='finite_differance',  loss=F.mse_loss, domain_length=1.0, reduce=True):
        self.method = method
        self.loss = loss
        self.domain_length = domain_length
        self.reduce = reduce

    def get_gt_forcing_fn(self, a, y, dx, dy):
        """
        In the case where the forcing function is unknown, 
        we can calculate it from the ground truth solution function
        """
        gt_x = (y[:, 2:, 1:-1] - y[:, :-2, 1:-1]) / (2 * dx)
        gt_y = (y[:, 1:-1, 2:] - y[:, 1:-1, :-2]) / (2 * dy)

        a_gt_x = a * gt_x
        a_gt_y = a * gt_y

        a_gt_xx = (a_gt_x[:, 2:, 1:-1] - a_gt_x[:, :-2, 1:-1]) / (2 * dx)
        a_gt_yy = (a_gt_y[:, 1:-1, 2:] - a_gt_y[:, 1:-1, :-2]) / (2 * dy)
        gt_forcing_fn =  -(a_gt_xx + a_gt_yy)
        return gt_forcing_fn
    
    def finite_difference(self, x, u, u_gt):
        """u_gt is the ground truth solution, u is the output of the model (predicted solution) and a is the input to the model."""

        a = x[:, 0, :, :]       # a(x, y) in the Darcy Flow equation (i.e. the difussion coefficient)

        # compute the left hand side of the Darcy Flow equation
        # note: here we assume that the input is a regular grid
        _, nx, ny = u.shape
        dx = self.domain_length / (nx - 1)
        dy = dx

        # first order derivatives
        ux, uy = central_diff_2d(u, [dx, dy], fix_x_bnd=True, fix_y_bnd=True)

        a_ux = a * ux
        a_uy = a * uy

        # second order derivatives
        a_uxx, _ = central_diff_2d(a_ux, [dx, dy], fix_x_bnd=True, fix_y_bnd=True)
        _, a_uyy = central_diff_2d(a_uy, [dx, dy], fix_x_bnd=True, fix_y_bnd=True)

        left_hand_side =  -(a_uxx + a_uyy)

        forcing_fn = torch.ones(left_hand_side.shape, device=u.device)                      # forcing function in the Darcy Flow equation (i.e. f(x, y))
        # empirical_forcing_fn = self.get_gt_forcing_fn(a[:,1:-1, 1:-1], u_gt, dx, dy)      # compute the forcing function if it is unknown

        # compute the loss of the left and right hand sides of the Darcy Flow equation
        if self.reduce:
            pde_loss = self.loss(left_hand_side, forcing_fn)
            return pde_loss
        else:
            # don't average loss across samples but only withing samples
            per_sample_pde_loss = self.loss(left_hand_side, forcing_fn, reduction='none')
            per_sample_pde_loss = per_sample_pde_loss.mean(dim=[1,2])   # average over spatial and temporal dimension
            return per_sample_pde_loss
    
    def autograd(self, x, u):
        """x is input to model and u is the output of the model."""

        a = x[:, 0, :, :]   # a(x, y) in the Darcy Flow equation (i.e. the difussion coefficient)

        # compute derivatives
        u_grad = grad(outputs=u.sum(), inputs=x, create_graph=True, retain_graph=True)[0]
        u_x = u_grad[:, 1, :, :]
        u_y = u_grad[:, 2, :, :]

        au_x = a * u_x
        au_y = a * u_y

        # compute second derivatives
        au_xx = grad(outputs=au_x.sum(), inputs=x, create_graph=True, retain_graph=True)[0][:, 1, :, :]
        au_yy = grad(outputs=au_y.sum(), inputs=x, create_graph=True, retain_graph=True)[0][:, 2, :, :]

        left_hand_side =  -(au_xx + au_yy)
        forcing_fn = torch.ones(left_hand_side.shape, device=u.device)
        loss = self.loss(left_hand_side, forcing_fn)
        
        del u_grad, u_x, u_y, au_x, au_y, au_xx, au_yy
        return loss
        
    def __call__(self, x, u, u_gt=None):
        if self.method == 'autograd':
            return self.autograd(x, u)
        elif self.method == 'finite_difference':
            return self.finite_difference(x, u, u_gt)
        else:
            raise NotImplementedError()
        

class NavierStokesPDE_Loss:
    """Physics-informed loss for Navier-Stokes equation in worticity form: w_t + (u_x * w_x + u_y * w_y) = viscosity * (w_xx + w_yy) + forcing.
    forcing function: f = 0.1 * (sin(2pi * (x + y)) + cos(2pi * (x + y)))""" 

    def __init__(self, viscosity, grid_x, grid_y, gridt_1d, method='fdm_fourier_hybrid', loss=F.mse_loss, reduce=True):
        self.viscosity = viscosity
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.dt = gridt_1d[1] - gridt_1d[0]
        self.method = method
        self.loss = loss
        self.reduce = reduce

    def fdm_fourier_hybrid(self, u):   
        assert u.shape[1:] == (50, 64, 64)

        w = u       # shape: (batchsize, nt, nx, ny)
        _, nt, nx, ny = w.shape

        dt = self.dt        # dt = 1.0
        dx = 1 / nx
        dy = 1 / ny


        # wavenumbers in x and y-direction
        kx = torch.fft.fftfreq(nx, dtype=torch.float32, device=u.device) * nx * 2 * np.pi
        ky = torch.fft.fftfreq(ny, dtype=torch.float32, device=u.device) * ny * 2 * np.pi
        kx = kx[:, None].repeat(1, ny)
        ky = ky[None, :].repeat(nx, 1)

        # negative Laplacian
        Δ = kx ** 2 + ky ** 2
        Δ[0, 0] = 1  

        wt, wx, wy = central_diff_3d(w, [dt, dx, dy], fix_x_bnd=True, fix_y_bnd=True, fix_z_bnd=True)
        _, wxx, wyy = central_diff_3d(w, [dt, dx, dy], fix_x_bnd=True, fix_y_bnd=True, fix_z_bnd=True)

        # laplacian of w
        w_lap = wxx + wyy

        # vecocities (u in the fomulation of the Navier-Stokes equation)
        what = torch.fft.fft2(w, dim=(-2, -1))

        ux = torch.fft.irfft2(what * 1j * ky / Δ, s=what.shape[-2:], dim=(-2, -1))
        uy = torch.fft.irfft2(what * -1j * kx / Δ, s=what.shape[-2:], dim=(-2, -1))

        forcing = self.get_forcing()
        forcing = forcing.to(u.device)

        lhs = wt + ux * wx + uy * wy
        rhs = self.viscosity * w_lap + forcing

        if self.reduce:
            pde_loss = self.loss(lhs, rhs)
            return pde_loss
        else:
            # don't average loss across samples but only withing samples
            per_sample_pde_loss = self.loss(lhs, rhs, reduction='none')
            per_sample_pde_loss = per_sample_pde_loss.mean(dim=[1,2, 3])   # average over spatial and temporal dimension
            return per_sample_pde_loss
    
    def get_forcing(self):
        nx, ny = len(self.grid_x), len(self.grid_y)
        grid_x = self.grid_x.reshape(1,1,nx,1).repeat(1,1,1,ny)
        grid_y = self.grid_y.reshape(1,1,1,ny).repeat(1,1,nx,1)
        f = 0.1 * (torch.sin(2 * torch.pi * (grid_x + grid_y)) + torch.cos( 2 * torch.pi * (grid_x + grid_y)))
        return f
    
    def __call__(self, u):
        if self.method == 'fdm_fourier_hybrid':
            return self.fdm_fourier_hybrid(u)
        else:
            raise NotImplementedError()
        

# class NavierStokesPDE_Loss_PINO:
#     """Physics-informed loss for Navier-Stokes equation in worticity form: w_t + (u_x * w_x + u_y * w_y) = viscosity * (w_xx + w_yy) + forcing.
#     forcing function: f = 0.1 * (sin(2pi * (x + y)) + cos(2pi * (x + y)))""" 

#     def __init__(self, viscosity, grid_x, grid_y, method='fdm_fourier_hybrid', loss=F.mse_loss):
#         self.viscosity = viscosity
#         self.grid_x = grid_x
#         self.grid_y = grid_y
#         self.method = method
#         self.loss = loss

#     def fdm_fourier_hybrid(self, u):

#         w = u.permute(0,2,3,1)         # new shape: (batchsize, nx, ny, nt)
#         assert u.shape[1:] == (64, 64, 50)
        
#         _, nx, ny, nt = w.shape
#         device = w.device

#         dt = 1.0 / (nt - 1)

#         w_h = torch.fft.fft2(w, dim=[1, 2])

#         # wavenumbers in x and y-direction
#         k_max = nx//2
#         N = nx
#         k_x = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device),
#                         torch.arange(start=-k_max, end=0, step=1, device=device)), 0).reshape(N, 1).repeat(1, N).reshape(1,N,N,1)
#         k_y = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device),
#                         torch.arange(start=-k_max, end=0, step=1, device=device)), 0).reshape(1, N).repeat(N, 1).reshape(1,N,N,1)
        
#         # negative Laplacian in Fourier space
#         lap = (k_x ** 2 + k_y ** 2)
#         lap[0, 0, 0, 0] = 1.0
#         f_h = w_h / lap

#         ux_h = 1j * k_y * f_h
#         uy_h = -1j * k_x * f_h
#         wx_h = 1j * k_x * w_h
#         wy_h = 1j * k_y * w_h
#         wlap_h = -lap * w_h

#         ux = torch.fft.irfft2(ux_h[:, :, :k_max + 1], dim=[1, 2])
#         uy = torch.fft.irfft2(uy_h[:, :, :k_max + 1], dim=[1, 2])
#         wx = torch.fft.irfft2(wx_h[:, :, :k_max+1], dim=[1,2])
#         wy = torch.fft.irfft2(wy_h[:, :, :k_max+1], dim=[1,2])
#         wlap = torch.fft.irfft2(wlap_h[:, :, :k_max+1], dim=[1,2])

#         _, _, wt = central_diff_3d(w, [1, 1, dt], fix_x_bnd=True, fix_y_bnd=True, fix_z_bnd=True)

#         forcing = self.get_forcing()

#         lhs = wt + (ux*wx + uy*wy)
#         rhs = self.viscosity * wlap + forcing

#         pde_loss = self.loss(lhs, rhs)
#         return pde_loss
    
#     def get_forcing(self):
#         nx, ny = len(self.grid_x), len(self.grid_y)
#         grid_x = self.grid_x.reshape(1,nx,1,1).repeat(1,1,ny,1)
#         grid_y = self.grid_y.reshape(1,1,ny,1).repeat(1,nx,1,1)
#         f = 0.1 * (torch.sin(2 * torch.pi * (grid_x + grid_y)) + torch.cos( 2 * torch.pi * (grid_x + grid_y)))
#         return f
    
#     def __call__(self, u):
#         if self.method == 'fdm_fourier_hybrid':
#             return self.fdm_fourier_hybrid(u)
#         else:
#             raise NotImplementedError()


class DiffusionSorptionPDE_Loss:
    """Physics-informed loss for Diffusion-Sorption: u_t = (D / R(u)) * u_xx"""

    def __init__(self, method='finite_difference', loss=F.mse_loss, reduce=True, num_collocation_points=20000):
        super().__init__()
        self.method = method
        self.loss = loss
        self.reduce = reduce
        self.num_collocation_points = num_collocation_points

        self.D = 5e-4
        self.por = 0.29
        self.rho_s = 2880
        self.k_f = 3.5e-4
        self.n_f = 0.874
        
        # self.dx = 0.001
        self.dx = 1 / 1024
        self.dt = 5

    def finite_difference(self, u):
        ut, _ = central_diff_2d(u, [self.dt, self.dx], fix_x_bnd=True, fix_y_bnd=True)
        _, uxx = central_diff_2nd_2d(u, [self.dt, self.dx], fix_x_bnd=True, fix_y_bnd=True)

        safe_u = torch.where(u > 0, u, 0)
        retardation_factor = 1 + ((1 - self.por) / self.por) * \
            self.rho_s * self.k_f * self.n_f * \
            torch.where(u > 0, torch.pow(safe_u + 1e-6, self.n_f - 1), 1e-6**(self.n_f-1))
        
        pde_residual = ut - (self.D / retardation_factor) * uxx

        # # sample #num_collocation_points points for each sample that are used to update the loss
        # n_samples = pde_residual.shape[0]
        # pde_residual = pde_residual.view(n_samples, -1)
        # indices = torch.randint(0, pde_residual.shape[1], (n_samples, self.num_collocation_points))
        # pde_residual = torch.gather(pde_residual, 1, indices)

        if self.reduce:
            pde_loss = self.loss(pde_residual, torch.zeros_like(pde_residual))
            return pde_loss
        else:
            # don't average loss across samples but only withing samples
            per_sample_pde_loss = self.loss(pde_residual, torch.zeros_like(pde_residual), reduction='none')
            per_sample_pde_loss = per_sample_pde_loss.mean(dim=[1,2])   # average over spatial and temporal dimension
            return per_sample_pde_loss

    def __call__(self, u):
        if self.method == 'finite_difference':
            return self.finite_difference(u)
        else:
            raise NotImplementedError()
        