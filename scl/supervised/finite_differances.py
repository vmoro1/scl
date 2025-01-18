# Computes finite differances to approximate derivatives.
# This is used to compute the pde loss.

import torch

# Set fix{x,y,z}_bnd=True if function is non-periodic in {x,y,z} direction
# central_diff_nd functions are used to compute the first derivatives
# central_diff_2nd_nd functions are used to compute the second derivatives

# x is the solution we weish to  compute derivatives for, h is the grid spacing

def central_diff_1d(x, h, fix_x_bnd=True):
    dx = (torch.roll(x, -1, dims=-1) - torch.roll(x, 1, dims=-1))/(2.0*h)

    if fix_x_bnd:
        dx[...,0] = (x[...,1] - x[...,0])/h
        dx[...,-1] = (x[...,-1] - x[...,-2])/h
    
    return dx


def central_diff_2d(x, h, fix_x_bnd=True, fix_y_bnd=True):
    if isinstance(h, float):
        h = [h, h]

    dx = (torch.roll(x, -1, dims=-2) - torch.roll(x, 1, dims=-2))/(2.0*h[0])
    dy = (torch.roll(x, -1, dims=-1) - torch.roll(x, 1, dims=-1))/(2.0*h[1])

    if fix_x_bnd:
        dx[...,0,:] = (x[...,1,:] - x[...,0,:])/h[0]
        dx[...,-1,:] = (x[...,-1,:] - x[...,-2,:])/h[0]
    
    if fix_y_bnd:
        dy[...,:,0] = (x[...,:,1] - x[...,:,0])/h[1]
        dy[...,:,-1] = (x[...,:,-1] - x[...,:,-2])/h[1]
        
    return dx, dy


def central_diff_3d(x, h, fix_x_bnd=True, fix_y_bnd=True, fix_z_bnd=True):
    if isinstance(h, float):
        h = [h, h, h]

    dx = (torch.roll(x, -1, dims=-3) - torch.roll(x, 1, dims=-3))/(2.0*h[0])
    dy = (torch.roll(x, -1, dims=-2) - torch.roll(x, 1, dims=-2))/(2.0*h[1])
    dz = (torch.roll(x, -1, dims=-1) - torch.roll(x, 1, dims=-1))/(2.0*h[2])

    if fix_x_bnd:
        dx[...,0,:,:] = (x[...,1,:,:] - x[...,0,:,:])/h[0]
        dx[...,-1,:,:] = (x[...,-1,:,:] - x[...,-2,:,:])/h[0]
    
    if fix_y_bnd:
        dy[...,:,0,:] = (x[...,:,1,:] - x[...,:,0,:])/h[1]
        dy[...,:,-1,:] = (x[...,:,-1,:] - x[...,:,-2,:])/h[1]
    
    if fix_z_bnd:
        dz[...,:,:,0] = (x[...,:,:,1] - x[...,:,:,0])/h[2]
        dz[...,:,:,-1] = (x[...,:,:,-1] - x[...,:,:,-2])/h[2]
        
    return dx, dy, dz


def central_diff_2nd_1d(x, h, fix_x_bnd=True):
    dxx = (torch.roll(x, -1, dims=-1) - 2*x + torch.roll(x, 1, dims=-1)) / (h**2)

    if fix_x_bnd:
        dxx[...,0] = (x[...,2] - 2*x[...,1] + x[...,0]) / (h**2)
        dxx[...,-1] = (x[...,-1] - 2*x[...,-2] + x[...,-3]) / (h**2)
    
    return dxx


def central_diff_2nd_2d(x, h, fix_x_bnd=True, fix_y_bnd=True):
    if isinstance(h, float):
        h = [h, h]

    dxx = (torch.roll(x, -1, dims=-2) - 2*x + torch.roll(x, 1, dims=-2)) / (h[0]**2)
    dyy = (torch.roll(x, -1, dims=-1) - 2*x + torch.roll(x, 1, dims=-1)) / (h[1]**2)

    if fix_x_bnd:
        dxx[...,0,:] = (x[...,2,:] - 2*x[...,1,:] + x[...,0,:]) / (h[0]**2)
        dxx[...,-1,:] = (x[...,-1,:] - 2*x[...,-2,:] + x[...,-3,:]) / (h[0]**2)
    
    if fix_y_bnd:
        dyy[...,:,0] = (x[...,:,2] - 2*x[...,:,1] + x[...,:,0]) / (h[1]**2)
        dyy[...,:,-1] = (x[...,:,-1] - 2*x[...,:,-2] + x[...,:,-3]) / (h[1]**2)
        
    return dxx, dyy


def central_diff_2nd_3d(x, h, fix_x_bnd=True, fix_y_bnd=True, fix_z_bnd=True):
    if isinstance(h, float):
        h = [h, h, h]

    dxx = (torch.roll(x, -1, dims=-3) - 2*x + torch.roll(x, 1, dims=-3)) / (h[0]**2)
    dyy = (torch.roll(x, -1, dims=-2) - 2*x + torch.roll(x, 1, dims=-2)) / (h[1]**2)
    dzz = (torch.roll(x, -1, dims=-1) - 2*x + torch.roll(x, 1, dims=-1)) / (h[2]**2)

    if fix_x_bnd:
        dxx[...,0,:,:] = (x[...,2,:,:] - 2*x[...,1,:,:] + x[...,0,:,:]) / (h[0]**2)
        dxx[...,-1,:,:] = (x[...,-1,:,:] - 2*x[...,-2,:,:] + x[...,-3,:,:]) / (h[0]**2)
    
    if fix_y_bnd:
        dyy[...,:,0,:] = (x[...,:,2,:] - 2*x[...,:,1,:] + x[...,:,0,:]) / (h[1]**2)
        dyy[...,:,-1,:] = (x[...,:,-1,:] - 2*x[...,:,-2,:] + x[...,:,-3,:]) / (h[1]**2)
    
    if fix_z_bnd:
        dzz[...,:,:,0] = (x[...,:,:,2] - 2*x[...,:,:,1] + x[...,:,:,0]) / (h[2]**2)
        dzz[...,:,:,-1] = (x[...,:,:,-1] - 2*x[...,:,:,-2] + x[...,:,:,-3]) / (h[2]**2)
        
    return dxx, dyy, dzz
