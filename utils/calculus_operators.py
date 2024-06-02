import numpy as np
import torch
import torch.nn.functional as F

def construct_velocity_from_pos(pos):
    dist = torch.norm(pos, dim=0, p=2, keepdim=True)
    R = pos.flip([0])
    R[0] *= -1
    R = F.normalize(R, dim=0)
    return R/dist

class CalculusOperators:
    device = "cuda"

    def construct_laplace_kernel(device="cuda"):
        out_channels = 2  # u and v
        in_channels = 1  # u or v
        kh, kw = 3, 3

        filter_x = torch.FloatTensor([[0, 0, 0], [1, -2, 1], [0, 0, 0]])
        filter_y = torch.FloatTensor([[0, 1, 0], [0, -2, 0], [0, 1, 0]])

        weight = torch.ones(out_channels, in_channels, kh, kw, requires_grad=False)
        weight[0, 0, :, :] = filter_x
        weight[1, 0, :, :] = filter_y
        return weight.to(device)
    
    def construct_gradient_kernel(device="cuda"):
        out_channels = 2  # u and v
        in_channels = 1  # u or v
        kh, kw = 3, 3

        filter_x = torch.FloatTensor([[0, 0, 0], [0, -1, 1], [0, 0, 0]])
        filter_y = torch.FloatTensor([[0, 0, 0], [0, -1, 0], [0, 1, 0]])

        weight = torch.ones(out_channels, in_channels, kh, kw, requires_grad=False)
        weight[0, 0, :, :] = filter_x
        weight[1, 0, :, :] = filter_y
        return weight.to(device)

    def construct_3D_gradient_kernel(device="cuda"):
        out_channels = 3  # u and v
        in_channels = 1  # u or v
        kh, kw, kt = 3, 3, 3

        filter_x = torch.FloatTensor([[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                                      [[0, 0, 0], [0, -1, 1], [0, 0, 0]],
                                      [[0, 0, 0], [0, 0, 0], [0, 0, 0]]])
        filter_y = torch.FloatTensor([[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                                      [[0, 0, 0], [0, -1, 0], [0, 1, 0]],
                                      [[0, 0, 0], [0, 0, 0], [0, 0, 0]]])
        filter_z = torch.FloatTensor([[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                                      [[0, 0, 0], [0, -1, 0], [0, 0, 0]],
                                      [[0, 0, 0], [0, 1, 0], [0, 0, 0]]])


        weight = torch.ones(out_channels, in_channels, kh, kw, kt, requires_grad=False)
        weight[0, 0, :, :, :] = filter_x
        weight[1, 0, :, :, :] = filter_y
        weight[2, 0, :, :, :] = filter_z
        return weight.to(device)

    def construct_vorticity_kernel(device="cuda"):
        out_channels = 2  # u and v
        in_channels = 1  # u or v
        kh, kw = 6, 6

        weight = torch.zeros(in_channels, out_channels, kh, kw, requires_grad=False)
        u_pos = torch.stack(torch.meshgrid(torch.arange(-kh/2+1, kh/2, 1.0), torch.arange(-kh/2+0.5, kh/2+0.5, 1), indexing = "xy"), dim=0)
        v_pos = torch.stack(torch.meshgrid(torch.arange(-kh/2+0.5, kh/2+0.5, 1), torch.arange(-kh/2+1, kh/2, 1.0), indexing = "xy"), dim=0)

        weight[0,0,:,:-1] = construct_velocity_from_pos(u_pos)[0]
        weight[0,1,:-1,:] = construct_velocity_from_pos(v_pos)[1]

        return weight.to(device)

    def construct_forward_difference_kernels(device="cuda"):
        filter_x = torch.ones(1, 1, 3, 3, requires_grad=False)
        filter_y = torch.ones(1, 1, 3, 3, requires_grad=False)
        filter_x[0, 0, :, :] = torch.FloatTensor([[0, 0, 0], [0, -1, 1], [0, 0, 0]])
        filter_y[0, 0, :, :] = torch.FloatTensor([[0, 0, 0], [0, -1, 0], [0, 1, 0]])
        return filter_x.to(device), filter_y.to(device)
    
    laplace_kernel = construct_laplace_kernel(device)
    gradient_kernel = construct_gradient_kernel(device)
    gradient_kernel_3d = construct_3D_gradient_kernel(device)
    divergence_kernel_x, divergence_kernel_y = construct_forward_difference_kernels(device)
    vorticity_kernel = construct_vorticity_kernel(device)
    
    @classmethod
    def laplace(cls, flow):
        u, v = torch.split(flow, split_size_or_sections=1, dim=1)
        laplace_u = F.conv2d(u, cls.laplace_kernel, padding=1)
        laplace_v = F.conv2d(v, cls.laplace_kernel, padding=1)
        return laplace_u, laplace_v
    
    @classmethod
    def gradient(cls, flow):
        u, v = torch.split(flow, split_size_or_sections=1, dim=1)
        grad_u = F.conv2d(u, cls.gradient_kernel, padding=1)
        grad_v = F.conv2d(v, cls.gradient_kernel, padding=1)
        return grad_u, grad_v
    
    @classmethod
    def divergence(cls, flow):
        u, v = torch.split(flow, split_size_or_sections=1, dim=1)
        grad_ux = F.conv2d(u, cls.divergence_kernel_x, padding=1)
        grad_vy = F.conv2d(v, cls.divergence_kernel_y, padding=1)
        return grad_ux + grad_vy

    @classmethod
    def curl(cls, flow):
        u, v = torch.split(flow, split_size_or_sections=1, dim=1)
        grad_uy = F.conv2d(u, cls.divergence_kernel_y, padding=1)
        grad_vx = F.conv2d(v, cls.divergence_kernel_x, padding=1)
        return grad_vx - grad_uy