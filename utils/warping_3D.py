import torch
from utils.simulation_utils_3D import advect_quantity_batched_BFECC

class FieldWarper:
    backward_warp_tensor_grid = {}
    BFECC_warp_tensor_grid = {}

    # tensorInput B, C, H, W, D
    # tensorFlow B, 3, H, W, D
    @classmethod
    def BFECC_warp(cls, tensorInput, tensorFlow, device="cuda"):
        tensorFlow = torch.flip(tensorFlow, dims=[1])
        tensorFlow = tensorFlow.permute(0,2,3,4,1)
        if str(tensorFlow.size()) not in cls.BFECC_warp_tensor_grid:
            # x, y, z, 3
            cls.BFECC_warp_tensor_grid[str(tensorFlow.size())] = gen_grid(tensorInput.size(2), tensorInput.size(3), tensorInput.size(4), device)
        # print(str(tensorFlow.size()),tensorInput.size(), cls.BFECC_warp_tensor_grid[str(tensorFlow.size())].size())
        return advect_quantity_batched_BFECC(tensorInput.permute(0,2,3,4,1), tensorFlow,
                                             cls.BFECC_warp_tensor_grid[str(tensorFlow.size())], 1.0/tensorInput.size(3), None).permute(0,4,1,2,3)

def gen_grid(width, height, depth, device):
    img_n_grid_x = width
    img_n_grid_y = height
    img_n_grid_z = depth
    img_dx = 1./img_n_grid_y
    c_x, c_y, c_z = torch.meshgrid(torch.arange(img_n_grid_x), torch.arange(img_n_grid_y), torch.arange(img_n_grid_z),  indexing = "ij")
    img_x = img_dx * (torch.cat((c_x[..., None], c_y[..., None], c_z[..., None]), axis = -1) + 0.5).to(device) # grid center locations
    return img_x