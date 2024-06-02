import torch
from utils.simulation_utils import advect_quantity_batched_BFECC

class FieldWarper:
    backward_warp_tensor_grid = {}
    BFECC_warp_tensor_grid = {}

    @classmethod
    def backward_warp(cls, tensorInput, tensorFlow, device="cuda"):
        if str(tensorFlow.size()) not in cls.backward_warp_tensor_grid:
            tensorHorizontal = torch.linspace(-1.0, 1.0, tensorFlow.size(3)).view(
                1, 1, 1, tensorFlow.size(3)).expand(tensorFlow.size(0), -1,
                                                    tensorFlow.size(2), -1)
            tensorVertical = torch.linspace(-1.0, 1.0, tensorFlow.size(2)).view(
                1, 1, tensorFlow.size(2), 1).expand(tensorFlow.size(0), -1, -1,
                                                    tensorFlow.size(3))

            cls.backward_warp_tensor_grid[str(tensorFlow.size())] = torch.cat(
                [tensorHorizontal, tensorVertical], 1).to(device)
        # end
        tensorFlow = torch.cat([
            tensorFlow[:, 0:1, :, :] / ((tensorInput.size(3) - 1.0) / 2.0),
            tensorFlow[:, 1:2, :, :] / ((tensorInput.size(2) - 1.0) / 2.0)
        ], 1)
        return torch.nn.functional.grid_sample(
            input=tensorInput,
            grid=(cls.backward_warp_tensor_grid[str(tensorFlow.size())] +
                tensorFlow).permute(0, 2, 3, 1),
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True)

    @classmethod
    def BFECC_warp(cls, tensorInput, tensorFlow, device="cuda"):
        # tensorFlow = torch.cat([
        #     tensorFlow[:, 0:1, :, :] / ((tensorInput.size(3) - 1.0)),
        #     tensorFlow[:, 1:2, :, :] / ((tensorInput.size(2) - 1.0))
        # ], 1).permute(0,2,3,1)
        tensorFlow = torch.flip(tensorFlow, dims=[1])
        tensorFlow = tensorFlow.permute(0,2,3,1)
        if str(tensorFlow.size()) not in cls.BFECC_warp_tensor_grid:
            # tensorHorizontal = torch.linspace(-1.0, 1.0, tensorFlow.size(2)).view(
            #     1, 1, 1, tensorFlow.size(2)).expand(tensorFlow.size(0), -1,
            #                                         tensorFlow.size(1), -1)
            # tensorVertical = torch.linspace(-1.0, 1.0, tensorFlow.size(1)).view(
            #     1, 1, tensorFlow.size(1), 1).expand(tensorFlow.size(0), -1, -1,
            #                                         tensorFlow.size(2))
            cls.BFECC_warp_tensor_grid[str(tensorFlow.size())] = gen_grid(tensorInput.size(2), tensorInput.size(3), device)
        # print(str(tensorFlow.size()),tensorInput.size(), cls.BFECC_warp_tensor_grid[str(tensorFlow.size())].size())
        return advect_quantity_batched_BFECC(tensorInput.permute(0,2,3,1), tensorFlow,
                                             cls.BFECC_warp_tensor_grid[str(tensorFlow.size())], 1.0/tensorInput.size(3), None).permute(0,3,1,2)

def gen_grid(width, height, device):
    img_n_grid_x = width
    img_n_grid_y = height
    img_dx = 1./img_n_grid_y
    c_x, c_y = torch.meshgrid(torch.arange(img_n_grid_x), torch.arange(img_n_grid_y), indexing = "ij")
    img_x = img_dx * (torch.cat((c_x[..., None], c_y[..., None]), axis = 2) + 0.5).to(device) # grid center locations
    return img_x
