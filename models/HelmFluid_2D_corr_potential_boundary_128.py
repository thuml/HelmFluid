"""
@author: Lanxiang Xing
"""
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import math
from utils.calculus_operators import CalculusOperators
from utils.warping import FieldWarper
from utils.correlation import FunctionCorrelation


################################################################
# Multiscale modules 2D
################################################################
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, dropout=False, up=False):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        if dropout:
            self.double_conv = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) if up else nn.Identity(),
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                # nn.Dropout(0.1),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) if up else nn.Identity(),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.double_conv = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) if up else nn.Identity(),
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) if up else nn.Identity(),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, dropout=False):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, dropout=dropout)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=groups)

    def forward(self, x):
        return self.conv(x)

################################################################
# Calculate velocity from phi and vorticity
################################################################

def phi_to_Cgrid_velocity(phi):
    vel = F.conv2d(phi, CalculusOperators.gradient_kernel,padding=1)
    u = vel[:, 0:1, :, :-1]
    v = vel[:, 1:2, :-1, :]
    return u, v

def stream_to_Cgrid_velocity(phi):
    vel = F.conv2d(phi, CalculusOperators.gradient_kernel,padding=1)
    v = vel[:, 0:1, :, :-1]
    u = -vel[:, 1:2, :-1, :]
    return u, v

def vort_to_Cgrid_velocity(vorticity):
    vel = F.conv_transpose2d(vorticity, CalculusOperators.vorticity_kernel)
    u = vel[:, 0:1, 2:-2, 2:-3]
    v = vel[:, 1:2, 2:-3, 2:-2]
    return u, v

def Cgrid_to_motion(u, v, shape):
    motion_u = (u[:,:,1:-1,1:]+u[:,:,1:-1,:-1])/2
    motion_v = (v[:,:,1:,1:-1]+v[:,:,:-1,1:-1])/2
    motion = torch.cat([motion_u, motion_v], dim=1)
    height, width = shape
    up = (motion.shape[2] - height) // 2
    down = (height + motion.shape[2]) // 2
    left = (motion.shape[3] - width) // 2
    right = (width + motion.shape[3]) // 2
    return motion[..., up:down, left:right]

def calc_phi_velocity(phi, corr_size):
    u_phi, v_phi = phi_to_Cgrid_velocity(phi)
    vel_phi = Cgrid_to_motion(u_phi, v_phi, corr_size)
    return vel_phi

def calc_stream_velocity(stream, corr_size):
    u_stream, v_stream = stream_to_Cgrid_velocity(stream)
    vel_stream = Cgrid_to_motion(u_stream, v_stream, corr_size)
    return vel_stream

def calc_vort_velocity(vorticity, corr_size):
    u_vorticity, v_vorticity = vort_to_Cgrid_velocity(vorticity[:,:,:-1,:-1])
    vel_vorticity = Cgrid_to_motion(u_vorticity, v_vorticity, corr_size)
    return vel_vorticity

def aggregate_to_velocity(phi, stream, corr_size):
    vel_phi = calc_phi_velocity(phi, corr_size)
    vel_stream = calc_vort_velocity(stream, corr_size)

    return vel_phi+vel_stream, vel_phi, vel_stream

################################################################
# Helmholtz decomposition blocks
################################################################
def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_2d_sincos_pos_embed(embed_dim, height, width):
    """
            grid_size: int of the grid height and width
            return:
            pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
            """
    grid_h = np.arange(height, dtype=np.float32)
    grid_w = np.arange(width, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, height, width])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed

class FeedForward(nn.Module):
    def __init__(self, dim, out_dim, hidden_dim=None, dropout = 0.):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = out_dim
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class HelmBlock(nn.Module):
    def __init__(self, input_dim, out_dim, correlation_dim=None, encoder_dim=None, groups=8, patch_size=4):
        super(HelmBlock, self).__init__()
        self.groups = groups
        self.patch_size = patch_size
        if encoder_dim == None:
            encoder_dim = out_dim
        if correlation_dim == None:
            correlation_dim = out_dim
        self.encoder_dim = encoder_dim
        self.correlation_dim = correlation_dim

        neighbours = 81
        self.encoder = nn.Conv2d(input_dim // groups, correlation_dim, kernel_size=patch_size, stride=patch_size)
        self.vorticity_decoder = nn.Sequential(
            DoubleConv(neighbours*2, encoder_dim // groups, up=True),
            OutConv(encoder_dim // groups, 1)
        )
        self.phi_decoder = nn.Sequential(
            DoubleConv(neighbours*2, encoder_dim // groups, up=True),
            OutConv(encoder_dim // groups, 1)
        )

        self.vorticity_weight = nn.Parameter(torch.tensor([0.005]))
        self.phi_weight = nn.Parameter(torch.tensor([0.5]))
        self.warping = FieldWarper.BFECC_warp
        self.decoder = nn.Sequential(
            nn.LayerNorm(out_dim),
            FeedForward(out_dim, out_dim, encoder_dim)
        )
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.softmax = nn.Softmax(dim=-1)
        self.pos_embed = None

    def forward(self, prev, next, texture, mask, boundary, helm=None):
        shape = texture.shape[-2:]
        prev_mask = prev * mask[:, None]
        prev_boundary = prev * boundary[:, None]
        next_mask = next * mask[:, None]
        B, C, H, W = prev.shape

        prev_mask = prev_mask.reshape([B * self.groups, C // self.groups, H, W])
        prev_boundary = prev_boundary.reshape([B * self.groups, C // self.groups, H, W])
        next_mask = next_mask.reshape([B * self.groups, C // self.groups, H, W])

        prev_mask_features = self.encoder(prev_mask)
        prev_boundary_features = self.encoder(prev_boundary)
        next_mask_features = self.encoder(next_mask)
        correlation_mask = FunctionCorrelation(prev_mask_features, next_mask_features)
        correlation_boundary = FunctionCorrelation(prev_boundary_features, next_mask_features)
        correlation = torch.cat([correlation_mask, correlation_boundary], dim=1)

        phi = self.phi_decoder(correlation) * self.phi_weight
        vorticity = self.vorticity_decoder(correlation) * self.vorticity_weight * next.shape[-2]
        helmholtz = torch.cat([phi, vorticity], dim=1).reshape([B * self.groups, 2, H, W])

        if helm is not None:
            helmholtz = self.up(helm*2) + helmholtz
        vel, vel_phi, vel_vort = aggregate_to_velocity(helmholtz[:, 0:1], helmholtz[:, 1:2], shape)
        B, C, H, W = texture.shape
        texture = texture.reshape([B * self.groups, C // self.groups, H, W])
        pred_features = self.warping(texture, vel)
        pred_features = pred_features.reshape([B, C, H, W])
        # (5) decoder
        # de-patchify
        # x = pred_features
        x = pred_features + self.decoder(pred_features.permute(0,2,3,1)).permute(0,3,1,2)
        return x, helmholtz, vel, vel_phi, vel_vort

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, 1, size_x, 1).repeat([batchsize, 1, 1, size_y])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, 1, size_y).repeat([batchsize, 1, size_x, 1])
        return torch.cat((gridx, gridy), dim=1).to(device)


class Model(nn.Module):
    def __init__(self, args, bilinear=True):
        super(Model, self).__init__()
        self.groups = args.groups
        self.dataset_name = args.dataset_name
        print('Groups:', self.groups)
        in_channels = args.in_dim
        out_channels = args.out_dim
        width = args.d_model
        padding = [int(x) for x in args.padding.split(',')]
        # multiscale modules
        self.texture_inc = DoubleConv(width, width)
        self.texture_down1 = Down(width, width * 2)
        factor = 2 if bilinear else 1
        self.texture_down2 = Down(width * 2, width * 4)
        self.texture_down3 = Down(width * 4, width * 8 // factor)

        self.up1 = Up(width * 16, width * 8 // factor, bilinear)
        self.up2 = Up(width * 8, width * 4 // factor, bilinear)
        self.up3 = Up(width * 4, width * 2 // factor, bilinear)
        self.up4 = Up(width * 2, width, bilinear)
        self.outc = OutConv(width, width)

        # Multiscale Helmholtz Warping Block
        self.feature_inc = DoubleConv(width, width)
        self.feature_down1 = Down(width, width * 2)
        self.feature_down2 = Down(width * 2, width * 4)
        self.feature_down3 = Down(width * 4, width * 8)

        self.process1 = HelmBlock(width, width, 128, groups=self.groups) # original 1024
        self.process2 = HelmBlock(width * 2, width * 2, 256, groups=self.groups) # original 1024
        self.process3 = HelmBlock(width * 4, width * 4, 512, groups=self.groups) # original 1024
        self.process4 = HelmBlock(width * 8, width * 8 // factor, 512, groups=self.groups)
        # projectors
        self.padding = padding
        self.mask_up = nn.MaxPool2d(2)
        if 'boundary' in self.dataset_name:
            self.texture_fc0 = nn.Linear(in_channels + 1, width)
            self.feature_fc0 = nn.Linear(in_channels, width)
        elif self.dataset_name == 'real_video':
            self.texture_fc0 = nn.Linear(in_channels + 1, width)
            self.feature_fc0 = nn.Linear(in_channels - 2, width)
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x, mask, boundary):
        prev = x[..., :-1]
        next = x[..., 1:]

        if len(x.shape) == 5:
            B, H, W, C, L = x.shape
            x = x.flatten(-2)
            prev = prev.flatten(-2)
            next = next.flatten(-2)
        elif len(x.shape) == 4:
            B, H, W, L = x.shape
        input_mask = (mask * 1.0).to(x.device).unsqueeze(-1)

        if 'boundary' in self.dataset_name:
            prev = prev.permute(0, 3, 1, 2)
            prev = torch.nn.functional.pad(prev, [0,16,16,16], mode='constant')
            prev = torch.nn.functional.pad(prev, [16,0,0,0], mode='replicate')

            prev = prev.permute(0, 2, 3, 1).contiguous()
            next = next.permute(0, 3, 1, 2)
            next = torch.nn.functional.pad(next, [0,16,16,16], mode='constant')
            next = torch.nn.functional.pad(next, [16,0,0,0], mode='replicate')
            next = next.permute(0, 2, 3, 1).contiguous()
            feature_mask = torch.nn.functional.pad(input_mask, [0, 0, 0, 0, 16, 16], mode='constant')
            feature_mask = torch.nn.functional.pad(feature_mask, [0, 0, 16, 16, 0, 0], mode='replicate')
            mask = torch.nn.functional.pad(mask * 1.0, [16, 16, 16, 16], mode='constant') > 0
            boundary = torch.nn.functional.pad(boundary * 1.0, [16, 16, 16, 16], mode='constant') > 0
            mask = mask.to(x.device)
            boundary = boundary.to(x.device)
        elif self.dataset_name == 'real_video':
            prev = torch.nn.functional.pad(prev, [0,0,16,16,16,16], mode='replicate')
            next = torch.nn.functional.pad(next, [0,0,16,16,16,16], mode='replicate')
            feature_mask = torch.nn.functional.pad(input_mask, [0,0,16,16,16,16], mode='constant')
            mask = torch.nn.functional.pad(mask * 1.0, [16,16,16,16], mode='constant') > 0
            boundary = torch.nn.functional.pad(boundary * 1.0, [16,16,16,16], mode='constant') > 0
            mask = mask.to(x.device)
            boundary = boundary.to(x.device)

        prev = self.feature_fc0(torch.cat((prev, feature_mask), dim=-1))
        next = self.feature_fc0(torch.cat((next, feature_mask), dim=-1))
        prev = prev.permute(0, 3, 1, 2)
        next = next.permute(0, 3, 1, 2)

        x = torch.cat((x, input_mask), dim=-1)
        x = self.texture_fc0(x)
        x = x.permute(0, 3, 1, 2)


        x1 = self.texture_inc(x)
        x2 = self.texture_down1(x1)
        x3 = self.texture_down2(x2)
        x4 = self.texture_down3(x3)

        prev_x1 = self.feature_inc(prev)
        prev_x2 = self.feature_down1(prev_x1)
        prev_x3 = self.feature_down2(prev_x2)
        prev_x4 = self.feature_down3(prev_x3)
        next_x1 = self.feature_inc(next)
        next_x2 = self.feature_down1(next_x1)
        next_x3 = self.feature_down2(next_x2)
        next_x4 = self.feature_down3(next_x3)

        helm = None
        mask2 = self.mask_up(mask * 1.0) == 1
        boundary2 = self.mask_up(boundary * 1.0) == 1
        mask3 = self.mask_up(mask2 * 1.0) == 1
        boundary3 = self.mask_up(boundary2 * 1.0) == 1
        mask4 = self.mask_up(mask3 * 1.0) == 1
        boundary4 = self.mask_up(boundary3 * 1.0) == 1

        # vel_vort = None
        x4, helm, vel, _, _ = self.process4(prev_x4, next_x4, x4, mask4, boundary4, helm)
        x3, helm, vel, _, _ = self.process3(prev_x3, next_x3, x3, mask3, boundary3, helm)
        x2, helm, vel, _, _ = self.process2(prev_x2, next_x2, x2, mask2, boundary2, helm)
        x1, helm, vel, _, _ = self.process1(prev_x1, next_x1, x1, mask, boundary, helm)

        x = self.up2(x4, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)

        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        vel = vel.reshape([B, self.groups, 2, H, W])[:, 0]
        if self.dataset_name == 'z500_era5':
            helm = helm.reshape([B, self.groups, 2, H+16, W+16])[:, 0, ..., 8:-8, 8:-8]
        else:
            helm = helm.reshape([B, self.groups, 2, H+32, W+32])[:, 0, ..., 16:-16, 16:-16]

        if self.dataset_name == 'real_video':
            x = x.unsqueeze(-1)
        return x, helm, vel

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)
