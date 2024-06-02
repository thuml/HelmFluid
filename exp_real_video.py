import torch.nn.functional as F
import matplotlib.pyplot as plt
from timeit import default_timer
from utils.utilities3 import *
from utils.params import get_args
from model_dict import get_model
from data_provider.real_video import InputHandle
from utils.adam import Adam
import math
from utils.Perceptual_loss import VGGLoss
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import os
import cv2
import torch

torch.manual_seed(0)
np.random.seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True

################################################################
# configs
################################################################
args = get_args()
vgg_ckpt = os.path.join(args.data_path, 'vgg16-397923af.pth') # please download this file from https://download.pytorch.org/models/vgg16-397923af.pth
ntrain = args.ntrain
ntest = args.ntest
in_channels = args.in_dim
out_channels = args.out_dim
r1 = args.h_down
r2 = args.w_down
s1 = int(((args.h - 1) / r1) + 1)
s2 = int(((args.w - 1) / r2) + 1)
T_in = args.T_in
T_out = args.T_out

if args.file_path == 'real_1':
    test_T_out = 50
elif args.file_path == 'real_2':
    test_T_out = 63
elif args.file_path == 'real_3':
    test_T_out = 46

batch_size = args.batch_size
learning_rate = args.learning_rate
epochs = args.epochs
step_size = args.step_size
gamma = args.gamma

model_save_path = args.model_save_path
model_save_name = args.model_save_name

if args.anylearn == 1:
    results_save_path = os.path.join(model_save_path.split('/')[0], 'results')
    args.data_path = os.path.join(args.data_path, 'data')
    os.mkdir(results_save_path)
else:
    results_save_path = '../results_temp'
    os.makedirs(results_save_path, exist_ok=True)

################################################################
# models
################################################################
model = get_model(args)
print(count_params(model))

################################################################
# load data and data normalization
################################################################


train_params = {
    'path': args.data_path,
    'file_path': args.file_path,
    'total_length': T_in+T_out,
    'input_length': T_in,
    'type': 'train'
}

train_params_ft = {
    'path': args.data_path,
    'file_path': args.file_path,
    'total_length': T_in+2*T_out,
    'input_length': T_in,
    'type': 'train'
}

test_params = {
    'path': args.data_path,
    'file_path': args.file_path,
    'total_length': T_in+test_T_out,
    'input_length': T_in,
    'type': 'valid'
}
train_loader = DataLoader(InputHandle(train_params), batch_size=args.batch_size, shuffle=True, drop_last=True)
train_loader_ft = DataLoader(InputHandle(train_params_ft), batch_size=args.batch_size // 2, shuffle=True, drop_last=True)
test_loader = DataLoader(InputHandle(test_params), batch_size=1, shuffle=False, drop_last=True)

################################################################
# training and evaluation
################################################################
optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

writer = SummaryWriter('results/logdir')
myloss = LpLoss(size_average=False)
perceptual_loss = VGGLoss(pretrained_ckpt=vgg_ckpt)

def rgb_to_yuv(_rgb_image):
    rgb_image = _rgb_image
    matrix = np.array([[0.299, 0.587, 0.114],
                    [-0.14713, -0.28886, 0.436],
                    [0.615, -0.51499, -0.10001]]).astype(np.float32)
    if len(rgb_image.shape) == 4:
        rgb_image = rgb_image.unsqueeze(-1)
    if len(rgb_image.shape) == 5:
        matrix = torch.from_numpy(matrix)[None, None, None, ...].to(rgb_image.get_device())
        yuv_image = torch.einsum("abcde, abcef -> abcdf", matrix, rgb_image)
    elif len(rgb_image.shape) == 6:
        matrix = torch.from_numpy(matrix)[None, None, None, ..., None].to(rgb_image.get_device())
        yuv_image = torch.einsum("abcdel, abcefl -> abcdfl", matrix, rgb_image)
    return yuv_image.squeeze(-1)

ep_boundary = 50
ep_further = 100

step = 1
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_l2_step = 0
    train_l2_full = 0
    for xx, yy, mask, boundary in train_loader:
        # H, W, C, L
        loss = 0
        xx = xx.to(device)
        yy = yy.to(device)
        m_out = (mask.to(device)*1.0).unsqueeze(-1).unsqueeze(-1)
        xx_out = xx[..., -1:] * (1-m_out)

        for t in range(0, T_out, step):
            # print(t)
            y = yy[..., t:t + step]
            if 'Helm' in args.model:
                im, helm, vel = model(xx, mask, boundary)
            else:
                im = model(xx, mask, boundary)
            if ep > ep_boundary:
                im = im * m_out + xx_out
            loss += myloss(rgb_to_yuv(im)[:, mask[0]], rgb_to_yuv(y)[:, mask[0]])

            if t == 0:
                pred = im
            else:
                pred = torch.cat((pred, im), -1)

            xx = torch.cat((xx[..., step:], im), dim=-1)

        train_l2_step += loss.item()
        l2_full = myloss(rgb_to_yuv(pred).reshape(batch_size, -1), rgb_to_yuv(yy).reshape(batch_size, -1))
        train_l2_full += l2_full.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if ep == ep_further:
        train_loader = train_loader_ft
        batch_size = batch_size // 2
        T_out = T_out * 2

    test_l2_step = 0
    test_l2_full = 0
    MSE_test = 0
    test_l2_mask = 0
    test_VGG_loss = 0
    with torch.no_grad():
        for xx, yy, mask, boundary in test_loader:
            loss = 0
            xx = xx.to(device)
            yy = yy.to(device)
            m_out = (mask.to(device)*1.0).unsqueeze(-1).unsqueeze(-1)
            xx_out = xx[..., -1:] * (1-m_out)

            if 'Helm' in args.model:
                helm_list = []
                vel_list = []
            for t in range(0, test_T_out, step):
                y = yy[..., t:t + step]
                if 'Helm' in args.model:
                    im, helm, vel = model(xx, mask, boundary)
                    helm_list.append(helm.detach().cpu().numpy())
                    vel_list.append(vel.detach().cpu().numpy())
                else:
                    im = model(xx, mask, boundary)
                if ep > ep_boundary:
                    im = im * m_out + xx_out
                loss += myloss(rgb_to_yuv(im)[:, mask[0]], rgb_to_yuv(y)[:, mask[0]])
                test_VGG_loss += perceptual_loss(im[..., 0].permute(0,3,1,2), y[..., 0].permute(0,3,1,2)).item()

                if t == 0:
                    pred = im
                else:
                    pred = torch.cat((pred, im), -1)

                xx = torch.cat((xx[..., step:], im), dim=-1)

            test_l2_step += loss.item()
            mse_bound = mask[0]*1.0-boundary[0]*1.0
            test_l2_mask += myloss(pred[:, mse_bound==1], yy[:, mse_bound==1]).item()
            test_l2_full += myloss(pred[:, mask[0]], yy[:, mask[0]]).item()

            MSE_test += nn.MSELoss()(pred[:, mse_bound==1], yy[:, mse_bound==1]).item()
            if ep % 50 == 0:
                X, Y = np.meshgrid(np.arange(0, yy.shape[-4], 1), np.arange(yy.shape[-3], 0, -1))

                save_path = os.path.join(results_save_path, str(ep))
                os.mkdir(save_path)
                pred = pred.detach().cpu().numpy()
                yy = yy.detach().cpu().numpy()
                for t in range(test_T_out):
                    plt.imshow(pred[0, ..., t])
                    plt.colorbar()
                    plt.savefig(os.path.join(save_path, 'pd_{}.jpg'.format(str(100+t)[1:])))
                    plt.clf()
                    plt.imshow(yy[0, ..., t])
                    plt.colorbar()
                    plt.savefig(os.path.join(save_path, 'gt_{}.jpg'.format(str(100+t)[1:])))
                    plt.clf()
                    plt.imshow(pred[0, ..., t] - yy[0, ..., t])
                    plt.colorbar()
                    plt.savefig(os.path.join(save_path, 'err_{}.jpg'.format(str(100+t)[1:])))
                    plt.clf()
                    if 'Helm' in args.model:
                        plt.imshow(helm_list[t][0, 0])
                        plt.colorbar()
                        plt.savefig(os.path.join(save_path, 'phi_{}.jpg'.format(str(100+t)[1:])))
                        plt.clf()
                        plt.imshow(helm_list[t][0, 1])
                        plt.colorbar()
                        plt.savefig(os.path.join(save_path, 'vorticity_{}.jpg'.format(str(100+t)[1:])))
                        plt.clf()
                        plt.imshow(yy[0, ..., t])

                        vel_draw = np.flip(vel_list[t][0], axis=-2)
                        vel_draw[1:] = -vel_draw[1:]
                        plt.quiver(X[::4, ::4] + 1.5, Y[::4, ::4] - 2, vel_draw[0, ::4, ::4],
                                   vel_draw[1, ::4, ::4], scale_units='xy', scale=1)
                        plt.savefig(os.path.join(save_path, 'gt_flow_{}.jpg'.format(str(100+t)[1:])))
                        plt.clf()
    t2 = default_timer()
    scheduler.step()
    print(ep, t2 - t1, train_l2_step / ntrain / (T_out / step), train_l2_full / ntrain,
          test_l2_step / ntest / (T_out / step),
          test_l2_full / ntest, MSE_test / ntest, test_VGG_loss / ntest / test_T_out)
    writer.add_scalar('train_l2_step',
                      train_l2_step / ntrain / (T_out / step),
                      ep)
    writer.add_scalar('test_l2_step',
                      test_l2_step / ntest / (T_out / step),
                      ep)
    writer.add_scalar('test_l2_full',
                      test_l2_full / ntest,
                      ep)
    writer.add_scalar('MSE',
                      MSE_test / ntest,
                      ep)
    writer.add_scalar('test_l2_mask',
                      test_l2_mask / ntest,
                      ep)
    writer.add_scalar('VGG_loss',
                      test_VGG_loss / ntest / test_T_out,
                      ep)
    if ep % 100 == 0:
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        print('save model')
        torch.save(model.state_dict(), os.path.join(model_save_path, model_save_name.format(ep)))