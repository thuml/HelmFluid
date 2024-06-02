import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from timeit import default_timer
from data_provider.smoke import SmokeDatasetMemory
from utils.utilities3 import *
from utils.params import get_args
from model_dict import get_model
from utils.adam import Adam
import math
from torch.utils.tensorboard import SummaryWriter
import os


torch.manual_seed(0)
np.random.seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True

################################################################
# configs
################################################################
args = get_args()

ntrain = args.ntrain
ntest = args.ntest
N = args.ntotal
in_channels = args.in_dim
out_channels = args.out_dim
r1 = args.z_down
r2 = args.h_down
r3 = args.w_down
s1 = int(((args.z - 1) / r1) + 1)
s2 = int(((args.h - 1) / r2) + 1)
s3 = int(((args.w - 1) / r3) + 1)
T_in = args.T_in
T_out = args.T_out

batch_size = args.batch_size
learning_rate = args.learning_rate
epochs = args.epochs
step_size = args.step_size
gamma = args.gamma

model_save_path = args.model_save_path
model_save_name = args.model_save_name

if args.anylearn == 1:
    results_save_path = os.path.join(model_save_path.split('/')[0], 'results')
    os.mkdir(results_save_path)
else:
    results_save_path = '../results'

################################################################
# models
################################################################
model = get_model(args)
print(count_params(model))

################################################################
# load data and data normalization
################################################################

test_batch_size = 4

train_dataset = SmokeDatasetMemory(args, split='train')
test_dataset = SmokeDatasetMemory(args, split='test')# var='f'
train_loader = train_dataset.loader()
test_loader = test_dataset.loader()
torch_boundary = torch.ones(s1, s2, s3)
torch_boundary[1:-1, 1:-1, 1:-1] = 0
torch_boundary = 1.0 - torch_boundary.to(device).unsqueeze(0).float()
torch_mask = torch.ones(s1, s2, s3).to(device).unsqueeze(0).float()

################################################################
# training and evaluation
################################################################
optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

writer = SummaryWriter('results/logdir')
myloss = LpLoss(size_average=False)

step = 1
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_l2_step = 0
    train_l2_full = 0
    cnt = 0
    for xx, yy in train_loader:
        cnt += 1
        if cnt % 10 == 0:
            print(cnt*2)
        loss = 0
        train_batch_size = batch_size
        train_T_out = T_out
        xx = xx.to(device)
        yy = yy.to(device)

        for t in range(0, train_T_out, step):
            # print(t)
            y = yy[..., t:t + step]
            if 'Helm' in args.model:
                im, helm, vel = model(xx, torch_mask.repeat(xx.shape[0],1,1,1), torch_boundary.repeat(xx.shape[0],1,1,1))
            else:
                im = model(xx, torch_mask.repeat(xx.shape[0],1,1,1))
            loss += myloss(im.reshape(xx.shape[0], -1), y.reshape(xx.shape[0], -1))

            if t == 0:
                pred = im.unsqueeze(-1)
            else:
                pred = torch.cat((pred, im.unsqueeze(-1)), -1)

            xx = torch.cat((xx[..., step:], im.unsqueeze(-1)), dim=-1)

        train_l2_step += loss.item()
        l2_full = myloss(pred.reshape(xx.shape[0], -1), yy.reshape(xx.shape[0], -1))
        train_l2_full += l2_full.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # break

    test_l2_step = 0
    test_l2_full = 0
    ep_save_path = os.path.join(results_save_path, str(ep))
    with torch.no_grad():
        sample = 0
        for xx, yy in test_loader:
            sample = sample + 1
            loss = 0
            xx = xx.to(device)
            yy = yy.to(device)
            if 'Helm' in args.model:
                helm_list = []
                vel_list = []
            for t in range(0, T_out, step):
                y = yy[..., t:t + step]
                if 'Helm' in args.model:
                    im, helm, vel = model(xx, torch_mask.repeat(xx.shape[0], 1, 1, 1),
                                          torch_boundary.repeat(xx.shape[0], 1, 1, 1))
                    helm_list.append(helm.detach().cpu().numpy())
                    vel_list.append(vel.detach().cpu().numpy())
                else:
                    im = model(xx, torch_mask.repeat(xx.shape[0],1,1,1))
                loss += myloss(im.reshape(xx.shape[0], -1), y.reshape(xx.shape[0], -1))

                if t == 0:
                    pred = im.unsqueeze(-1)
                else:
                    pred = torch.cat((pred, im.unsqueeze(-1)), -1)

                xx = torch.cat((xx[..., step:], im.unsqueeze(-1)), dim=-1)

            test_l2_step += loss.item()
            test_l2_full += myloss(pred.reshape(xx.shape[0], -1), yy.reshape(xx.shape[0], -1)).item()

            if ep % 5 == 0 and sample % 20 == 0:
                X, Y = np.meshgrid(np.arange(0, yy.shape[-4], 1), np.arange(yy.shape[-3], 0, -1))
                # print(yy.shape, X.shape, Y.shape, vel_list[0].shape)
                save_path_one = os.path.join(ep_save_path, str(sample))
                os.makedirs(save_path_one, exist_ok=True)
                pred = pred.detach().cpu().numpy()
                yy = yy.detach().cpu().numpy()
                for t in range(T_out):
                    plt.imshow(pred[0, ..., 16, 0, t], vmax=1,vmin=0)
                    plt.colorbar()
                    plt.savefig(os.path.join(save_path_one, 'pd_{}.jpg'.format(str(100+t)[1:])))
                    plt.clf()
                    plt.imshow(yy[0, ..., 16, 0, t], vmax=1,vmin=0)
                    plt.colorbar()
                    plt.savefig(os.path.join(save_path_one, 'gt_{}.jpg'.format(str(100+t)[1:])))
                    plt.clf()
                    plt.imshow(pred[0, ..., 16, 0, t] - yy[0, ..., 16, 0, t], vmin=-0.5, vmax=0.5, cmap='coolwarm')
                    plt.colorbar()
                    plt.savefig(os.path.join(save_path_one, 'err_{}.jpg'.format(str(100+t)[1:])))
                    plt.clf()
                    if 'Helm' in args.model:
                        plt.imshow(helm_list[t][0, 0, 8:-8, 8:-8, 24])
                        plt.colorbar()
                        plt.savefig(os.path.join(save_path_one, 'phi_{}.jpg'.format(str(100+t)[1:])))
                        plt.clf()
                        plt.imshow(helm_list[t][0, 3, 8:-8, 8:-8, 24])
                        plt.colorbar()
                        plt.savefig(os.path.join(save_path_one, 'vorticity_{}.jpg'.format(str(100+t)[1:])))
                        plt.clf()

                        vel_draw = np.flip(vel_list[t][0, :2, ..., 16], axis=-3)
                        vel_draw[1:] = -vel_draw[1:]

                        plt.imshow(yy[0, ..., 16, 0, t], vmax=1,vmin=0)
                        plt.quiver(X[::2, ::2] + 0.5, Y[::2, ::2] - 1, vel_draw[0, ::2, ::2],
                                   vel_draw[1, ::2, ::2], scale_units='xy', scale=1)
                        plt.savefig(os.path.join(save_path_one, 'gt_flow_{}.jpg'.format(str(100+t)[1:])))
                        plt.clf()
    t2 = default_timer()
    scheduler.step()
    print(ep, t2 - t1, train_l2_step / ntrain / (T_out / step), train_l2_full / ntrain,
          test_l2_step / ntest / (T_out / step),
          test_l2_full / ntest)
    writer.add_scalar('train_l2_step',
                      train_l2_step / ntrain / (T_out / step),
                      ep)
    writer.add_scalar('test_l2_step',
                      test_l2_step / ntest / (T_out / step),
                      ep)
    writer.add_scalar('test_l2_full',
                      test_l2_full / ntest,
                      ep)
    if ep % step_size == 0:
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        print('save model')
        torch.save(model.state_dict(), os.path.join(model_save_path, model_save_name.format(ep)))