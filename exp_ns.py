import torch.nn.functional as F
import matplotlib.pyplot as plt
from timeit import default_timer
from utils.utilities3 import *
from utils.params import get_args
from model_dict import get_model
from utils.adam import Adam
import math
from torch.utils.tensorboard import SummaryWriter
import os
import torch
import pickle

torch.manual_seed(0)
np.random.seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True

################################################################
# configs
################################################################
args = get_args()

if args.viscosity == '1e-5':
    args.data_path = os.path.join(args.data_path, 'NavierStokes_V1e-5_N1200_T20')
    TRAIN_PATH = os.path.join(args.data_path, './NavierStokes_V1e-5_N1200_T20.mat')
    TEST_PATH = os.path.join(args.data_path, './NavierStokes_V1e-5_N1200_T20.mat')
elif args.viscosity == '1e-3':
    args.data_path = os.path.join(args.data_path, 'NavierStokes_V1e-3_N5000_T50')
    TRAIN_PATH = os.path.join(args.data_path, './ns_V1e-3_N5000_T50.mat')
    TEST_PATH = os.path.join(args.data_path, './ns_V1e-3_N5000_T50.mat')
elif args.viscosity == '1e-4':
    args.data_path = os.path.join(args.data_path, 'NavierStokes_V1e-4_N10000_T30')
    TRAIN_PATH = os.path.join(args.data_path, './ns_V1e-4_N10000_T30.mat')
    TEST_PATH = os.path.join(args.data_path, './ns_V1e-4_N10000_T30.mat')
elif args.viscosity == '1e-5_128':
    args.data_path = os.path.join(args.data_path, 'NavierStokes_V1e-5_N1200_T20')
    TRAIN_PATH = os.path.join(args.data_path, './ns_data_128_1e-5.dat')
    TEST_PATH = os.path.join(args.data_path, './ns_data_128_1e-5.dat')
elif args.viscosity == '1e-5_256':
    args.data_path = os.path.join(args.data_path, 'NavierStokes_V1e-5_N1200_T20')
    TRAIN_PATH = os.path.join(args.data_path, './NavierStokes_R256_V1e-5_N1200_T20.dat')
    TEST_PATH = os.path.join(args.data_path, './NavierStokes_R256_V1e-5_N1200_T20.dat')
elif args.viscosity == '1e-5_512':
    args.data_path = os.path.join(args.data_path, 'NavierStokes_V1e-5_N1200_T20')
    TRAIN_PATH = os.path.join(args.data_path, './ns_data_512_1e-5.dat')
    TEST_PATH = os.path.join(args.data_path, './ns_data_512_1e-5.dat')

ntrain = args.ntrain
ntest = args.ntest
N = args.ntotal
in_channels = args.in_dim
out_channels = args.out_dim
r1 = args.h_down
r2 = args.w_down
s1 = int(((args.h - 1) / r1) + 1)
s2 = int(((args.w - 1) / r2) + 1)
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

if args.viscosity not in ['1e-5_128','1e-5_256','1e-5_512']:
    reader = MatReader(TRAIN_PATH)
    # print(reader.read_field('u').shape)
    # assert 1 == 0
    train_a = reader.read_field('u')[:ntrain, ::r1, ::r2, :T_in]
    train_u = reader.read_field('u')[:ntrain, ::r1, ::r2, T_in:T_in + T_out]

    test_a = reader.read_field('u')[-ntest:, ::r1, ::r2, :T_in]
    test_u = reader.read_field('u')[-ntest:, ::r1, ::r2, T_in:T_in + T_out]
else:
    with open(TRAIN_PATH, 'rb') as f:
        reader = pickle.load(f)
    train_a = reader['u'][:ntrain, ::r1, ::r2, :T_in]
    train_u = reader['u'][:ntrain, ::r1, ::r2, T_in:T_in + T_out]

    test_a = reader['u'][-ntest:, ::r1, ::r2, :T_in]
    test_u = reader['u'][-ntest:, ::r1, ::r2, T_in:T_in + T_out]
    train_u = torch.tensor(train_u)
    train_a = torch.tensor(train_a)
    test_u = torch.tensor(test_u)
    test_a = torch.tensor(test_a)

print(train_u.shape)
print(test_u.shape)

train_a = train_a.reshape(ntrain, s1, s2, T_in)
test_a = test_a.reshape(ntest, s1, s2, T_in)

test_batch_size = batch_size if 'Helm' not in args.model else 10

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=test_batch_size,
                                          shuffle=False)

################################################################
# training and evaluation
################################################################
if args.model == 'UNO_2D':
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)
else:
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

writer = SummaryWriter('./results/logdir')
myloss = LpLoss(size_average=False)

step = args.step
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_l2_step = 0
    train_l2_full = 0
    for xx, yy in train_loader:
        loss = 0
        if 'Helm' in args.model and args.viscosity == '1e-3' and ep <= 5:
            seq = torch.cat([xx, yy], dim=-1)
            xx = []
            yy = []
            for i in range(2):
                xx.append(seq[..., i*20:i*20+10])
                yy.append(seq[..., i*20+10:i*20+30])
            xx = torch.cat(xx, dim=0)
            yy = torch.cat(yy, dim=0)
            train_batch_size = batch_size * 2
            train_T_out = 20
        else:
            train_batch_size = batch_size
            train_T_out = T_out
        xx = xx.to(device)
        yy = yy.to(device)

        for t in range(0, train_T_out, step):
            # print(t)
            y = yy[..., t:t + step]
            if 'Helm' in args.model:
                im, helm, vel = model(xx)
            else:
                im = model(xx)
            loss += myloss(im.reshape(train_batch_size, -1), y.reshape(train_batch_size, -1))

            if t == 0:
                pred = im
            else:
                pred = torch.cat((pred, im), -1)

            xx = torch.cat((xx[..., step:], im), dim=-1)

        train_l2_step += loss.item()
        l2_full = myloss(pred.reshape(train_batch_size, -1), yy.reshape(train_batch_size, -1))
        train_l2_full += l2_full.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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
                    im, helm, vel = model(xx)
                    helm_list.append(helm.detach().cpu().numpy())
                    vel_list.append(vel.detach().cpu().numpy())
                else:
                    im = model(xx)
                loss += myloss(im.reshape(test_batch_size, -1), y.reshape(test_batch_size, -1))

                if t == 0:
                    pred = im
                else:
                    pred = torch.cat((pred, im), -1)

                xx = torch.cat((xx[..., step:], im), dim=-1)

            test_l2_step += loss.item()
            test_l2_full += myloss(pred.reshape(test_batch_size, -1), yy.reshape(test_batch_size, -1)).item()

            if ep % 50 == 0 and sample % 4 == 0:
                X, Y = np.meshgrid(np.arange(0, yy.shape[-3], 1), np.arange(yy.shape[-2], 0, -1))
                # print(yy.shape, X.shape, Y.shape, vel_list[0].shape)
                save_path_one = os.path.join(ep_save_path, str(sample))
                os.makedirs(save_path_one, exist_ok=True)
                pred = pred.detach().cpu().numpy()
                yy = yy.detach().cpu().numpy()
                for t in range(T_out):
                    plt.imshow(pred[0, ..., t], vmin=-3, vmax=3)
                    plt.colorbar()
                    plt.savefig(os.path.join(save_path_one, 'pd_{}.jpg'.format(str(100+t)[1:])))
                    plt.clf()
                    plt.imshow(yy[0, ..., t], vmin=-3, vmax=3)
                    plt.colorbar()
                    plt.savefig(os.path.join(save_path_one, 'gt_{}.jpg'.format(str(100+t)[1:])))
                    plt.clf()
                    plt.imshow(pred[0, ..., t] - yy[0, ..., t], vmin=-2, vmax=2, cmap='coolwarm')
                    plt.colorbar()
                    plt.savefig(os.path.join(save_path_one, 'err_{}.jpg'.format(str(100+t)[1:])))
                    plt.clf()
                    if 'Helm' in args.model and args.viscosity != '1e-5_128':
                        plt.imshow(helm_list[t][0, 0, 16:-16, 16:-16])
                        plt.colorbar()
                        plt.savefig(os.path.join(save_path_one, 'phi_{}.jpg'.format(str(100+t)[1:])))
                        plt.clf()
                        plt.imshow(helm_list[t][0, 1, 16:-16, 16:-16])
                        plt.colorbar()
                        plt.savefig(os.path.join(save_path_one, 'vorticity_{}.jpg'.format(str(100+t)[1:])))
                        plt.clf()

                        # vel, vel_phi, vel_vorticity = models.HelmNet_2D_feature_group_trm.aggregate_to_velocity(torch.tensor(helm_list[t][:, 0, 0:1]).cuda(),
                        #                                                                                         torch.tensor(helm_list[t][:, 0, 1:2]).cuda(),
                        #                                                                                         (yy.shape[-3], yy.shape[-2]))
                        vel_draw = np.flip(vel_list[t][0], axis=-2)
                        # vel_draw = np.flip(vel[0].detach().cpu().numpy(), axis=-2)
                        vel_draw[1:] = -vel_draw[1:]
                        # vel_phi = np.flip(vel_phi[0].detach().cpu().numpy(), axis=-2)
                        # vel_phi[1:] = -vel_phi[1:]
                        # vel_vorticity = np.flip(vel_vorticity[0].detach().cpu().numpy(), axis=-2)
                        # vel_vorticity[1:] = -vel_vorticity[1:]

                        plt.imshow(yy[0, ..., t], vmin=-3, vmax=3)
                        plt.quiver(X[::4, ::4] + 1.5, Y[::4, ::4] - 2, vel_draw[0, ::4, ::4],
                                   vel_draw[1, ::4, ::4], scale_units='xy', scale=1)
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