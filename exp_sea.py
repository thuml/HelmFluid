import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib
from data_provider.sea_temperature_norm import InputHandle
matplotlib.use('Agg')
from timeit import default_timer
from utils.utilities3 import *
from utils.params import get_args
from model_dict import get_model
from utils.adam import Adam
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import math
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
    'total_length': T_in+T_out,
    'input_length': T_in,
    'type': 'train'
}
test_params = {
    'path': args.data_path,
    'total_length': T_in+T_out,
    'input_length': T_in,
    'type': 'valid'
}
train_loader = DataLoader(InputHandle(train_params), batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=20)
test_loader = DataLoader(InputHandle(test_params), batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=20)

################################################################
# training and evaluation
################################################################

optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

writer = SummaryWriter('results/logdir')

myloss = LpLoss(size_average=False)

train_iter = 0
step = 1
t1 = default_timer()
train_l2_step = 0
train_l2_full = 0

for ep in range(epochs):
    for xx, yy, time_data, position in train_loader:
        train_iter = train_iter + 1
        loss = 0
        xx = xx.to(device)
        yy = yy.to(device)
        time_data = time_data.to(device)
        position = position.to(device)

        for t in range(0, T_out, step):
            # print(t)
            y = yy[..., t:t + step]
            if 'Helm' in args.model:
                im, helm, vel = model(xx)
            elif 'embed' in args.model:
                im = model(xx)
            else:
                im = model(xx)
            loss += myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))

            if t == 0:
                pred = im
            else:
                pred = torch.cat((pred, im), -1)

            xx = torch.cat((xx[..., step:], im), dim=-1)

        train_l2_step += loss.item()
        l2_full = myloss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1))
        train_l2_full += l2_full.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if train_iter % ntrain == 0:
            t2 = default_timer()
            print(train_iter, t2 - t1, train_l2_step / ntrain / (T_out / step), train_l2_full / ntrain)
            t1 = default_timer()
            writer.add_scalar('train_l2_step',
                              train_l2_step / ntrain / (T_out / step),
                              train_iter)
            writer.add_scalar('train_l2_full',
                              train_l2_full / ntrain / (T_out / step),
                              train_iter)
            train_l2_step = 0
            train_l2_full = 0

            scheduler.step()

        if train_iter % ntest == 0:
            print('Start testing...')
            test_l2_step = 0
            test_l2_full = 0
            MSE_test = 0
            save_path = os.path.join(results_save_path, str(train_iter))
            os.mkdir(save_path)
            with torch.no_grad():
                sample = 0
                for xx, yy, time_data, position in test_loader:
                    loss = 0
                    sample = sample + 1
                    xx = xx.to(device)
                    yy = yy.to(device)
                    time_data = time_data.to(device)
                    position = position.to(device)
                    if 'Helm' in args.model:
                        helm_list = []
                        vel_list = []
                    for t in range(0, T_out, step):
                        y = yy[..., t:t + step]
                        if 'Helm' in args.model:
                            im, helm, vel = model(xx)#, time_data[..., t:t+T_in], time_data[..., t+T_in:t+T_in+step], position)
                            helm_list.append(helm.detach().cpu().numpy())
                            vel_list.append(vel.detach().cpu().numpy())
                        elif 'embed' in args.model:
                            im = model(xx, time_data[..., t:t+T_in], time_data[..., t+T_in:t+T_in+step], position)
                        else:
                            im = model(xx)
                        loss += myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))

                        if t == 0:
                            pred = im
                        else:
                            pred = torch.cat((pred, im), -1)

                        xx = torch.cat((xx[..., step:], im), dim=-1)

                    test_l2_step += loss.item()
                    yy = yy[..., :T_out]
                    test_l2_full += myloss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1)).item()
                    MSE_test += nn.MSELoss()(pred,yy).item()

                    if sample % 10 == 0 and sample <= 200:
                        X, Y = np.meshgrid(np.arange(0, yy.shape[-2], 1), np.arange(yy.shape[-3], 0, -1))
                        save_path_one = os.path.join(save_path, str(sample))
                        os.mkdir(save_path_one)
                        pred = pred.detach().cpu().numpy()
                        yy = yy.detach().cpu().numpy()
                        for t in range(T_out):
                            plt.imshow(pred[0, ..., t])
                            plt.colorbar()
                            plt.savefig(os.path.join(save_path_one, 'pd_{}.jpg'.format(str(100+t)[1:])))
                            plt.clf()
                            plt.imshow(yy[0, ..., t])
                            plt.colorbar()
                            plt.savefig(os.path.join(save_path_one, 'gt_{}.jpg'.format(str(100+t)[1:])))
                            plt.clf()
                            err = pred[0, ..., t] - yy[0, ..., t]
                            # m = max(abs(err.max()), abs(err.min()))
                            plt.imshow(err, cmap='coolwarm', vmax = 1, vmin = -1)
                            plt.colorbar()
                            plt.savefig(os.path.join(save_path_one, 'err_{}.jpg'.format(str(100+t)[1:])))
                            plt.clf()
                            if 'Helm' in args.model:
                                plt.imshow(helm_list[t][0,0])
                                plt.colorbar()
                                plt.savefig(os.path.join(save_path_one, 'phi_{}.jpg'.format(str(100+t)[1:])))
                                plt.clf()
                                plt.imshow(helm_list[t][0,1])
                                plt.colorbar()
                                plt.savefig(os.path.join(save_path_one, 'vorticity_{}.jpg'.format(str(100+t)[1:])))
                                plt.clf()
                                plt.imshow(yy[0, ..., t])

                                vel_draw = np.flip(vel_list[t][0], axis=-2)
                                vel_draw[1:] = -vel_draw[1:]
                                plt.quiver(X[::4, ::4] + 1.5, Y[::4, ::4] - 2, vel_draw[0, ::4, ::4],
                                           vel_draw[1, ::4, ::4], scale_units='xy', scale=1)
                                plt.savefig(os.path.join(save_path_one, 'gt_flow_{}.jpg'.format(str(100+t)[1:])))
                                plt.clf()
            model.train()

            print(test_l2_step / sample / (T_out / step),
                  test_l2_full / sample, MSE_test / sample)
            writer.add_scalar('test_l2_step',
                              test_l2_step / sample / (T_out / step),
                              train_iter)
            writer.add_scalar('test_l2_full',
                              test_l2_full / sample,
                              train_iter)
            writer.add_scalar('MSE_test',
                              MSE_test / sample,
                              train_iter)

            if not os.path.exists(os.path.join(model_save_path,str(train_iter))):
                os.makedirs(os.path.join(model_save_path,str(train_iter)))
            print('save model')
            torch.save(model.state_dict(), os.path.join(os.path.join(model_save_path,str(train_iter)), model_save_name))