import os
import numpy as np
import torch
import util, report_util, train_util
from trsfmr import subsequent_mask, BaseModelV9, BaseModelV9a, BaseModelV9b

# ################### spec ############################
dataset_dir = 'dataset'
train_data_file_name = 'v8_data.zip'
train_data_zip_pth = os.path.join(dataset_dir, train_data_file_name)

# assert torch.cuda.is_available(), f'cuda not available'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

feature_keys = (
    'node_x',
    # 'node',
    'hour_of_day',
    'day_of_week',
    # 'week_of_year',
)
num_node = 17833  # from voronoi partition
# num_node = 4267  # from voronoi partition
num_dim_pairs = [
    (17833, 116),  # node_x
    # (4267, 118),  # node
    (24, 8),  # hour_of_day
    (7, 4),  # day_of_week
    # (52, 10), # week_of_year
]

slot_stride = 20
slot_spec = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300]

# ---------------- model spec -----------------------
n_head = 8
d_ff = 1024
dropout = 0.05
n_layer = 9
n_class = 16

# ---------------- train spec -----------------------
epoch_size = 1
batch_size = 16
max_node_trj_len = 800
offset_minute = 20
trj_min_minute = 60

factor = 1  # lr scheduler
warmup_step = 4000  # lr scheduler

# ---------------- test spec ------------------------
aso_data_zip_name = 'aso_data_v8.zip'
dso_data_zip_name = 'dso_data_v8.zip'


def get_loader(dataset_dir, zip_name, test_size):
    test_dataset = train_util.TrajectoryDataZipV9(os.path.join(dataset_dir, zip_name),
                                                  num_node,
                                                  max_len=max_node_trj_len,
                                                  feature_keys=feature_keys,
                                                  observable_minute=trj_min_minute,
                                                  offset_minute=offset_minute)
    return test_dataset.get_loader(batch_size=test_size, shuffle=True)


def run_test(max_num=10000):
    model.eval()

    for set_name, test_loader in ((s_name, t_loader()) for s_name, t_loader in [
        ('Train set', lambda: loader.dataset.get_loader(batch_size=batch_size, shuffle=True)),
        ('Aso set', lambda: get_loader(dataset_dir, aso_data_zip_name, batch_size)),
        ('Dso set', lambda: get_loader(dataset_dir, dso_data_zip_name, batch_size)),
    ]):
        test_pre_y = np.array([])
        test_y = np.array([])
        with torch.no_grad():
            for data in test_loader:
                if len(test_y) > max_num:
                    break
                test_xs, test_ys, test_sizes = [d.to(device) for d in data]
                test_ys = util.domain_tran(test_ys, slot_spec).cpu().numpy()
                # model(xs.squeeze(), subsequent_mask(max(sizes)), sizes - 1)
                test_mask = subsequent_mask(max(test_sizes)).to(device)
                test_pred_ys = model(test_xs, test_mask, test_sizes - 1).cpu().argmax(-1).numpy()

                test_y = np.append(test_y, test_ys)
                test_pre_y = np.append(test_pre_y, test_pred_ys)

        print(f'{set_name} result: ')
        test_pre_y[test_pre_y > 15] = 15
        test_y[test_y > 15] = 15
        report_util.p_errors(test_pre_y, test_y, slot_stride)
    del test_xs, test_sizes, test_ys, test_pred_ys, test_y, test_pre_y


# # ################### train loop 0 ######################
# # ---------------- run script -----------------------
# print('train loop 0 starts ... ')
# loader = train_util.TrajectoryDataZipV9(train_data_zip_pth,
#                                         num_node,
#                                         max_len=max_node_trj_len,
#                                         feature_keys=feature_keys,
#                                         observable_minute=trj_min_minute,
#                                         offset_minute=offset_minute,
#                                         ).get_loader(batch_size)
#
# # model = BaseModelV9(num_dim_pairs, n_head, d_ff, n_layer, n_class, dropout).to(device)
# model = BaseModelV9b(num_dim_pairs, n_head, d_ff, n_layer, n_class, dropout).to(device)
# # model.auto_checkpoint_on = True
# avm = util.AverageMeter(300)
# # warmup = util.WarmUPRate(factor, sum([dim for _, dim in num_dim_pairs]), warmup_step)
#
# criterion = util.SmoothNLLLoss(smoothing=0.1)
# # criterion = util.SmoothDistNLLLoss(16, 0.2)
# # criterion = util.SmoothDistNLLLossV2(size=16, base_spread=0.1, max_sigma=20.0, device=device)
#
# for j in range(epoch_size):
#
#     # train
#     model.train()
#     for i, data in enumerate(loader, start=1):
#         # warmup.set_lr(model.optimizer, step=i)
#         xs, ys, sizes = [d.to(device) for d in data]
#         ys = util.domain_tran(ys, slot_spec)
#         mask = subsequent_mask(max(sizes)).to(device)
#         log_probs = model(xs, mask, sizes - 1)
#         loss = criterion(log_probs, ys)
#         with model.optimize_c():
#             loss.backward()
#
#         avm.log(float(loss))
#         print(f'batch {i}/{len(loader.dataset) // loader.batch_size}, instant loss {loss.item():.4f}, '
#               f'avg loss {avm.value:.4f} +- {avm.std:.2f}', end='\r')
#         del xs, ys, sizes, log_probs, loss
#
#     print(f'epoch {j + 1}/{epoch_size}, average loss is {avm}')
#
#     # test - training set
#     run_test(50000)
# print('train loop 0 finished ')

# ################### train loop 1 ######################
# ---------------- run script -----------------------
print('train loop 1 starts ... ')

feature_keys = (
    'node_x',
    'hour_of_day',
    'day_of_week',
)
num_node = 17833  # from voronoi partition
num_dim_pairs = [
    (17833, 52),  # node_x
    (24, 8),  # hour_of_day
    (7, 4),  # day_of_week
]

slot_stride = 20
slot_spec = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300]
n_head = 4
d_ff = 512
dropout = 0.05
n_layer = 12
n_class = 16
epoch_size = 1
batch_size = 16
max_node_trj_len = 1500
offset_minute = 20
trj_min_minute = 60

loader = train_util.TrajectoryDataZipV9(train_data_zip_pth,
                                        num_node,
                                        max_len=max_node_trj_len,
                                        feature_keys=feature_keys,
                                        observable_minute=trj_min_minute,
                                        offset_minute=offset_minute,
                                        ).get_loader(batch_size)
model = BaseModelV9b(num_dim_pairs, n_head, d_ff, n_layer, n_class, dropout).to(device)
avm = util.AverageMeter(300)

criterion = util.SmoothNLLLoss(smoothing=0.2)

for j in range(epoch_size):

    # train
    model.train()
    for i, data in enumerate(loader, start=1):
        xs, ys, sizes = [d.to(device) for d in data]
        ys = util.domain_tran(ys, slot_spec)
        mask = subsequent_mask(max(sizes)).to(device)
        log_probs = model(xs, mask, sizes - 1)
        loss = criterion(log_probs, ys)
        with model.optimize_c():
            loss.backward()

        avm.log(float(loss))
        print(f'batch {i}/{len(loader.dataset) // loader.batch_size}, instant loss {loss.item():.4f}, '
              f'avg loss {avm.value:.4f} +- {avm.std:.2f}', end='\r')
        del xs, ys, sizes, log_probs, loss

    print(f'epoch {j + 1}/{epoch_size}, average loss is {avm}')

    # test - training set
    run_test(50000)
print('train loop 1 finished ')

# ################### train loop 2 ######################
# ---------------- run script -----------------------
print('train loop 2 starts ... ')

feature_keys = (
    'node_x',
    'hour_of_day',
    'day_of_week',
)
num_node = 17833  # from voronoi partition
num_dim_pairs = [
    (17833, 52),  # node_x
    (24, 8),  # hour_of_day
    (7, 4),  # day_of_week
]

slot_stride = 20
slot_spec = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300,
             320, 340, 360, 380, 400]
n_head = 4
d_ff = 512
dropout = 0.05
n_layer = 12
n_class = 16
epoch_size = 1
batch_size = 16
max_node_trj_len = 1500
offset_minute = 20
trj_min_minute = 60

loader = train_util.TrajectoryDataZipV9(train_data_zip_pth,
                                        num_node,
                                        max_len=max_node_trj_len,
                                        feature_keys=feature_keys,
                                        observable_minute=trj_min_minute,
                                        offset_minute=offset_minute,
                                        ).get_loader(batch_size)

model = BaseModelV9b(num_dim_pairs, n_head, d_ff, n_layer, n_class, dropout).to(device)
avm = util.AverageMeter(300)

criterion = util.SmoothNLLLoss(smoothing=0.1)

for j in range(epoch_size):

    # train
    model.train()
    for i, data in enumerate(loader, start=1):
        xs, ys, sizes = [d.to(device) for d in data]
        ys = util.domain_tran(ys, slot_spec)
        mask = subsequent_mask(max(sizes)).to(device)
        log_probs = model(xs, mask, sizes - 1)
        loss = criterion(log_probs, ys)
        with model.optimize_c():
            loss.backward()

        avm.log(float(loss))
        print(f'batch {i}/{len(loader.dataset) // loader.batch_size}, instant loss {loss.item():.4f}, '
              f'avg loss {avm.value:.4f} +- {avm.std:.2f}', end='\r')
        del xs, ys, sizes, log_probs, loss

    print(f'epoch {j + 1}/{epoch_size}, average loss is {avm}')

    # test - training set
    run_test(50000)
print('train loop 2 finished ')

# # ################### train loop 3 ######################
# # ---------------- run script -----------------------
# print('train loop 3 starts ... ')
# max_node_trj_len = 800
# loader = train_util.TrajectoryDataZipV9(train_data_zip_pth,
#                                         num_node,
#                                         max_len=max_node_trj_len,
#                                         feature_keys=feature_keys,
#                                         observable_minute=trj_min_minute,
#                                         offset_minute=offset_minute,
#                                         ).get_loader(batch_size)
#
# model = BaseModelV9a(num_dim_pairs, n_head, d_ff, n_layer, n_class, dropout).to(device)
# avm = util.AverageMeter(300)
# warmup = util.WarmUPRate(factor, sum([dim for _, dim in num_dim_pairs]), warmup_step)
#
# criterion = util.SmoothNLLLoss(smoothing=0.1)
#
# for j in range(epoch_size):
#
#     # train
#     model.train()
#     for i, data in enumerate(loader, start=1):
#         warmup.set_lr(model.optimizer, step=i)
#         xs, ys, sizes = [d.to(device) for d in data]
#         ys = util.domain_tran(ys, slot_spec)
#         mask = subsequent_mask(max(sizes)).to(device)
#         log_probs = model(xs, mask, sizes - 1)
#         loss = criterion(log_probs, ys)
#         with model.optimize_c():
#             loss.backward()
#
#         avm.log(float(loss))
#         print(f'batch {i}/{len(loader.dataset) // loader.batch_size}, instant loss {loss.item():.4f}, '
#               f'avg loss {avm.value:.4f} +- {avm.std:.2f}', end='\r')
#         del xs, ys, sizes, log_probs, loss
#
#     print(f'epoch {j + 1}/{epoch_size}, average loss is {avm}')
#
#     # test - training set
#     run_test(50000)
# print('train loop 3 finished ')
