import os
import numpy as np
import torch
import util, report_util, train_util

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
    # 'day_of_week',
    # 'week_of_year',
)
num_node = 17833  # from voronoi partition
# num_node = 4267  # from voronoi partition
num_dim_pairs = [
    (17833, 128),  # node_x
    # (4267, 128),  # node
    (24, 10),  # hour_of_day
    # (7, 10), # day_of_week
    # (52, 10), # week_of_year
]
slot_stride = 20
slot_spec = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300]

# ---------------- train spec -----------------------
epoch_size = 1
batch_size = 32
max_node_trj_len = 4000
offset_minute = 20
trj_min_minute = 60

# ---------------- test spec ------------------------
# test_size = 64
# num_test = 200
aso_data_zip_name = 'aso_data_v8.zip'
dso_data_zip_name = 'dso_data_v8.zip'


def get_loader(dataset_dir, zip_name, test_size):
    test_dataset = train_util.TrajectoryDataZipV8(os.path.join(dataset_dir, zip_name),
                                                  num_node,
                                                  feature_keys=feature_keys,
                                                  observable_minute=trj_min_minute,
                                                  offset_minute=offset_minute)
    return test_dataset.get_loader(batch_size=test_size, shuffle=True)


def run_test(max_num=10000):
    model.eval()

    for set_name, test_loader in ((s_name, t_loader()) for s_name, t_loader in [
        # ('Train set', lambda: loader.dataset.get_loader(batch_size=batch_size, shuffle=True)),
        ('Aso set', lambda: get_loader(dataset_dir, aso_data_zip_name, batch_size)),
        ('Dso set', lambda: get_loader(dataset_dir, dso_data_zip_name, batch_size)),
    ]):
        test_pre_y = np.array([])
        test_y = np.array([])
        # for j in range(num_test):
        with torch.no_grad():
            for data in test_loader:
                if len(test_y) > max_num:
                    break
                test_xs, test_x_sptrs, test_y_slots, test_seq_len = [d.to(device) for d in data]
                test_x_sptrs = test_x_sptrs.softmax(-1)
                if len(test_xs) > max_node_trj_len:
                    continue
                    # test_y_slots = test_log_probs = None
                    # break
                test_y_slots = util.domain_tran(test_y_slots, slot_spec).cpu().numpy()
                # test_log_probs = model(test_xs, test_x_sptrs, test_seq_len).cpu()
                test_log_probs = model(test_xs, test_x_sptrs, test_seq_len).cpu().argmax(-1).numpy()

                test_y = np.append(test_y, test_y_slots)
                test_pre_y = np.append(test_pre_y, test_log_probs)
                # test_y = test_y if test_log_probs is None else np.append(test_y, test_y_slots)
                # test_pre_y = test_pre_y if test_log_probs is None else np.append(test_pre_y, test_log_probs)

        print(f'{set_name} result: ')
        report_util.p_errors(test_pre_y, test_y, slot_stride)
    del test_xs, test_x_sptrs, test_y_slots, test_log_probs, test_y, test_pre_y


# ################### train loop ######################
# ---------------- run script -----------------------
loader = train_util.TrajectoryDataZipV8(train_data_zip_pth,
                                        num_node,
                                        feature_keys=feature_keys,
                                        observable_minute=trj_min_minute,
                                        offset_minute=offset_minute,
                                        ).get_loader(batch_size)
model = train_util.BaseModelV8(num_dim_pairs, len(slot_spec) + 1, dropout=0.01).to(device)
# model.auto_checkpoint_on = True
avm = util.AverageMeter()

criterion = util.SmoothNLLLoss(smoothing=0.1)

for j in range(epoch_size):

    # train
    model.train()
    for i, data in enumerate(loader):
        xs, x_sptrs, y_slots, seq_len = [d.to(device) for d in data]
        x_sptrs = x_sptrs.softmax(-1)
        if len(xs) > max_node_trj_len:
            print(f'len(x_nodes)={len(xs)}')
            continue
        y_slots = util.domain_tran(y_slots, slot_spec)
        slot_log_probs = model(xs, x_sptrs, seq_len)
        # slot_log_probs = model(xs, seq_len)
        loss = criterion(slot_log_probs, y_slots)
        with model.optimize_c():
            loss.backward()

        avm.log(float(loss))
        print(f'batch {i}/{len(loader.dataset) // loader.batch_size}, instant loss {loss.item():.4f}', end='\r')
        del xs, x_sptrs, y_slots, seq_len, slot_log_probs, loss

    print(f'epoch {j + 1}/{epoch_size}, average loss is {avm}')

    # test - training set
    run_test(50000)

    # checkpoint
    # model.check_point(avm.value, 'Model16062020.pt', verbose=True)
