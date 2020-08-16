import os
import numpy as np
import torch
import util, report_util, train_util

# ################### spec ############################
dataset_dir = 'dataset'
pre_train_data_file_name = 'v5_pretrain_data_lite.zip'
pre_train_data_zip_pth = os.path.join(dataset_dir, pre_train_data_file_name)
train_data_file_name = 'v5_data_n10.zip'
train_data_zip_pth = os.path.join(dataset_dir, train_data_file_name)

assert torch.cuda.is_available(), f'cuda not available'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

minute_interval = 1
slot_count = 1440
# slot_spec = [10, 20, 40, 60, 90, 130, 180, 230, 300, 390, 530, 800]
slot_stride = 20
slot_spec = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300]

# ---------------- train spec -----------------------
num_node = 4267  # from voronoi partition
pre_model = train_util.RNNPre(num_node)
# pre_model.load('PreModel02062020.pt')
epoch_size = 3
batch_size = 64
max_node_trj_len = 500
offset_minute = 10
trj_min_minute = 60

# ---------------- test spec ------------------------
test_size = 64
num_test = 50
aso_data_zip_name = 'aso_data_v5_n10.zip'
dso_data_zip_name = 'dso_data_v5_n10.zip'


def get_loader(dataset_dir, zip_name, test_size):
    test_dataset = train_util.TrajectoryDataZip(os.path.join(dataset_dir, zip_name))
    return test_dataset.get_loader(batch_size=test_size, shuffle=True)


# ################### train loop ######################
# ---------------- run script -----------------------
loader = train_util.TrajectoryDataZipV6(train_data_zip_pth,
                                        max_node_seq_len=max_node_trj_len,
                                        offset_minute=offset_minute,
                                        min_minute=trj_min_minute
                                        ).get_loader(batch_size)
model = train_util.AttentionClassifier(pre_model, len(slot_spec) + 1).to(device)
# model.auto_checkpoint_on = True
avm = util.AverageMeter()

criterion = util.SmoothNLLLoss(smoothing=0.1)

for j in range(epoch_size):

    # train
    model.train()
    for i, data in enumerate(loader):
        x_nodes, x_ts, y_slots, seq_len = [d.to(device) for d in data]
        if len(x_nodes) > max_node_trj_len:
            print(f'len(x_nodes)={len(x_nodes)}')
            continue
        y_slots = util.domain_tran(y_slots, slot_spec)
        slot_log_probs = model(x_nodes, x_ts, seq_len)
        loss = criterion(slot_log_probs, y_slots)
        with model.optimize_c():
            loss.backward()

        avm.log(float(loss))
        print(f'batch {i}/{len(loader.dataset) // loader.batch_size}, instant loss {loss.item():.4f}', end='\r')
        del x_nodes, x_ts, y_slots, seq_len, y_slots, slot_log_probs, loss

    print(f'epoch {j + 1}/{epoch_size}, average loss is {avm.value:.4f}')

    # test - training set

    model.eval()

    for set_name, test_loader in ((s_name, t_loader()) for s_name, t_loader in [
        ('Train set', lambda: loader.dataset.get_loader(batch_size=test_size, shuffle=True)),
        ('Aso set', lambda: get_loader(dataset_dir, aso_data_zip_name, test_size)),
        ('Dso set', lambda: get_loader(dataset_dir, dso_data_zip_name, test_size)),
    ]):
        test_pre_y = np.array([])
        test_y = np.array([])
        for j in range(num_test):
            with torch.no_grad():
                for data in test_loader:
                    test_x_nodes, test_x_ts, test_y_slots, test_seq_len = [d.to(device) for d in data]
                    if len(test_x_nodes) > 300:
                        test_y_slots = test_log_probs = None
                        break
                    test_y_slots = util.domain_tran(test_y_slots, slot_spec)
                    test_log_probs = model(test_x_nodes, test_x_ts, test_seq_len).cpu()
                    break

            test_y = test_y if test_log_probs is None else np.append(test_y, test_y_slots.cpu().numpy())
            test_pre_y = test_pre_y if test_log_probs is None else np.append(test_pre_y,
                                                                             test_log_probs.argmax(-1).numpy())
        print(f'{set_name} result: ', report_util.p_errors(test_pre_y, test_y, slot_stride))
        del test_x_nodes, test_x_ts, test_y_slots, test_log_probs, test_y, test_pre_y

    # checkpoint
    model.check_point(avm.value, 'Model02062020.pt', verbose=True)
