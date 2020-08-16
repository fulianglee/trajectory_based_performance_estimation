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
slot_spec = [10, 20, 40, 60, 90, 130, 180, 230, 300, 390, 530, 800]
slot_stride = 10
# slot_spec = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300]

# ---------------- train spec -----------------------
num_node = 4267  # from voronoi partition
pre_model = train_util.RNNPre(num_node)

epoch_size = 100
batch_size = 256
max_node_trj_len = 2000
min_obsrv_len = 300
# min_num_row_of_trj = 30
# assert 0 < min_obsrv_len < min_num_row_of_trj

# ---------------- test spec ------------------------
test_size = 500
num_test = 10
aso_data_zip_name = 'aso_data_v5_l30.zip'
dso_data_zip_name = 'dso_data_v5_l30.zip'


def get_loader(dataset_dir, zip_name, test_size):
    test_dataset = train_util.TrajectoryDataZip(os.path.join(dataset_dir, zip_name))
    return test_dataset.get_loader(batch_size=test_size, shuffle=True)


# ################### train loop ######################
# ---------------- run script -----------------------
loader = train_util.TrajectoryDataZip(train_data_zip_pth,
                                      threshld_len=max_node_trj_len,
                                      output_len=max_node_trj_len,
                                      min_obsrv_len=min_obsrv_len).get_loader(batch_size)
model = train_util.TransferedClassifier(pre_model, len(slot_spec) + 1).to(device)
# model.auto_checkpoint_on = True
avm = util.AverageMeter()

criterion = util.SmoothNLLLoss(smoothing=0.1)

for j in range(epoch_size):

    # train
    model.train()
    for i, data in enumerate(loader):
        x_nodes, x_ts, y_slots, seq_len = [d.to(device) for d in data]
        util.domain_tran(y_slots, slot_spec)
        slot_log_probs = model(x_nodes, x_ts, seq_len)
        loss = criterion(slot_log_probs, y_slots)
        with model.optimize_c():
            loss.backward()

        avm.log(loss.tolist())
        print(f'batch {i}/{len(loader.dataset) // loader.batch_size}, instant loss {loss.item():.4f}', end='\r')
    print(f'epoch {j + 1}/{epoch_size}, average loss is {avm.value:.4f}')

    # test - training set
    if j == epoch_size - 1:
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
                        util.domain_tran(test_y_slots, slot_spec)
                        test_log_probs = model(test_x_nodes, test_x_ts, test_seq_len).cpu()
                        break

                test_pre_y = np.append(test_pre_y, test_log_probs.argmax(-1).numpy())
                test_y = np.append(test_y, test_y_slots.cpu().numpy())
            print(f'{set_name} result: ', report_util.p_errors(test_pre_y, test_y, slot_stride))

    # checkpoint
    model.check_point(avm.value, 'Model02062020.pt', verbose=True)

# test_loader = loader.dataset.get_loader(batch_size=test_size, shuffle=True)
# test_pre_y = np.array([])
# test_y = np.array([])
# for j in range(num_test):
#     with torch.no_grad():
#         for data in test_loader:
#             test_x_nodes, test_x_ts, test_y_slots, test_seq_len = [d.to(device) for d in data]
#             util.domain_tran(test_y_slots, slot_spec)
#             test_log_probs = model(test_x_nodes, test_x_ts, test_seq_len).cpu()
#             break
#
#     test_pre_y = np.append(test_pre_y, test_log_probs.argmax(-1).numpy())
#     test_y = np.append(test_y, test_y_slots.cpu().numpy())
# print('Train set result: ', report_util.p_errors(test_pre_y, test_y, minute_interval))
# # test - asso set
# test_loader = get_loader(dataset_dir, aso_data_zip_name, test_size)
# test_pre_y = np.array([])
# test_y = np.array([])
# for j in range(num_test):
#     with torch.no_grad():
#         for data in test_loader:
#             test_x_nodes, test_x_ts, test_y_slots, test_seq_len = [d.to(device) for d in data]
#             util.domain_tran(test_y_slots, slot_spec)
#             test_log_probs = model(test_x_nodes, test_x_ts, test_seq_len).cpu()
#
#     test_pre_y = np.append(test_pre_y, test_log_probs.argmax(-1).numpy())
#     test_y = np.append(test_y, test_y_slots.cpu().numpy())
# print('Asso set result: ', report_util.p_errors(test_pre_y, test_y, minute_interval))
# # test - disso set
# test_loader = get_loader(dataset_dir, dso_data_zip_name, test_size)
# test_pre_y = np.array([])
# test_y = np.array([])
# for j in range(num_test):
#     with torch.no_grad():
#         for data in test_loader:
#             test_x_nodes, test_x_ts, test_y_slots, test_seq_len = [d.to(device) for d in data]
#             util.domain_tran(test_y_slots, slot_spec)
#             test_log_probs = model(test_x_nodes, test_x_ts, test_seq_len).cpu()
#
#     test_pre_y = np.append(test_pre_y, test_log_probs.argmax(-1).numpy())
#     test_y = np.append(test_y, test_y_slots.cpu().numpy())
# print('Disso set result: ', report_util.p_errors(test_pre_y, test_y, minute_interval))

# 198 -> 145
# 192 -> 141
