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
pre_model.load('PreModel02062020.pt')
epoch_size = 100
batch_size = 256
max_node_trj_len = 2000
min_obsrv_len = 300
# min_num_row_of_trj = 30
# assert 0 < min_obsrv_len < min_num_row_of_trj

# ---------------- test spec ------------------------
test_size = 500
num_test = 10
aso_data_zip_name = 'aso_data_v5_n10.zip'
dso_data_zip_name = 'dso_data_v5_n10.zip'


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
model.auto_checkpoint_on = True
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


# from zipfile import ZipFile
# import copy, os, random
# import numpy as np
# import pandas as pd
# import torch
# from torch.nn.utils.rnn import pad_sequence
# from torch import nn
# import util, data_util, report_util, transform_util
# from torch.utils.data import Dataset, DataLoader
#
# #################### spec ############################
# assert torch.cuda.is_available(), f'cuda not available'
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)
#
# # minute_interval = 10
# # slot_count = 144
# # slot_spec = [1, 2, 4, 6, 9, 13, 18, 23, 30, 39, 53, 80]
# # slot_spec = [1, 2, 4, 6, 9, 13, 18, 23, 30]
# minute_interval = 1
# slot_count = 1440
# # slot_spec = [10, 20, 40, 60, 90, 130, 180, 230, 300, 390, 530, 800]
# slot_spec = [10, 20, 40, 60, 90, 130, 180, 230, 300]
# # slot_spec = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300]
#
# min_num_nodes_of_trj = 30
#
# dataset_dir = 'dataset'
# pre_train_data_file_name = 'v5_pretrain_data_lite.zip'
# pre_train_data_zip_pth = os.path.join(dataset_dir, pre_train_data_file_name)
# train_data_file_name = 'v5_data_lite.zip'
# train_data_zip_pth = os.path.join(dataset_dir, train_data_file_name)
# test_data_file_name = 'v5_data_lite.zip'
# test_data_zip_pth = os.path.join(dataset_dir, test_data_file_name)
#
# num_node = 4267  # from voronoi partition
#
#
# # ################### pre train RNN ###########################
# class RNNPre(nn.Module):
#     """
#     pre train RNN on full trajectories to predict next node from previous,
#         may as well try to predict t_of_day
#     """
#
#     def __init__(self, num_node, dropout=0.00):
#         super(self.__class__, self).__init__()
#
#         hidden_size = 256
#         rnn_num_layer = 2
#         node_embed_dim = 128
#         clf_dim = 100
#
#         time_of_day_dim = 2
#         total_embed_dim = node_embed_dim + time_of_day_dim
#
#         self.node_embed = nn.Embedding(num_node + 1, node_embed_dim)
#
#         self.rnn = nn.GRU(total_embed_dim, hidden_size, rnn_num_layer, dropout=dropout)
#         self.fc0 = nn.Linear(self.rnn.hidden_size, clf_dim)
#         self.ac = nn.SELU()
#         self.fc1 = nn.Linear(clf_dim, num_node)
#
#         util.add_util(self.__class__, self, save_pth='check_points')
#         util.add_checkpoint_util(self.__class__, self, auto_checkpoint_on=False, mode='descend')
#         util.get_nn_params(self, self.__class__.__name__, True)
#
#     def forward(self, node: torch.Tensor, t_of_day: torch.Tensor, seq_lengths=None):
#         embeds = torch.cat([
#             self.node_embed(node),
#             t_of_day
#         ], dim=-1)
#
#         packed_input = nn.utils.rnn.pack_padded_sequence(embeds, seq_lengths, batch_first=False)
#         out, hn = self.rnn(packed_input)
#         out, input_sizes = nn.utils.rnn.pad_packed_sequence(out, batch_first=False)
#         out = self.ac(self.fc0(out))
#         out = self.fc1(out)
#         log_probs = torch.log_softmax(out, dim=-1)
#         return log_probs
#
#
# # ######################## Data Set ##############################
# class TrajectoryDataZip(Dataset):
#     def __init__(
#             self, zip_path: str, target_key='eta_slot',
#             feature_keys=('node', 't_of_day_sin', 't_of_day_cos'),
#             threshld_len=100, output_len=100, min_obsrv_len=20,
#     ):
#         self.zip_path = zip_path
#         self.threshld_len = threshld_len
#         self.output_len = output_len
#         self.min_obsrv_len = min_obsrv_len
#         assert 0 < min_obsrv_len < min_num_nodes_of_trj
#
#         print(f'making ZipFile from {zip_path}... ', end='')
#         self.zip_file = ZipFile(zip_path)
#         print(f'done.')
#
#         print(f'setting up access logic ... ', end='')
#         self.df_keys = self.zip_file.namelist()
#         self.size = len(self.df_keys)
#         print(f'found {self.size} trajectories.')
#
#         self.target_key = target_key
#         self.feature_keys = list(feature_keys)
#
#     def __len__(self):
#         return self.size
#
#     def __getitem__(self, idx):
#         df = self.get_df(idx)
#         df = self.random_peek(df)
#
#         node = torch.tensor(df['node'], dtype=torch.long)
#         t_of_day = torch.tensor(df[['t_of_day_sin', 't_of_day_cos']].to_numpy(), dtype=torch.float)
#         eta_slot = torch.tensor(df.tail(1)['eta_slot'].to_numpy()[0], dtype=torch.long)
#
#         return node, t_of_day, eta_slot
#
#     def get_loader(self, batch_size=32, shuffle=True, num_workers=0):
#         return DataLoader(self, batch_size, shuffle=shuffle, collate_fn=self.collat_fn, num_workers=num_workers)
#
#     @staticmethod
#     def collat_fn(data):
#         # packing varying length of sequences
#         # https://github.com/HarshTrivedi/packing-unpacking-pytorch-minimal-tutorial
#
#         x_nodes = [x for x, t, y in data]
#         x_ts = [t for x, t, y in data]
#         ys = torch.stack([y for x, t, y in data])
#
#         padded_x_nodes = pad_sequence(x_nodes)
#         padded_x_ts = pad_sequence(x_ts)
#
#         seq_lengths = torch.tensor(list(map(len, x_nodes)))
#         seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
#
#         sorted_padded_x_nodes = padded_x_nodes[:, perm_idx]
#         sorted_padded_x_ts = padded_x_ts[:, perm_idx]
#         sorted_ys = ys[perm_idx]
#
#         return sorted_padded_x_nodes, sorted_padded_x_ts, sorted_ys, seq_lengths
#
#     def get_df(self, df_idx):
#         df_key = self.df_keys[df_idx]
#         return data_util.get_df_from_zip_file(self.zip_file, df_key)
#
#     def random_peek(self, df: pd.DataFrame):
#         """peek at a trajectory at random point with a maximum peek length, earliest point is second row"""
#         idx = random.randint(self.min_obsrv_len, len(df))
#         start_idx = max(0, idx - self.output_len)
#         df = df.iloc[start_idx:idx]
#
#         return transform_util.drop_duplicate_node(df).reset_index(drop=True)
#
#
# # ######################## Model #################################
# class TransferedClassifier(nn.Module):
#     def __init__(self, pre_train_model: RNNPre, num_eta_slots):
#         super(self.__class__, self).__init__()
#
#         clf_dim = 168
#
#         self.node_embed = copy.deepcopy(pre_train_model.node_embed)
#         self.rnn = copy.deepcopy(pre_train_model.rnn)
#
#         self.fc0 = nn.Linear(pre_train_model.rnn.hidden_size, clf_dim)
#         self.ac = nn.SELU()
#         self.fc1 = nn.Linear(self.fc0.out_features, num_eta_slots)
#
#         util.add_util(self.__class__, self, save_pth='check_points')
#         util.add_checkpoint_util(self.__class__, self, auto_checkpoint_on=False, mode='descend')
#         util.get_nn_params(self, self.__class__.__name__, True)
#
#     def forward(self, node: torch.Tensor, ts, seq_lengths=None):
#         embeds = torch.cat([
#             self.node_embed(node),
#             ts
#         ], dim=-1)
#
#         packed_input = nn.utils.rnn.pack_padded_sequence(embeds, seq_lengths, batch_first=False)
#         out, hn = self.rnn(packed_input)
#         out = self.ac(self.fc0(hn[-1]))
#         out = self.fc1(out)
#         log_probs = torch.log_softmax(out, dim=-1)
#         return log_probs
#
#
# # ################### train loop ######################
# pre_model = RNNPre(num_node)
# pre_model.load('PreModel02062020.pt')
#
# # ---------------- train spec -----------------------
# epoch_size = 200
#
# # ---------------- test spec ------------------------
# test_size = 1000
# num_test = 5
# aso_data_zip_name = 'aso_data_v5_lite.zip'
# dso_data_zip_name = 'dso_data_v5_lite.zip'
#
#
# def get_loader(dataset_dir, zip_name, test_size):
#     test_dataset = TrajectoryDataZip(os.path.join(dataset_dir, zip_name))
#     return test_dataset.get_loader(batch_size=test_size, shuffle=True)
#
#
# # ---------------- run script -----------------------
# loader = TrajectoryDataZip(train_data_zip_pth, threshld_len=500, output_len=500, min_obsrv_len=3).get_loader(256)
# model = TransferedClassifier(pre_model, len(slot_spec) + 1).to(device)
# model.auto_checkpoint_on = True
# avm = util.AverageMeter()
# # criterion = nn.NLLLoss()
# criterion = util.SmoothNLLLoss(smoothing=0.1)
#
# for j in range(epoch_size):
#
#     # train
#     model.train()
#     for i, data in enumerate(loader):
#         x_nodes, x_ts, y_slots, seq_len = [d.to(device) for d in data]
#         util.domain_tran(y_slots, slot_spec)
#         slot_log_probs = model(x_nodes, x_ts, seq_len)
#         loss = criterion(slot_log_probs, y_slots)
#         with model.optimize_c():
#             loss.backward()
#
#         avm.log(loss.tolist())
#         print(f'batch {i}/{len(loader.dataset) // loader.batch_size}, instant loss {loss.item():.4f}', end='\r')
#     print(f'epoch {j + 1}/{epoch_size}, average loss is {avm.value:.4f}')
#
#     # test - training set
#     if j == epoch_size - 1:
#         model.eval()
#         test_loader = loader.dataset.get_loader(batch_size=test_size, shuffle=True)
#         test_pre_y = np.array([])
#         test_y = np.array([])
#         for j in range(num_test):
#             with torch.no_grad():
#                 for data in test_loader:
#                     test_x_nodes, test_x_ts, test_y_slots, test_seq_len = [d.to(device) for d in data]
#                     util.domain_tran(test_y_slots, slot_spec)
#                     test_log_probs = model(test_x_nodes, test_x_ts, test_seq_len).cpu()
#                     break
#
#             test_pre_y = np.append(test_pre_y, test_log_probs.argmax(-1).numpy())
#             test_y = np.append(test_y, test_y_slots.cpu().numpy())
#         print('Train set result: ', report_util.p_errors(test_pre_y, test_y, minute_interval))
#         # test - asso set
#         test_loader = get_loader(dataset_dir, aso_data_zip_name, test_size)
#         test_pre_y = np.array([])
#         test_y = np.array([])
#         for j in range(num_test):
#             with torch.no_grad():
#                 for data in test_loader:
#                     test_x_nodes, test_x_ts, test_y_slots, test_seq_len = [d.to(device) for d in data]
#                     util.domain_tran(test_y_slots, slot_spec)
#                     test_log_probs = model(test_x_nodes, test_x_ts, test_seq_len).cpu()
#
#             test_pre_y = np.append(test_pre_y, test_log_probs.argmax(-1).numpy())
#             test_y = np.append(test_y, test_y_slots.cpu().numpy())
#         print('Asso set result: ', report_util.p_errors(test_pre_y, test_y, minute_interval))
#         # test - disso set
#         test_loader = get_loader(dataset_dir, dso_data_zip_name, test_size)
#         test_pre_y = np.array([])
#         test_y = np.array([])
#         for j in range(num_test):
#             with torch.no_grad():
#                 for data in test_loader:
#                     test_x_nodes, test_x_ts, test_y_slots, test_seq_len = [d.to(device) for d in data]
#                     util.domain_tran(test_y_slots, slot_spec)
#                     test_log_probs = model(test_x_nodes, test_x_ts, test_seq_len).cpu()
#
#             test_pre_y = np.append(test_pre_y, test_log_probs.argmax(-1).numpy())
#             test_y = np.append(test_y, test_y_slots.cpu().numpy())
#         print('Disso set result: ', report_util.p_errors(test_pre_y, test_y, minute_interval))
#
#     # checkpoint
#     model.check_point(avm.value, 'Model02062020.pt', verbose=True)
#
# # 198 -> 145
# # 192 -> 141
