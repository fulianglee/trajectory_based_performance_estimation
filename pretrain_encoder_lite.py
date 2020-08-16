from zipfile import ZipFile
import os
import random
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch import nn
from torch.utils.data import Dataset, DataLoader
import data_util
import util

# net.to(device)
# inputs, labels = data[0].to(device), data[1].to(device)

#################### spec ############################
assert torch.cuda.is_available(), f'cuda not available'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

dataset_dir = 'dataset'
pre_train_data_file_name = 'v5_pretrain_data_lite.zip'
pre_train_data_zip_pth = os.path.join(dataset_dir, pre_train_data_file_name)

categorical_feature = ['node_edge', 'second_dff']
numerical_feature = ['second_of_day_cos', 'second_of_day_sin']

num_node = 4267  # from voronoi partition


# ################### pre train data set ########################
class PreTrainDataZip(Dataset):
    def __init__(self, zip_path: str, feature_keys=('node', 't_of_day_sin', 't_of_day_cos'),
                 threshld_len=100, output_len=100):
        self.zip_path = zip_path
        self.threshld_len = threshld_len
        self.output_len = output_len

        print(f'making ZipFile from {zip_path}... ', end='')
        self.zip_file = ZipFile(zip_path)
        print(f'done.')

        print(f'setting up access logic ... ', end='')
        self.df_keys = self.zip_file.namelist()
        self.num_dfs = len(self.df_keys)
        print(f'found {self.num_dfs} trajectories.')
        self.size = self.num_dfs
        print(f'data set size is {self.size}', flush=True)

        self.feature_keys = list(feature_keys)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        while True:
            df = self.get_df(idx)
            df = self.random_crop(df)
            if len(df) < 2:
                idx = random.randint(0, self.size - 1)
            else:
                break

        node = torch.tensor(df['node'], dtype=torch.long)
        t_of_day = torch.tensor(df[['t_of_day_sin', 't_of_day_cos']].to_numpy(), dtype=torch.float)

        return node[:-1], t_of_day[:-1], node[1:]

    def get_loader(self, batch_size=32, shuffle=True, num_workers=0):
        return DataLoader(self, batch_size, shuffle=shuffle, collate_fn=self.collat_fn, num_workers=num_workers)

    @staticmethod
    def collat_fn(data):
        # packing varying length of sequences
        # https://github.com/HarshTrivedi/packing-unpacking-pytorch-minimal-tutorial

        x_nodes = [x for x, t, y in data]
        x_ts = [t for x, t, y in data]
        y_nodes = [y for x, t, y in data]

        padded_x_nodes = pad_sequence(x_nodes)
        padded_x_ts = pad_sequence(x_ts)
        padded_y_nodes = pad_sequence(y_nodes)

        seq_lengths = torch.tensor(list(map(len, x_nodes)))
        seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)

        sorted_padded_x_nodes = padded_x_nodes[:, perm_idx]
        sorted_padded_x_ts = padded_x_ts[:, perm_idx]
        sorted_padded_y_nodes = padded_y_nodes[:, perm_idx]

        return sorted_padded_x_nodes, sorted_padded_x_ts, sorted_padded_y_nodes, seq_lengths

    def get_df(self, df_idx):
        df_key = self.df_keys[df_idx]
        return data_util.get_df_from_zip_file(self.zip_file, df_key)

    def random_crop(self, df: pd.DataFrame):
        df = df.reset_index(drop=True)
        df_len = len(df)
        if df_len > self.threshld_len:
            idx = random.randint(0, df_len - self.output_len)
            return df.iloc[idx:idx + self.output_len].reset_index(drop=True)
        return df


# ################### pre train RNN ###########################
class RNNPre(nn.Module):
    """
    pre train RNN on full trajectories to predict next node from previous,
        may as well try to predict t_of_day
    """

    def __init__(self, num_node, dropout=0.00):
        super(self.__class__, self).__init__()

        hidden_size = 256
        rnn_num_layer = 2
        node_embed_dim = 128
        clf_dim = 100

        time_of_day_dim = 2
        total_embed_dim = node_embed_dim + time_of_day_dim

        self.node_embed = nn.Embedding(num_node + 1, node_embed_dim)

        self.rnn = nn.GRU(total_embed_dim, hidden_size, rnn_num_layer, dropout=dropout)
        self.fc0 = nn.Linear(self.rnn.hidden_size, clf_dim)
        self.ac = nn.SELU()
        self.fc1 = nn.Linear(clf_dim, num_node)

        util.add_util(self.__class__, self, save_pth='check_points')
        util.add_checkpoint_util(self.__class__, self, auto_checkpoint_on=False, mode='descend')
        util.get_nn_params(self, self.__class__.__name__, True)

    def forward(self, node: torch.Tensor, t_of_day: torch.Tensor, seq_lengths=None):
        embeds = torch.cat([
            self.node_embed(node),
            t_of_day
        ], dim=-1)

        packed_input = nn.utils.rnn.pack_padded_sequence(embeds, seq_lengths, batch_first=False)
        out, hn = self.rnn(packed_input)
        out, input_sizes = nn.utils.rnn.pad_packed_sequence(out, batch_first=False)
        out = self.ac(self.fc0(out))
        out = self.fc1(out)
        log_probs = torch.log_softmax(out, dim=-1)
        return log_probs


def compute_masked_nlll(log_probs: torch.Tensor, target: torch.Tensor, seq_len):
    losses = -torch.gather(log_probs, dim=-1, index=target.unsqueeze(-1))

    mask = torch.ones_like(losses)
    for i, seq_length in enumerate(seq_len):
        mask[seq_length:, i] = 0

    losses = losses * mask
    loss = losses.sum() / seq_len.sum()
    return loss


# ################### train loop ######################
# ----------------- pre train -----------------
pre_loader = PreTrainDataZip(pre_train_data_zip_pth, threshld_len=300, output_len=300).get_loader(256, num_workers=3)
pre_model = RNNPre(num_node).to(device)
pre_model.auto_checkpoint_on = True
avm = util.AverageMeter()
epoch_size = 200

for j in range(epoch_size):
    # train
    for i, data in enumerate(pre_loader):
        x_nodes, x_ts, y_nodes, seq_len = [d.to(device) for d in data]
        node_log_probs = pre_model(x_nodes, x_ts, seq_len)
        loss = compute_masked_nlll(node_log_probs, y_nodes, seq_len)
        with pre_model.optimize_c():
            loss.backward()

        avm.log(loss.tolist())
        print(f'batch {i}/{len(pre_loader.dataset) // pre_loader.batch_size}, instant loss {loss.item():.4f}')
    print(f'epoch {j + 1}/{epoch_size}, average loss is {avm.value:.4f}')

    # checkpoint
    pre_model.check_point(avm.value, 'PreModel02062020.pt', verbose=True)

# loss goes from 8 to 0.1337
