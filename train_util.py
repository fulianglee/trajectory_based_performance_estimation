import torch
from zipfile import ZipFile
import copy, random
import pandas as pd
import util, data_util, transform_util
from data_util import get_df_from_zip_file
from util import add_util, add_checkpoint_util, get_nn_params
from transform_util import drop_duplicate_node
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import json
import os
from collections import Counter


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

        if seq_lengths is None:
            out, hn = self.rnn(embeds.unsqueeze(1))
            out = self.ac(self.fc0(hn[-1]))
            out = self.fc1(out).squeeze()
            log_probs = torch.log_softmax(out, dim=-1)
            return log_probs

        packed_input = nn.utils.rnn.pack_padded_sequence(embeds, seq_lengths, batch_first=False)
        out, hn = self.rnn(packed_input)
        out, input_sizes = nn.utils.rnn.pad_packed_sequence(out, batch_first=False)
        out = self.ac(self.fc0(out))
        out = self.fc1(out)
        log_probs = torch.log_softmax(out, dim=-1)
        return log_probs


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
                print('found short data frame!')
                idx = random.randint(0, self.size - 1)
            else:
                break

        node = torch.tensor(df['node'], dtype=torch.long)
        t_of_day = torch.tensor(df[['t_of_day_sin', 't_of_day_cos']].to_numpy(), dtype=torch.float)

        return node[:-1], t_of_day[:-1], node[1:]

    def get_loader(self, batch_size=32, shuffle=True):
        return DataLoader(self, batch_size, shuffle=shuffle, collate_fn=self.collat_fn)

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


# ######################## Model #################################
class TransferedClassifier(nn.Module):
    def __init__(self, pre_train_model: RNNPre, num_eta_slots):
        super(self.__class__, self).__init__()

        clf_dim = 168

        self.node_embed = copy.deepcopy(pre_train_model.node_embed)
        self.rnn = copy.deepcopy(pre_train_model.rnn)

        self.fc0 = nn.Linear(pre_train_model.rnn.hidden_size, clf_dim)
        self.ac = nn.SELU()
        self.fc1 = nn.Linear(self.fc0.out_features, num_eta_slots)

        util.add_util(self.__class__, self, save_pth='check_points')
        util.add_checkpoint_util(self.__class__, self, auto_checkpoint_on=False, mode='descend')
        util.get_nn_params(self, self.__class__.__name__, True)

    def forward(self, node: torch.Tensor, ts, seq_lengths=None):
        embeds = torch.cat([
            self.node_embed(node),
            ts
        ], dim=-1)

        packed_input = nn.utils.rnn.pack_padded_sequence(embeds, seq_lengths, batch_first=False)
        out, hn = self.rnn(packed_input)
        out = self.ac(self.fc0(hn[-1]))
        out = self.fc1(out)
        log_probs = torch.log_softmax(out, dim=-1)
        return log_probs


# ######################## Data Set ##############################
class TrajectoryDataZip(Dataset):
    def __init__(
            self, zip_path: str, target_key='eta_slot',
            feature_keys=('node', 't_of_day_sin', 't_of_day_cos'),
            threshld_len=100, output_len=100, min_obsrv_len=200, min_obsrv_node=5
    ):
        self.zip_path = zip_path
        self.threshld_len = threshld_len
        self.output_len = output_len
        self.min_obsrv_len = min_obsrv_len
        self.min_obsrv_node = min_obsrv_node

        print(f'making ZipFile from {zip_path}... ', end='')
        self.zip_file = ZipFile(zip_path)
        print(f'done.')

        print(f'setting up access logic ... ', end='')
        self.df_keys = self.zip_file.namelist()
        self.size = len(self.df_keys)
        print(f'found {self.size} trajectories.')

        self.target_key = target_key
        self.feature_keys = list(feature_keys)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        df = self.get_df(idx)
        df = self.random_peek(df)

        node = torch.tensor(df['node'], dtype=torch.long)
        t_of_day = torch.tensor(df[['t_of_day_sin', 't_of_day_cos']].to_numpy(), dtype=torch.float)
        eta_slot = torch.tensor(df.tail(1)['eta_slot'].to_numpy()[0], dtype=torch.long)

        return node, t_of_day, eta_slot

    def get_loader(self, batch_size=32, shuffle=True, num_workers=0):
        return DataLoader(self, batch_size, shuffle=shuffle, collate_fn=self.collat_fn, num_workers=num_workers)

    @staticmethod
    def collat_fn(data):
        # packing varying length of sequences
        # https://github.com/HarshTrivedi/packing-unpacking-pytorch-minimal-tutorial

        x_nodes = [x for x, t, y in data]
        x_ts = [t for x, t, y in data]
        ys = torch.stack([y for x, t, y in data])

        padded_x_nodes = pad_sequence(x_nodes)
        padded_x_ts = pad_sequence(x_ts)

        seq_lengths = torch.tensor(list(map(len, x_nodes)))
        seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)

        sorted_padded_x_nodes = padded_x_nodes[:, perm_idx]
        sorted_padded_x_ts = padded_x_ts[:, perm_idx]
        sorted_ys = ys[perm_idx]

        return sorted_padded_x_nodes, sorted_padded_x_ts, sorted_ys, seq_lengths

    def get_df(self, df_idx):
        df_key = self.df_keys[df_idx]
        return data_util.get_df_from_zip_file(self.zip_file, df_key)

    def random_peek(self, df: pd.DataFrame):
        """peek at a trajectory at random point with a maximum peek length, earliest point is second row"""
        end_idx = len(df)
        start_idx = min(self.min_obsrv_len, end_idx)
        idx = random.randint(start_idx, end_idx)
        # node_idx = df[df['node'] == df['node'].unique()[self.min_obsrv_node - 1]].index[0]
        # start_idx = max(0, idx - self.output_len)
        # df = df.iloc[start_idx:idx]
        df = df.iloc[:idx]

        return transform_util.drop_duplicate_node(df).reset_index(drop=True)


#################### data set ########################
class TrajectoryDataZipV4(Dataset):
    def __init__(
            self, zip_path: str, observable_minute=60, offset_minute=15, target_key='eta_slot',
            idx_key='minute_of_trj',
            feature_keys=('node', 'hour_of_day')
    ):
        self.observable_minute = observable_minute
        self.offset_minute = offset_minute
        self.zip_path = zip_path
        self.idx_key = idx_key

        print(f'making ZipFile from {zip_path}... ', end='')
        self.zip_file = ZipFile(zip_path)
        print(f'done.')

        print(f'setting up access logic ... ', end='')
        self.df_keys = self.zip_file.namelist()
        self.num_dfs = len(self.df_keys)
        print(f'found {self.num_dfs} trajectories.')

        print(f'counting data set size ... ')
        self.df_sizes = []
        with tqdm(total=len(self.df_keys)) as pbar:
            for df_key in self.df_keys:
                self.df_sizes.append(self.get_df_size(df_key))
                pbar.update(1)

        self._partitions = np.cumsum(self.df_sizes).tolist()
        self.size = sum(self.df_sizes)
        print(f'data set size is {self.size}', flush=True)

        self.target_key = target_key
        self.feature_keys = list(feature_keys)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):

        df_idx = self.get_df_idx(idx)
        df = self.get_df(df_idx)

        offset = self.get_offset(idx)
        cut_df = self.cut_df_by_offset(df, offset)

        x, y = cut_df[self.feature_keys].to_numpy(), cut_df[self.target_key].iloc[-1]
        x = np.expand_dims(x, axis=0) if x.ndim == 1 else x

        x, y = map(torch.tensor, [x, y])

        return x, y

    def get_loader(self, batch_size=32, shuffle=True):
        return DataLoader(self, batch_size, shuffle=shuffle, collate_fn=self.collat_fn)

    @staticmethod
    def collat_fn(data):
        # packing varying length of sequences
        # https://github.com/HarshTrivedi/packing-unpacking-pytorch-minimal-tutorial
        xs = [x for x, y in data]
        ys = torch.stack([y for x, y in data])

        padded_xs = pad_sequence(xs)

        seq_lengths = torch.tensor(list(map(len, xs)))
        # assert seq_lengths.type() == 'torch.LongTensor', f'seq_lengths need to be a LongTensor!'
        seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)

        sorted_padded_xs = padded_xs[:, perm_idx]
        sorted_ys = ys[perm_idx]

        return sorted_padded_xs, sorted_ys, seq_lengths

    def get_offset(self, idx):
        df_idx = self.get_df_idx(idx)
        offset = idx - ([0] + self._partitions)[df_idx]
        return offset

    def get_df_idx(self, idx):
        return next(part_idx for part_idx, part_right in enumerate(self._partitions) if part_right > idx)

    def get_df(self, df_idx):
        df_key = self.df_keys[df_idx]
        return data_util.get_df_from_zip_file(self.zip_file, df_key)

    def cut_df_by_offset(self, df: pd.DataFrame, offset: int):
        # offset start from backward
        last_minute = df[self.idx_key].tail(1).to_list().pop()
        later_time = last_minute - offset * self.offset_minute
        earlier_time = last_minute - offset * self.offset_minute - self.observable_minute

        while True:
            assert not earlier_time < 0, 'earlier time smaller than 0!'
            cut_df = df[(df[self.idx_key] > earlier_time) & (df[self.idx_key] < later_time)]
            if len(cut_df) > 0:
                return cut_df

            later_time -= self.offset_minute
            earlier_time -= self.offset_minute

    def get_df_size(self, df_key):
        df = data_util.get_df_from_zip_file(self.zip_file, df_key)
        minute_seq = df[self.idx_key]
        size = 0
        later_time = minute_seq.tail(1).to_list().pop()
        earlier_time = later_time - self.observable_minute
        while True:
            if earlier_time < 0:
                break
            if len(minute_seq[(minute_seq < later_time) & (minute_seq > earlier_time)]) > 0:
                size += 1
            later_time -= self.offset_minute
            earlier_time -= self.offset_minute
        return size


class TrajectoryDataZipV4a6(Dataset):
    def __init__(
            self, zip_path: str, observable_minute=60, offset_minute=15, target_key='eta_slot',
            idx_key='eta_slot',
            feature_keys=('node', 't_of_day_sin', 't_of_day_cos'),
    ):
        self.observable_minute = observable_minute
        self.offset_minute = offset_minute
        self.zip_path = zip_path
        self.idx_key = idx_key

        print(f'making ZipFile from {zip_path}... ', end='')
        self.zip_file = ZipFile(zip_path)
        print(f'done.')

        print(f'setting up access logic ... ', end='')
        self.df_keys = self.zip_file.namelist()
        self.num_dfs = len(self.df_keys)
        print(f'found {self.num_dfs} trajectories.')

        print(f'counting data set size ... ')
        self.df_sizes = []
        for df_key in tqdm(self.df_keys):
            self.df_sizes.append(self.get_df_size(df_key))
        self._partitions = np.cumsum(self.df_sizes).tolist()
        self.size = sum(self.df_sizes)
        print(f'data set size is {self.size}', flush=True)

        self.target_key = target_key
        self.feature_keys = list(feature_keys)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):

        df_idx = self.get_df_idx(idx)
        df = self.get_df(df_idx)

        offset = self.get_offset(idx)
        cut_df = self.cut_df_by_offset(df, offset)

        node = torch.tensor(cut_df['node'], dtype=torch.long)
        t_of_day = torch.tensor(cut_df[['t_of_day_sin', 't_of_day_cos']].to_numpy(), dtype=torch.float)
        eta_slot = torch.tensor(cut_df.tail(1)['eta_slot'].to_numpy()[0], dtype=torch.long)

        return node, t_of_day, eta_slot

        # x, y = cut_df[self.feature_keys].to_numpy(), cut_df[self.target_key].iloc[-1]
        # x = np.expand_dims(x, axis=0) if x.ndim == 1 else x
        #
        # x, y = map(torch.tensor, [x, y])
        #
        # return x, y

    def get_loader(self, batch_size=32, shuffle=True):
        return DataLoader(self, batch_size, shuffle=shuffle, collate_fn=self.collat_fn)

    @staticmethod
    def collat_fn(data):
        # packing varying length of sequences
        # https://github.com/HarshTrivedi/packing-unpacking-pytorch-minimal-tutorial

        x_nodes = [x for x, t, y in data]
        x_ts = [t for x, t, y in data]
        ys = torch.stack([y for x, t, y in data])

        padded_x_nodes = pad_sequence(x_nodes)
        padded_x_ts = pad_sequence(x_ts)

        seq_lengths = torch.tensor(list(map(len, x_nodes)))
        seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)

        sorted_padded_x_nodes = padded_x_nodes[:, perm_idx]
        sorted_padded_x_ts = padded_x_ts[:, perm_idx]
        sorted_ys = ys[perm_idx]

        return sorted_padded_x_nodes, sorted_padded_x_ts, sorted_ys, seq_lengths

    def get_offset(self, idx):
        df_idx = self.get_df_idx(idx)
        offset = idx - ([0] + self._partitions)[df_idx]
        return offset

    def get_df_idx(self, idx):
        return next(part_idx for part_idx, part_right in enumerate(self._partitions) if part_right > idx)

    def get_df(self, df_idx):
        df_key = self.df_keys[df_idx]
        return data_util.get_df_from_zip_file(self.zip_file, df_key)

    def cut_df_by_offset(self, df: pd.DataFrame, offset: int):
        # offset start from backward
        minute_seq = df[self.idx_key][::-1].reset_index(drop=True)
        last_minute = minute_seq.tail(1).to_list().pop()
        later_time = last_minute - offset * self.offset_minute
        earlier_time = last_minute - offset * self.offset_minute - self.observable_minute

        while True:
            assert not earlier_time < 0, 'earlier time smaller than 0!'
            cut_df = df[(minute_seq > earlier_time) & (minute_seq < later_time)]
            if len(cut_df) > 0:
                return cut_df.reset_index(drop=True)

            later_time -= self.offset_minute
            earlier_time -= self.offset_minute

    def get_df_size(self, df_key):
        df = data_util.get_df_from_zip_file(self.zip_file, df_key)
        minute_seq = df[self.idx_key][::-1]
        size = 0
        later_time = minute_seq.tail(1).to_list().pop()
        earlier_time = later_time - self.observable_minute
        while True:
            if earlier_time < 0:
                break
            if len(minute_seq[(minute_seq < later_time) & (minute_seq > earlier_time)]) > 0:
                size += 1
            later_time -= self.offset_minute
            earlier_time -= self.offset_minute
        return size


class TrajectoryDataZipV4b(Dataset):
    """works on v5 v6 data"""

    def __init__(
            self, zip_path: str, observable_minute=60, offset_minute=15, y_norm_factor=1000,
            target_key='eta_slot',
            idx_key='eta_slot',
            feature_keys=('node', 't_of_day_sin', 't_of_day_cos'),
    ):
        self.observable_minute = observable_minute
        self.offset_minute = offset_minute
        self.y_norm_factor = y_norm_factor
        self.zip_path = zip_path
        self.idx_key = idx_key

        print(f'making ZipFile from {zip_path}... ', end='')
        self.zip_file = ZipFile(zip_path)
        print(f'done.')

        print(f'setting up access logic ... ', end='')
        self.df_keys = self.zip_file.namelist()
        self.num_dfs = len(self.df_keys)
        print(f'found {self.num_dfs} trajectories.')

        print(f'counting data set size ... ')
        self.df_sizes = []
        for df_key in tqdm(self.df_keys):
            self.df_sizes.append(self.get_df_size(df_key))
        self._partitions = np.cumsum(self.df_sizes).tolist()
        self.size = sum(self.df_sizes)
        print(f'data set size is {self.size}', flush=True)

        self.target_key = target_key
        self.feature_keys = list(feature_keys)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):

        df_idx = self.get_df_idx(idx)
        df = self.get_df(df_idx)

        offset = self.get_offset(idx)
        cut_df = self.cut_df_by_offset(df, offset)

        node = torch.tensor(cut_df['node'], dtype=torch.long)
        t_of_day = torch.tensor(cut_df[['t_of_day_sin', 't_of_day_cos']].to_numpy(), dtype=torch.float)
        eta_slot = torch.tensor(cut_df.tail(1)['eta_slot'].to_numpy()[0], dtype=torch.float) / 1000

        return node, t_of_day, eta_slot

    def get_loader(self, batch_size=32, shuffle=True):
        return DataLoader(self, batch_size, shuffle=shuffle, collate_fn=self.collat_fn)

    @staticmethod
    def collat_fn(data):
        # packing varying length of sequences
        # https://github.com/HarshTrivedi/packing-unpacking-pytorch-minimal-tutorial

        x_nodes = [x for x, t, y in data]
        x_ts = [t for x, t, y in data]
        ys = torch.stack([y for x, t, y in data])

        padded_x_nodes = pad_sequence(x_nodes)
        padded_x_ts = pad_sequence(x_ts)

        seq_lengths = torch.tensor(list(map(len, x_nodes)))
        seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)

        sorted_padded_x_nodes = padded_x_nodes[:, perm_idx]
        sorted_padded_x_ts = padded_x_ts[:, perm_idx]
        sorted_ys = ys[perm_idx]

        return sorted_padded_x_nodes, sorted_padded_x_ts, sorted_ys, seq_lengths

    def get_offset(self, idx):
        df_idx = self.get_df_idx(idx)
        offset = idx - ([0] + self._partitions)[df_idx]
        return offset

    def get_df_idx(self, idx):
        return next(part_idx for part_idx, part_right in enumerate(self._partitions) if part_right > idx)

    def get_df(self, df_idx):
        df_key = self.df_keys[df_idx]
        return get_df_from_zip_file(self.zip_file, df_key)

    def cut_df_by_offset(self, df: pd.DataFrame, offset: int):
        # offset start from backward
        minute_seq = df[self.idx_key][::-1].reset_index(drop=True)
        last_minute = minute_seq.tail(1).to_list().pop()
        later_time = last_minute - offset * self.offset_minute
        earlier_time = last_minute - offset * self.offset_minute - self.observable_minute

        while True:
            assert not earlier_time < 0, 'earlier time smaller than 0!'
            cut_df = df[(minute_seq > earlier_time) & (minute_seq < later_time)]
            if len(cut_df) > 0:
                return cut_df.reset_index(drop=True)

            later_time -= self.offset_minute
            earlier_time -= self.offset_minute

    def get_df_size(self, df_key):
        df = get_df_from_zip_file(self.zip_file, df_key)
        minute_seq = df[self.idx_key][::-1]
        size = 0
        later_time = minute_seq.tail(1).to_list().pop()
        earlier_time = later_time - self.observable_minute
        while True:
            if earlier_time < 0:
                break
            if len(minute_seq[(minute_seq < later_time) & (minute_seq > earlier_time)]) > 0:
                size += 1
            later_time -= self.offset_minute
            earlier_time -= self.offset_minute
        return size


#################### model ###########################


class BaseModelV4(nn.Module):
    """
    1. cross product of features (embedding of category feature with normalized non_embedded feature)
    2. GRU
    3. Dense classifier
    4. Cross Entropy Loss
    """

    def __init__(self, num_node, num_hour, dropout=0.00):
        super(self.__class__, self).__init__()

        hidden_size = 256
        rnn_num_layer = 2
        num_eta_slots = 16
        clf_dim = 128
        node_embed_dim = 100
        hour_embed_dim = 10
        total_embed_dim = node_embed_dim + hour_embed_dim

        self.node_embed = nn.Embedding(num_node + 1, node_embed_dim)
        self.hour_embed = nn.Embedding(num_hour + 1, hour_embed_dim)

        self.rnn = nn.GRU(total_embed_dim, hidden_size, rnn_num_layer, dropout=dropout)
        self.fc0 = nn.Linear(self.rnn.hidden_size, clf_dim)
        self.ac = nn.SELU()
        self.fc1 = nn.Linear(clf_dim, num_eta_slots)

        util.add_util(self.__class__, self, save_pth='check_points')
        util.add_checkpoint_util(self.__class__, self, auto_checkpoint_on=False, mode='descend')
        util.get_nn_params(self, self.__class__.__name__, True)

    def forward(self, x: torch.Tensor, seq_lengths=None):
        embeds = torch.cat([
            self.node_embed(x.select(-1, 0)),
            self.hour_embed(x.select(-1, 1)),
        ], dim=-1)

        if seq_lengths is None:
            out = embeds.unsqueeze(0).permute(1, 0, 2)
            out, hn = self.rnn(out)
            out = self.ac(self.fc0(hn[-1]))
            out = self.fc1(out)
            log_probs = torch.log_softmax(out, dim=-1)
            return log_probs.squeeze()

        packed_input = nn.utils.rnn.pack_padded_sequence(embeds, seq_lengths, batch_first=False)
        out, hn = self.rnn(packed_input)
        out = self.ac(self.fc0(hn[-1]))
        out = self.fc1(out)
        log_probs = torch.log_softmax(out, dim=-1)
        return log_probs


class BaseModelV4a(nn.Module):
    """
    1. cross product of features (embedding of category feature with normalized non_embedded feature)
    2. GRU
    3. Dense classifier
    4. Cross Entropy Loss
    """

    def __init__(self, num_node, num_class, dropout=0.00):
        super(self.__class__, self).__init__()

        hidden_size = 256
        rnn_num_layer = 2
        clf_dim = 128
        node_embed_dim = 128

        self.node_embed = nn.Embedding(num_node + 1, node_embed_dim)

        self.rnn = nn.GRU(node_embed_dim + 2, hidden_size, rnn_num_layer, dropout=dropout)
        self.fc0 = nn.Linear(self.rnn.hidden_size, clf_dim)
        self.ac = nn.LeakyReLU()
        self.fc1 = nn.Linear(clf_dim, num_class)

        add_util(self.__class__, self, save_pth='check_points')
        add_checkpoint_util(self.__class__, self, auto_checkpoint_on=False, mode='descend')
        get_nn_params(self, self.__class__.__name__, True)

    def forward(self, x: torch.Tensor, x_t: torch.Tensor, seq_lengths=None):
        embeds = torch.cat([
            self.node_embed(x),
            x_t
        ], dim=-1)

        if seq_lengths is None:
            out = embeds.unsqueeze(0).permute(1, 0, 2)
            out, hn = self.rnn(out)
            out = self.ac(self.fc0(hn[-1]))
            out = self.fc1(out)
            log_probs = torch.log_softmax(out, dim=-1)
            return log_probs.squeeze()

        packed_input = nn.utils.rnn.pack_padded_sequence(embeds, seq_lengths, batch_first=False)
        out, hn = self.rnn(packed_input)
        out = self.ac(self.fc0(hn[-1]))
        out = self.fc1(out)
        log_probs = torch.log_softmax(out, dim=-1)
        return log_probs


class BaseModelV4b(nn.Module):
    """
    1. cross product of features (embedding of category feature with normalized non_embedded feature)
    2. GRU
    3. Dense classifier
    4. Cross Entropy Loss
    """

    def __init__(self, num_node, dropout=0.00):
        super(self.__class__, self).__init__()

        hidden_size = 256
        rnn_num_layer = 2
        reg_dim = 128
        node_embed_dim = 128

        self.node_embed = nn.Embedding(num_node, node_embed_dim)

        self.rnn = nn.GRU(node_embed_dim + 2, hidden_size, rnn_num_layer, dropout=dropout)
        self.fc0 = nn.Linear(self.rnn.hidden_size, reg_dim)
        self.ac = nn.LeakyReLU()
        self.fc1 = nn.Linear(reg_dim, 1)

        add_util(self.__class__, self, save_pth='check_points')
        add_checkpoint_util(self.__class__, self, auto_checkpoint_on=False, mode='descend')
        get_nn_params(self, self.__class__.__name__, True)

    def forward(self, x: torch.Tensor, x_t: torch.Tensor, seq_lengths=None):
        embeds = torch.cat([
            self.node_embed(x),
            x_t
        ], dim=-1)

        if seq_lengths is None:
            out = embeds.unsqueeze(0).permute(1, 0, 2)
            out, hn = self.rnn(out)
            out = self.ac(self.fc0(hn[-1]))
            out = self.fc1(out)
            return out.squeeze()

        packed_input = nn.utils.rnn.pack_padded_sequence(embeds, seq_lengths, batch_first=False)
        out, hn = self.rnn(packed_input)
        out = self.ac(self.fc0(hn[-1]))
        out = self.fc1(out)
        return out.squeeze()


# ################## attention ###############################
class AttentionClassifier(nn.Module):
    def __init__(self, pre_train_model: RNNPre, num_eta_slots):
        super(self.__class__, self).__init__()

        clf_dim = 168
        self.num_head = 8

        self.node_embed = copy.deepcopy(pre_train_model.node_embed)
        self.rnn = copy.deepcopy(pre_train_model.rnn)

        self.attn = Attention(pre_train_model.node_embed.embedding_dim * 2,
                              pre_train_model.node_embed.embedding_dim, self.num_head)

        self.fc0 = nn.Linear(pre_train_model.rnn.hidden_size * (self.num_head + 1), clf_dim)
        self.ac = nn.SELU()
        self.fc1 = nn.Linear(self.fc0.out_features, num_eta_slots)

        add_util(self.__class__, self, save_pth='check_points')
        add_checkpoint_util(self.__class__, self, auto_checkpoint_on=False, mode='descend')
        get_nn_params(self, self.__class__.__name__, True)

    def forward(self, node: torch.Tensor, ts: torch.Tensor, seq_lengths=None):
        node_embed = self.node_embed(node)
        embeds = torch.cat([node_embed, ts], dim=-1)

        # if seq_lengths is None:
        #     out, hn = self.rnn(embeds.unsqueeze(1))
        #     out = self.ac(self.fc0(hn[-1]))
        #     out = self.fc1(out).squeeze()
        #     log_probs = torch.log_softmax(out, dim=-1)
        #     return log_probs

        # ------ attention --------
        embed_len = node_embed.size(0)
        a = node_embed.repeat(embed_len, *[1] * node_embed.dim())  # 102, 102, 32, 128
        b = a.transpose(0, 1)

        ...

        attn_weight = self.attn(torch.cat([a, b], -1))  # 102, 102, 32, 8
        attn_weight = attn_weight.sum(1)  # 102, 32, 8

        mask = torch.ones_like(attn_weight)
        for i, seq_length in enumerate(seq_lengths):
            mask[seq_length:, i] = 0

        attn_weight = attn_weight * mask
        attn_weight = attn_weight.softmax(0)

        packed_input = nn.utils.rnn.pack_padded_sequence(embeds, seq_lengths, batch_first=False)
        out, hn = self.rnn(packed_input)
        out, input_sizes = nn.utils.rnn.pad_packed_sequence(out, batch_first=False)  # out: 102, 32, 168
        out = out.unsqueeze(2).expand(-1, -1, self.num_head, -1)  # 102, 32, 8, 168

        out = attn_weight.unsqueeze(-1) * out
        out = out.sum(0)
        out = out.view(out.size(0), -1)

        out = self.fc0(torch.cat([out, hn[-1]], dim=-1))
        out = self.ac(out)
        out = self.fc1(out)
        log_probs = torch.log_softmax(out, dim=-1)
        return log_probs


class Attention(nn.Module):

    def __init__(self, in_size, h_size, num_head=8):
        super(self.__class__, self).__init__()

        self.fc0 = nn.Linear(in_size, h_size)
        self.ac = nn.ReLU()
        self.fc1 = nn.Linear(h_size, num_head)

    def forward(self, x):
        x = self.fc0(x)
        x = self.ac(x)
        x = self.fc1(x)
        return x


class TrajectoryDataZipV6(Dataset):
    def __init__(
            self, zip_path: str, target_key='eta_slot',
            feature_keys=('node', 't_of_day_sin', 't_of_day_cos'),
            max_node_seq_len=500, min_minute=60, offset_minute=10
    ):
        self.zip_path = zip_path
        self.max_node_seq_len = max_node_seq_len
        self.min_minute = min_minute
        self.offset_minute = offset_minute

        print(f'setting up access logic ... ', end='')
        self.zip_file = ZipFile(zip_path)
        self.df_keys = self.zip_file.namelist()
        self.df_sizes = [self.get_df_size(df_key) for df_key in tqdm(self.df_keys)]
        self._partitions = np.cumsum(self.df_sizes).tolist()
        self.size = sum(self.df_sizes)
        print(f'found {self.size} trajectories.')

        self.target_key = target_key
        self.feature_keys = list(feature_keys)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        df_idx = self.get_df_idx(idx)
        df = self.get_df(df_idx)
        df = self.random_peek(df)
        df = df.iloc[:self.max_node_seq_len]

        node = torch.tensor(df['node'], dtype=torch.long)
        t_of_day = torch.tensor(df[['t_of_day_sin', 't_of_day_cos']].to_numpy(), dtype=torch.float)
        eta_slot = torch.tensor(df.tail(1)['eta_slot'].to_numpy()[0], dtype=torch.long)

        return node, t_of_day, eta_slot

    def get_loader(self, batch_size=32, shuffle=True, num_workers=0):
        return DataLoader(self, batch_size, shuffle=shuffle, collate_fn=self.collat_fn, num_workers=num_workers)

    @staticmethod
    def collat_fn(data):
        # packing varying length of sequences
        # https://github.com/HarshTrivedi/packing-unpacking-pytorch-minimal-tutorial

        x_nodes = [x for x, t, y in data]
        x_ts = [t for x, t, y in data]
        ys = torch.stack([y for x, t, y in data])

        padded_x_nodes = pad_sequence(x_nodes)
        padded_x_ts = pad_sequence(x_ts)

        seq_lengths = torch.tensor(list(map(len, x_nodes)))
        seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)

        sorted_padded_x_nodes = padded_x_nodes[:, perm_idx]
        sorted_padded_x_ts = padded_x_ts[:, perm_idx]
        sorted_ys = ys[perm_idx]

        return sorted_padded_x_nodes, sorted_padded_x_ts, sorted_ys, seq_lengths

    def get_df(self, df_idx):
        df_key = self.df_keys[df_idx]
        return get_df_from_zip_file(self.zip_file, df_key)

    def random_peek(self, df: pd.DataFrame):
        """peek at a trajectory at random point with a maximum peek length, earliest point is second row"""

        etas = df['eta_slot'].unique().astype(int).tolist()
        conditioned_etas = [i for i in etas if i <= (max(etas) - self.min_minute)] + [min(etas)]
        eta = random.choice(conditioned_etas)

        idx = random.choice(df.index[df['eta_slot'] == eta])
        df = df.iloc[:idx]

        return drop_duplicate_node(df).reset_index(drop=True)

    def get_df_idx(self, idx):
        return next(part_idx for part_idx, part_right in enumerate(self._partitions) if part_right > idx)

    def get_df_size(self, df_key):
        df = get_df_from_zip_file(self.zip_file, df_key)
        length = (df['eta_slot'].head(1).to_list().pop() - self.min_minute) // 10
        return max(1, int(length))


#################### data set ########################
class TrajectoryDataZipV7(Dataset):
    """works on v5 v6 data"""

    def __init__(
            self, zip_path: str, num_node, observable_minute=60, offset_minute=15,
            target_key='eta_slot',
            idx_key='eta_slot',
            feature_keys=('node', 't_of_day_sin', 't_of_day_cos'),
    ):
        self.observable_minute = observable_minute
        self.offset_minute = offset_minute
        self.zip_path = zip_path
        self.num_node = num_node
        self.idx_key = idx_key

        print(f'making ZipFile from {zip_path}... ', end='')
        self.zip_file = ZipFile(zip_path)
        print(f'done.')

        print(f'setting up access logic ... ', end='')
        self.df_keys = self.zip_file.namelist()
        self.num_dfs = len(self.df_keys)
        print(f'found {self.num_dfs} trajectories.')

        print(f'counting data set size ... ')
        self.df_sizes = []
        for df_key in tqdm(self.df_keys):
            self.df_sizes.append(self.get_df_size(df_key))
        self._partitions = np.cumsum(self.df_sizes).tolist()
        self.size = sum(self.df_sizes)
        print(f'data set size is {self.size}', flush=True)

        self.target_key = target_key
        self.feature_keys = list(feature_keys)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):

        df_idx = self.get_df_idx(idx)
        df = self.get_df(df_idx)

        offset = self.get_offset(idx)
        cut_df, half_cut_df = self.cut_df_by_offset(df, offset)

        node = torch.tensor(cut_df['node'], dtype=torch.long)
        t_of_day = torch.tensor(cut_df[['t_of_day_sin', 't_of_day_cos']].to_numpy(), dtype=torch.float)
        eta_slot = torch.tensor(cut_df.tail(1)['eta_slot'].to_numpy()[0], dtype=torch.long)

        c = Counter(half_cut_df.node)
        spectro = torch.zeros(self.num_node)
        spectro[list(c.keys())] = torch.tensor(list(c.values()), dtype=torch.float)

        return node, t_of_day, spectro, eta_slot

    def get_loader(self, batch_size=32, shuffle=True):
        return DataLoader(self, batch_size, shuffle=shuffle, collate_fn=self.collat_fn)

    @staticmethod
    def collat_fn(data):
        # packing varying length of sequences
        # https://github.com/HarshTrivedi/packing-unpacking-pytorch-minimal-tutorial

        x_nodes = [x for x, t, s, y in data]
        x_ts = [t for x, t, s, y in data]
        x_sptrs = torch.stack([s for x, t, s, y in data])
        ys = torch.stack([y for x, t, s, y in data])

        padded_x_nodes = pad_sequence(x_nodes)
        padded_x_ts = pad_sequence(x_ts)

        seq_lengths = torch.tensor(list(map(len, x_nodes)))
        seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)

        sorted_padded_x_nodes = padded_x_nodes[:, perm_idx]
        sorted_padded_x_ts = padded_x_ts[:, perm_idx]
        sorted_x_sptrs = x_sptrs[perm_idx]
        sorted_ys = ys[perm_idx]

        return sorted_padded_x_nodes, sorted_padded_x_ts, sorted_x_sptrs, sorted_ys, seq_lengths

    def get_offset(self, idx):
        df_idx = self.get_df_idx(idx)
        offset = idx - ([0] + self._partitions)[df_idx]
        return offset

    def get_df_idx(self, idx):
        return next(part_idx for part_idx, part_right in enumerate(self._partitions) if part_right > idx)

    def get_df(self, df_idx):
        df_key = self.df_keys[df_idx]
        return get_df_from_zip_file(self.zip_file, df_key)

    def cut_df_by_offset(self, df: pd.DataFrame, offset: int):
        # offset start from backward
        minute_seq = df[self.idx_key][::-1].reset_index(drop=True)
        last_minute = minute_seq.tail(1).to_list().pop()
        later_time = last_minute - offset * self.offset_minute
        earlier_time = last_minute - offset * self.offset_minute - self.observable_minute

        while True:
            assert not earlier_time < 0, 'earlier time smaller than 0!'
            cut_df = df[(minute_seq > earlier_time) & (minute_seq < later_time)]
            if len(cut_df) > 0:
                return cut_df.reset_index(drop=True), df[minute_seq < later_time].reset_index(drop=True)

            later_time -= self.offset_minute
            earlier_time -= self.offset_minute

    def get_df_size(self, df_key):
        df = get_df_from_zip_file(self.zip_file, df_key)
        minute_seq = df[self.idx_key][::-1]
        size = 0
        later_time = minute_seq.tail(1).to_list().pop()
        earlier_time = later_time - self.observable_minute
        while True:
            if earlier_time < 0:
                break
            if len(minute_seq[(minute_seq < later_time) & (minute_seq > earlier_time)]) > 0:
                size += 1
            later_time -= self.offset_minute
            earlier_time -= self.offset_minute
        return size


#################### model ###########################


class BaseModelV7(nn.Module):
    """
    1. cross product of features (embedding of category feature with normalized non_embedded feature)
    2. GRU
    3. Dense classifier
    4. Cross Entropy Loss
    """

    def __init__(self, num_node, num_class, rnn_type='lstm', dropout=0.00):
        super(self.__class__, self).__init__()

        hidden_size = 256
        rnn_num_layer = 2
        clf_dim = 256
        node_embed_dim = 128

        self.node_embed = nn.Embedding(num_node + 1, node_embed_dim)
        self.rnn = nn.LSTM(node_embed_dim + 2, hidden_size, rnn_num_layer, dropout=dropout) if rnn_type == 'lstm' \
            else nn.GRU(node_embed_dim + 2, hidden_size, rnn_num_layer, dropout=dropout)

        self.sptr_fc0 = nn.Linear(num_node, num_node // 4)
        self.sptr_fc1 = nn.Linear(num_node // 4, num_node // 8)
        self.sptr_fc2 = nn.Linear(num_node // 8, num_node // 16)

        self.fc0 = nn.Linear(self.rnn.hidden_size + self.sptr_fc2.out_features, clf_dim)
        # self.ac = nn.LeakyReLU()
        self.ac = Selu()
        self.fc1 = nn.Linear(clf_dim, num_class)

        add_util(self.__class__, self, save_pth='check_points')
        add_checkpoint_util(self.__class__, self, auto_checkpoint_on=False, mode='descend')
        get_nn_params(self, self.__class__.__name__, True)

    def forward(self, x_node: torch.Tensor, x_t: torch.Tensor, x_spctr: torch.Tensor, seq_lengths=None):
        embeds = torch.cat([
            self.node_embed(x_node),
            x_t
        ], dim=-1)

        if seq_lengths is None:
            out = embeds.unsqueeze(0).permute(1, 0, 2)
            out, hn = self.rnn(out)
            out = self.ac(self.fc0(hn[-1]))
            out = self.fc1(out)
            log_probs = torch.log_softmax(out, dim=-1)
            return log_probs.squeeze()

        packed_input = nn.utils.rnn.pack_padded_sequence(embeds, seq_lengths, batch_first=False)
        if isinstance(self.rnn, nn.GRU):
            out, hn = self.rnn(packed_input)
        elif isinstance(self.rnn, nn.LSTM):
            out, (hn, cn) = self.rnn(packed_input)
        else:
            raise Exception(f'unknown rnn type!')

        spec_out = self.ac(self.sptr_fc0(x_spctr))
        spec_out = self.ac(self.sptr_fc1(spec_out))
        spec_out = self.ac(self.sptr_fc2(spec_out))

        out = torch.cat([hn[-1], spec_out], dim=-1)
        out = self.ac(self.fc0(out))
        out = self.fc1(out)
        log_probs = torch.log_softmax(out, dim=-1)
        return log_probs


class Selu(nn.Module):
    def __init__(self):
        super(Selu, self).__init__()
        self.alpha = 1.6732632423543772848170429916717
        self.scale = 1.0507009873554804934193349852946

    def forward(self, x):
        temp1 = self.scale * nn.functional.relu(x)
        temp2 = self.scale * self.alpha * (nn.functional.elu(-1 * nn.functional.relu(-1 * x)))
        return temp1 + temp2


class TrajectoryDataZipV8(Dataset):
    """works on v8 data, with idx caching """

    def __init__(
            self, zip_path: str, num_node, observable_minute=60, offset_minute=15,
            target_key='eta_minute',
            idx_key='minute_of_trj',
            feature_keys=('node_x',),
    ):
        self.observable_minute = observable_minute
        self.offset_minute = offset_minute
        self.zip_path = zip_path
        self.num_node = num_node
        self.idx_key = idx_key
        dirname, basename = os.path.dirname(zip_path), os.path.basename(zip_path)
        cache_base_name = basename.split('.')[0] + '_idx_cache.json'
        self.idx_cache_pth = os.path.join(dirname, cache_base_name)

        print(f'making ZipFile from {zip_path}... ', end='')
        self.zip_file = ZipFile(zip_path)
        print(f'done.')

        print(f'setting up access logic ... ', end='')
        self.df_keys = self.zip_file.namelist()
        self.num_dfs = len(self.df_keys)
        print(f'found {self.num_dfs} trajectories.')

        if os.path.exists(self.idx_cache_pth):
            print(f'found idx cache ... ')
            with open(self.idx_cache_pth) as f:
                idx_cache = json.load(f)
            if idx_cache['offset_minute'] == self.offset_minute:
                self.df_sizes = idx_cache['sizes']
            else:
                self.count_and_cache_sizes()
        else:
            self.df_sizes = []
            self.count_and_cache_sizes()
            # for df_key in tqdm(self.df_keys):
            #     self.df_sizes.append(self.get_df_size(df_key))
            # with open(self.idx_cache_pth, 'w') as f:
            #     json.dump(self.df_sizes, f)

        self._partitions = np.cumsum(self.df_sizes).tolist()
        self.size = sum(self.df_sizes)
        print(f'data set size is {self.size}', flush=True)

        self.target_key = target_key
        self.feature_keys = list(feature_keys)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):

        df_idx = self.get_df_idx(idx)
        df = self.get_df(df_idx)

        offset = self.get_offset(idx)
        cut_df, half_cut_df = self.cut_df_by_offset(df, offset)

        x = torch.tensor(cut_df[list(self.feature_keys)].to_numpy(), dtype=torch.long)
        eta = torch.tensor(cut_df.tail(1)['eta_minute'].to_numpy()[0], dtype=torch.long)

        c = Counter(half_cut_df[self.feature_keys[0]])  # node or node_x
        spectro = torch.zeros(self.num_node)
        spectro[list(c.keys())] = torch.tensor(list(c.values()), dtype=torch.float)

        return x, spectro, eta

    def get_loader(self, batch_size=32, shuffle=True):
        return DataLoader(self, batch_size, shuffle=shuffle, collate_fn=self.collat_fn)

    @staticmethod
    def collat_fn(data):
        # packing varying length of sequences
        # https://github.com/HarshTrivedi/packing-unpacking-pytorch-minimal-tutorial

        xs = [x for x, s, y in data]
        x_sptrs = torch.stack([s for x, s, y in data])
        ys = torch.stack([y for x, s, y in data])

        padded_xs = pad_sequence(xs)

        seq_lengths = torch.tensor(list(map(len, xs)))
        seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)

        sorted_padded_xs = padded_xs[:, perm_idx]
        sorted_x_sptrs = x_sptrs[perm_idx]
        sorted_ys = ys[perm_idx]

        return sorted_padded_xs, sorted_x_sptrs, sorted_ys, seq_lengths

    def get_offset(self, idx):
        df_idx = self.get_df_idx(idx)
        offset = idx - ([0] + self._partitions)[df_idx]
        return offset

    def get_df_idx(self, idx):
        return next(part_idx for part_idx, part_right in enumerate(self._partitions) if part_right > idx)

    def get_df(self, df_idx):
        df_key = self.df_keys[df_idx]
        return get_df_from_zip_file(self.zip_file, df_key)

    def cut_df_by_offset(self, df: pd.DataFrame, offset: int):
        # offset start from backward
        minute_seq = df[self.idx_key].reset_index(drop=True)
        last_minute = minute_seq.tail(1).to_list().pop()
        later_time = last_minute - offset * self.offset_minute
        earlier_time = last_minute - offset * self.offset_minute - self.observable_minute

        while True:
            assert not earlier_time < 0, 'earlier time smaller than 0!'
            cut_df = df[(minute_seq > earlier_time) & (minute_seq < later_time)]
            if len(cut_df) > 0:
                return cut_df.reset_index(drop=True), df[minute_seq < later_time].reset_index(drop=True)

            later_time -= self.offset_minute
            earlier_time -= self.offset_minute

    def get_df_size(self, df_key):
        df = get_df_from_zip_file(self.zip_file, df_key)
        minute_seq = df[self.idx_key]
        size = 0
        later_time = minute_seq.tail(1).to_list().pop()
        earlier_time = later_time - self.observable_minute
        while True:
            if earlier_time < 0:
                break
            if len(minute_seq[(minute_seq < later_time) & (minute_seq > earlier_time)]) > 0:
                size += 1
            later_time -= self.offset_minute
            earlier_time -= self.offset_minute
        return size

    def count_and_cache_sizes(self):
        print(f'counting data set size ... ')
        self.df_sizes = []
        for df_key in tqdm(self.df_keys):
            self.df_sizes.append(self.get_df_size(df_key))
        with open(self.idx_cache_pth, 'w') as f:
            json.dump({'sizes': self.df_sizes, 'offset_minute': self.offset_minute}, f)
        print(f'saved idx cache to {self.idx_cache_pth} ... ')


#################### model ###########################


class BaseModelV8(nn.Module):
    """
    1. embedding of category feature
    2. GRU
    2a. hist vector
    3. Dense classifier
    4. Cross Entropy Loss
    """

    def __init__(self, num_dim_pairs, num_class, dropout=0.00):
        super(self.__class__, self).__init__()

        hidden_size = 256
        rnn_num_layer = 2
        clf_dim = 256
        num_node = num_dim_pairs[0][0]
        # node_embed_dim = 128
        # self.node_embed = nn.Embedding(num_node, node_embed_dim)
        self.embeddings = [nn.Embedding(num_code, num_embed_dim) for num_code, num_embed_dim in num_dim_pairs]
        for i, embedding in enumerate(self.embeddings):
            setattr(self, f'embedding{i}', embedding)

        self.embeds_out_size = sum([num_embed_dim for num_code, num_embed_dim in num_dim_pairs])

        self.rnn = nn.GRU(self.embeds_out_size, hidden_size, rnn_num_layer, dropout=dropout)

        self.sptr_fc0 = nn.Linear(num_node, num_node // 4)
        self.sptr_fc1 = nn.Linear(num_node // 4, num_node // 8)
        self.sptr_fc2 = nn.Linear(num_node // 8, num_node // 16)

        self.fc0 = nn.Linear(self.rnn.hidden_size + self.sptr_fc2.out_features,
                             clf_dim)
        self.ac = Selu()
        self.fc1 = nn.Linear(clf_dim, num_class)

        add_util(self.__class__, self, save_pth='check_points')
        add_checkpoint_util(self.__class__, self, auto_checkpoint_on=False, mode='descend')
        get_nn_params(self, self.__class__.__name__, True)

    def forward(self, xs: torch.Tensor, x_spctr: torch.Tensor, seq_lengths=None):
        embeds = torch.cat([embedding(
            xs.select(-1, i)
        ) for i, embedding in enumerate(self.embeddings)], dim=-1)

        packed_input = nn.utils.rnn.pack_padded_sequence(embeds, seq_lengths, batch_first=False)
        out, hn = self.rnn(packed_input)

        spctr_out = self.ac(self.sptr_fc0(x_spctr))
        spctr_out = self.ac(self.sptr_fc1(spctr_out))
        spctr_out = self.ac(self.sptr_fc2(spctr_out))

        out = torch.cat([hn[-1], spctr_out], dim=-1)
        # out = self.ac(self.fc0(hn[-1]))
        out = self.ac(self.fc0(out))
        out = self.fc1(out)
        log_probs = torch.log_softmax(out, dim=-1)
        return log_probs


class TrajectoryDataZipV9(Dataset):
    """works on v8 data, with idx caching """

    def __init__(
            self, zip_path: str, num_node, observable_minute=60, offset_minute=15,
            max_len=5000,
            target_key='eta_minute',
            idx_key='minute_of_trj',
            feature_keys=('node_x',),
    ):
        self.observable_minute = observable_minute
        self.offset_minute = offset_minute
        self.max_len = max_len
        self.zip_path = zip_path
        self.num_node = num_node
        self.idx_key = idx_key
        dirname, basename = os.path.dirname(zip_path), os.path.basename(zip_path)
        cache_base_name = basename.split('.')[0] + '_idx_cache.json'
        self.idx_cache_pth = os.path.join(dirname, cache_base_name)

        print(f'making ZipFile from {zip_path}... ', end='')
        self.zip_file = ZipFile(zip_path)
        print(f'done.')

        print(f'setting up access logic ... ', end='')
        self.df_keys = self.zip_file.namelist()
        self.num_dfs = len(self.df_keys)
        print(f'found {self.num_dfs} trajectories.')

        if os.path.exists(self.idx_cache_pth):
            print(f'found idx cache ... ')
            with open(self.idx_cache_pth) as f:
                idx_cache = json.load(f)
            if idx_cache['offset_minute'] == self.offset_minute:
                self.df_sizes = idx_cache['sizes']
            else:
                self.count_and_cache_sizes()
        else:
            self.df_sizes = []
            self.count_and_cache_sizes()
            # for df_key in tqdm(self.df_keys):
            #     self.df_sizes.append(self.get_df_size(df_key))
            # with open(self.idx_cache_pth, 'w') as f:
            #     json.dump(self.df_sizes, f)

        self._partitions = np.cumsum(self.df_sizes).tolist()
        self.size = sum(self.df_sizes)
        print(f'data set size is {self.size}', flush=True)

        self.target_key = target_key
        self.feature_keys = list(feature_keys)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):

        df_idx = self.get_df_idx(idx)
        df = self.get_df(df_idx)

        offset = self.get_offset(idx)
        cut_df, half_cut_df = self.cut_df_by_offset(df, offset)

        half_cut_df = half_cut_df.groupby(['minute_of_trj']).tail(1)  # per minute node if avail

        x = torch.tensor(half_cut_df[list(self.feature_keys)].tail(self.max_len).to_numpy(), dtype=torch.long)
        eta = torch.tensor(half_cut_df.tail(1)['eta_minute'].to_numpy()[0], dtype=torch.long)

        return x, eta

    def get_loader(self, batch_size=32, shuffle=True):
        return DataLoader(self, batch_size, shuffle=shuffle, collate_fn=self.collat_fn)

    @staticmethod
    def collat_fn(data):
        # packing varying length of sequences
        # https://github.com/HarshTrivedi/packing-unpacking-pytorch-minimal-tutorial

        xs = [x for x, y in data]
        ys = torch.stack([y for x, y in data])

        padded_xs = pad_sequence(xs, batch_first=True)

        seq_lengths = torch.tensor(list(map(len, xs)))

        return padded_xs, ys, seq_lengths

    def get_offset(self, idx):
        df_idx = self.get_df_idx(idx)
        offset = idx - ([0] + self._partitions)[df_idx]
        return offset

    def get_df_idx(self, idx):
        return next(part_idx for part_idx, part_right in enumerate(self._partitions) if part_right > idx)

    def get_df(self, df_idx):
        df_key = self.df_keys[df_idx]
        return get_df_from_zip_file(self.zip_file, df_key)

    def cut_df_by_offset(self, df: pd.DataFrame, offset: int):
        # offset start from backward
        minute_seq = df[self.idx_key].reset_index(drop=True)
        last_minute = minute_seq.tail(1).to_list().pop()
        later_time = last_minute - offset * self.offset_minute
        earlier_time = last_minute - offset * self.offset_minute - self.observable_minute

        while True:
            assert not earlier_time < 0, 'earlier time smaller than 0!'
            cut_df = df[(minute_seq > earlier_time) & (minute_seq < later_time)]
            if len(cut_df) > 0:
                return cut_df.reset_index(drop=True), df[minute_seq < later_time].reset_index(drop=True)

            later_time -= self.offset_minute
            earlier_time -= self.offset_minute

    def get_df_size(self, df_key):
        df = get_df_from_zip_file(self.zip_file, df_key)
        minute_seq = df[self.idx_key]
        size = 0
        later_time = minute_seq.tail(1).to_list().pop()
        earlier_time = later_time - self.observable_minute
        while True:
            if earlier_time < 0:
                break
            if len(minute_seq[(minute_seq < later_time) & (minute_seq > earlier_time)]) > 0:
                size += 1
            later_time -= self.offset_minute
            earlier_time -= self.offset_minute
        return size

    def count_and_cache_sizes(self):
        print(f'counting data set size ... ')
        self.df_sizes = []
        for df_key in tqdm(self.df_keys):
            self.df_sizes.append(self.get_df_size(df_key))
        print(f'saving idx cache to {self.idx_cache_pth} ... ', end='')
        with open(self.idx_cache_pth, 'w') as f:
            json.dump({'sizes': self.df_sizes, 'offset_minute': self.offset_minute}, f)
        print('done!')
