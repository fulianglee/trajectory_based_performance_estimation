import os
import numpy as np
import torch
import util, report_util, train_util

# ################### spec ############################
dataset_dir = 'dataset'
train_data_file_name = 'v5_data_n10.zip'
train_data_zip_pth = os.path.join(dataset_dir, train_data_file_name)

assert torch.cuda.is_available(), f'cuda not available'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# ---------------- train spec -----------------------
num_node = 4267  # from voronoi partition
epoch_size = 3
batch_size = 64
max_node_trj_len = 2000
offset_minute = 9
trj_min_minute = 60
norm_factor = 800

# ---------------- test spec ------------------------
test_size = 64
num_test = 40
aso_data_zip_name = 'aso_data_v5_n10.zip'
dso_data_zip_name = 'dso_data_v5_n10.zip'


def get_loader(dataset_dir, zip_name, test_size):
    test_dataset = train_util.TrajectoryDataZip(os.path.join(dataset_dir, zip_name))
    return test_dataset.get_loader(batch_size=test_size, shuffle=True)


# ################### train loop ######################
# ---------------- run script -----------------------

loader = train_util.TrajectoryDataZipV4b(train_data_zip_pth, offset_minute=offset_minute,
                                         y_norm_factor=norm_factor).get_loader(batch_size=batch_size, shuffle=True)
model = train_util.BaseModelV4b(num_node, dropout=0.01).to(device)
avm = util.AverageMeter()
criterion = torch.nn.MSELoss()
for j in range(epoch_size):

    # train
    model.train()
    for i, data in enumerate(loader):
        x_nodes, x_ts, y, seq_len = [d.to(device) for d in data]
        if len(x_nodes) > max_node_trj_len:
            print(f'len(x_nodes)={len(x_nodes)}', end='\r')
            continue
        y_pred = model(x_nodes, x_ts, seq_len)
        loss = criterion(y_pred, y)
        with model.optimize_c():
            loss.backward()

        avm.log(float(loss))
        print(f'batch {i}/{len(loader.dataset) // loader.batch_size}, instant loss {loss.item():.4f}', end='\r')
        del x_nodes, x_ts, y, seq_len, y_pred, loss

    print(f'epoch {j + 1}/{epoch_size}, average loss is {avm.value:.4f}')

    # test - training set

    model.eval()
    for set_name, test_loader in ((s_name, t_loader()) for s_name, t_loader in [
        ('Train set', lambda: loader.dataset.get_loader(batch_size=test_size, shuffle=True)),
        ('Aso set', lambda: get_loader(dataset_dir, aso_data_zip_name, test_size)),
        ('Dso set', lambda: get_loader(dataset_dir, dso_data_zip_name, test_size)),
    ]):
        test_pre_y_agg = np.array([])
        test_y_agg = np.array([])
        for j in range(num_test):
            with torch.no_grad():
                for data in test_loader:
                    test_x_nodes, test_x_ts, test_y, test_seq_len = [d.to(device) for d in data]
                    if len(test_x_nodes) > max_node_trj_len:
                        print(f'len(x_nodes)={len(test_x_nodes)}', end='\r')
                        test_y = test_pre_y = None
                        break
                    test_pred_y = model(test_x_nodes, test_x_ts, test_seq_len).cpu()
                    break

            test_y_agg = test_y_agg if test_pred_y is None else np.append(test_y_agg, test_y.cpu().numpy())
            test_pre_y_agg = test_pre_y_agg if test_pred_y is None \
                else np.append(test_pre_y_agg, test_pred_y.numpy())
        print(f'{set_name} result: '
              f'RMSE={np.sqrt(((test_y_agg - test_pre_y_agg) ** 2).sum() / len(test_y_agg)):.4f}; '
              f'AME={np.abs(test_y_agg - test_pre_y_agg).sum() / len(test_y_agg):.4f}; '
              f'normed RMSE={np.sqrt(((norm_factor * (test_y_agg - test_pre_y_agg)) ** 2).sum() / len(test_y_agg)):.4f}; '
              f'normed AME={np.abs(norm_factor * (test_y_agg - test_pre_y_agg)).sum() / len(test_y_agg):.4f}; ')
    del test_x_nodes, test_x_ts, test_y, test_pred_y, test_y_agg, test_pre_y_agg

    # checkpoint
    model.save('BaseModel11062020V4b.pt')
