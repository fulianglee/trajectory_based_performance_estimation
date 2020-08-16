import os
import torch
import train_util, report_util
from util import AverageMeter, SmoothNLLLoss
import numpy as np

#################### spec ############################
assert torch.cuda.is_available(), f'cuda not available'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


categorical_feature = ['node_edge', 'hour_of_day']
numerical_feature = ['gps_long', 'gps_lat', 'gps_speed', 'mileage', 'direction', 'second_diff']

stop_threshold_minute = 5

# ---------------- train spec -----------------------
num_node = 3416  # from qtreee

num_hour = 24
min_duration_minute_threshold = 80
num_sample = 5000
num_test_sample = 500
minute_interval = 20
slot_count = 16
dataset_dir = 'dataset'
train_data_file_name = 'v4_data.zip'
train_data_zip_pth = os.path.join(dataset_dir, train_data_file_name)

batch_size = 256

# ---------------- test spec ------------------------
test_size = 500
num_test = 10
offset_minute = 20
num_of_trj = 500
aso_data_zip_name = 'aso_data_v4.zip'
dso_data_zip_name = 'dso_data_v4.zip'


def get_loader(dataset_dir, zip_name, test_size):
    test_dataset = train_util.TrajectoryDataZipV4(os.path.join(dataset_dir, zip_name))
    return test_dataset.get_loader(batch_size=test_size, shuffle=True)


# ################### train loop ######################
torch.manual_seed(1)

avm = AverageMeter()
epoch_size = 5

loader = train_util.TrajectoryDataZipV4(train_data_zip_pth, offset_minute=7).get_loader(batch_size=batch_size, shuffle=True)

model = train_util.BaseModelV4(num_node, num_hour, dropout=0.01).to(device)
# criterion = nn.NLLLoss()
criterion = SmoothNLLLoss(smoothing=0.01)

for j in range(epoch_size):

    # train
    model.train()
    for i, data in enumerate(loader):
        x_seq, y, seq_lengths = [d.to(device) for d in data]

        if x_seq.size(0) > 3000:
            continue

        log_probs = model(x_seq, seq_lengths)
        loss = criterion(log_probs, y.type(torch.LongTensor))
        with model.optimize_c():
            loss.backward()

        avm.log(loss.tolist())
        print(f'batch {i}/{len(loader.dataset) // loader.batch_size}, instant loss {loss.item():.4f}', end='\r')
    print(f'epoch {j + 1}/{epoch_size}, average loss is {avm.value:.4f}')

    # test - training set
    model.eval()

    for set_name, test_loader in ((s_name, t_loader()) for s_name, t_loader in [
        ('Train set', lambda: loader.dataset.get_loader(batch_size=test_size, shuffle=True)),
        ('Aso set', lambda: get_loader(dataset_dir, aso_data_zip_name, test_size)),
        ('Dso set', lambda: get_loader(dataset_dir, dso_data_zip_name, test_size)),
    ]):
        test_pre_ys = np.array([])
        test_ys = np.array([])
        for j in range(num_test):
            with torch.no_grad():
                for data in test_loader:
                    test_x_seq, test_y, test_seq_lengths = [d.to(device) for d in data]
                    test_log_probs = model(test_x_seq, test_seq_lengths).cpu()
                    break

            test_pre_ys = np.append(test_pre_ys, test_log_probs.argmax(-1).numpy())
            test_ys = np.append(test_ys, test_y.cpu().numpy())
        print(f'{set_name} result: ', report_util.p_errors(test_pre_ys, test_ys, minute_interval))


# check pointing
model.save('BaseModel29052020_2x.pt')


