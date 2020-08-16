"""
goals:
    generate a dictionary to associate number plate to its file paths
    stitching algorithm to patch a trajectory to some specified length and specified end point if possible
    collect 5000 non repeating trajectories and cache them to disk in the form of path dicts
"""
import sys
from tqdm import tqdm
import copy
import random
import pandas as pd
import io
import os
import datetime
import json
from zipfile import ZipFile
from pathlib import Path
import shutil

COL_NAMES = ['map_long', 'map_lat', 'gps_time', 'gps_speed', 'direction', 'event', 'alarm_code',
             'gps_long', 'gps_lat', 'altitude', 'recorder_speed', 'mileage',
             'error_code', 'api_code', 'sys_time']

ZIP_FILE_PATHS_LOCAL = ('sample/track_20190823.zip', 'sample/track_20190824.zip', 'sample/track_20190825.zip',
                        'sample/track_20190826.zip', 'sample/track_20190827.zip', 'sample/track_20190828.zip',
                        'sample/track_20190829.zip')

ZIP_FILE_PATHS = (
    'track_20190823.zip', 'track_20190824.zip', 'track_20190825.zip', 'track_20190826.zip', 'track_20190827.zip',
    'track_20190828.zip', 'track_20190829.zip', 'track_20190830.zip', 'track_20190831.zip', 'track_20190901.zip',
    'track_20190902.zip', 'track_20190903.zip', 'track_20190904.zip', 'track_20190905.zip', 'track_20190906.zip',
    'track_20190907.zip', 'track_20190908.zip', 'track_20190909.zip', 'track_20190910.zip', 'track_20190911.zip')


# ------------------------ number plate dictionary ---------------------
def get_plate_from_path(raw_pth: str):
    return raw_pth.split('/')[-1].split('.')[0]


def make_plate_dict(pths: list):
    plate_dict = dict()
    for pth in pths:
        plate = get_plate_from_path(pth)
        plate_dict[plate] = plate_dict.get(plate, []) + [pth]
    return plate_dict


# ------------------------- stitching algorithm ------------------------
# top params: path_file, loading fn, cutting fn, specified time duration
def get_zip_pth_from_raw_pth(zip_pth_dir, raw_pth):
    pieces = raw_pth.split('/')[:-1]
    zip_name = '_'.join([pieces[0], ''.join(pieces[1:])]) + '.zip'
    if zip_pth_dir is not None:
        return os.path.join(zip_pth_dir, zip_name)
        # return '/'.join([zip_pth_dir, zip_name])
    else:
        return zip_name


def get_date_from_raw_pth(raw_pth):
    return [*map(int, raw_pth.split('/')[1:-1])]


def get_prior_date(year, month, day):
    pre_day = datetime.date(year, month, day) - datetime.timedelta(days=1)
    ymd = [pre_day.year, pre_day.month, pre_day.day]
    return [*map(lambda i: str(i).zfill(2), ymd)]


def get_prior_date_from_raw_pth(raw_pth):
    return get_prior_date(*get_date_from_raw_pth(raw_pth))


def get_prior_raw_pth_from_raw_pth(raw_pth):
    prior_date = get_prior_date_from_raw_pth(raw_pth)
    plate = get_plate_from_path(raw_pth)
    return '/'.join(['track'] + prior_date + [plate + '.txt'])


def find_prior_file(zip_pth_dir, raw_pth, zip_file_dict=None):
    prior_raw_pth = get_prior_raw_pth_from_raw_pth(raw_pth)
    prior_zip_pth = get_zip_pth_from_raw_pth(zip_pth_dir, prior_raw_pth)
    if not os.path.isfile(prior_zip_pth):
        return None
    zip_file = ZipFile(prior_zip_pth) if zip_file_dict is None else zip_file_dict[prior_zip_pth]
    if prior_raw_pth in zip_file.namelist():
        return prior_zip_pth, prior_raw_pth
    else:
        return None


def get_zip_f_dict(zip_pths=ZIP_FILE_PATHS_LOCAL):
    return dict([(pth, ZipFile(pth)) for pth in tqdm(zip_pths)])


def load_fn(zip_dir, raw_pth, zip_f_dict):
    zip_pth = get_zip_pth_from_raw_pth(zip_dir, raw_pth)
    zip_file = zip_f_dict[zip_pth]
    io_bytes = io.BytesIO(zip_file.read(raw_pth))
    return pd.read_csv(io_bytes, ':', index_col=False, names=COL_NAMES)


def get_stitched_df(pth, load_fn, time_dur: datetime.timedelta, trim=True):
    df = load_fn(pth)
    # earliest time is earlier then (latest time - time_dur) return df
    # else find previous df and concat then trim


def get_df_from_raw_pth(raw_pth, load_fn, zip_f_dict=None, zip_dir='sample') -> pd.DataFrame:
    zip_pth = get_zip_pth_from_raw_pth(zip_dir, raw_pth)
    zip_file = ZipFile(zip_pth) if zip_f_dict is None else zip_f_dict[zip_pth]
    io_bytes = io.BytesIO(zip_file.read(raw_pth))
    return load_fn(io_bytes)


def default_load_fn(pth_or_io):
    return pd.read_csv(pth_or_io, ':', index_col=False, names=COL_NAMES)


def make_directory(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def get_file_dir(file_path):
    return file_path[:-file_path[::-1].index('/')]


def make_file_directory(file_path):
    file_dir = get_file_dir(file_path)
    make_directory(file_dir)
    assert os.path.exists(file_dir), f'Could not create {file_dir}!'


def remove_file(file_path, verbose=True):
    Path(file_path).unlink()
    if verbose:
        print(f'{file_path} removed.')


def remove_files(file_paths, verbose=True):
    for file_path in file_paths:
        remove_file(file_path, verbose)


def remove_dir(pth):
    shutil.rmtree(pth)


def save_file_to_zip(zfile_pth_or_zfile, file_path):
    zip_file = zfile_pth_or_zfile if type(zfile_pth_or_zfile) is ZipFile else ZipFile(zfile_pth_or_zfile, 'w')
    zip_file.write(file_path)
    zip_file.close()


def save_files_to_zip(zfile_pth_or_zfile, file_pths: list):
    zip_file = zfile_pth_or_zfile if type(zfile_pth_or_zfile) is ZipFile else ZipFile(zfile_pth_or_zfile, 'w')
    for pth in file_pths:
        zip_file.write(pth)
    zip_file.close()


def extract_store_to_zip(num_sample, data_extractor, dataset_dir='dataset', sub_dir='train', zip_name='dataset.zip'):
    dir = os.path.join(dataset_dir, sub_dir)
    file_pths = []
    with tqdm(total=num_sample) as pbar:
        for i in range(num_sample):
            data = data_extractor.get_named_data()

            if data is None:
                break
            else:
                pth, df = data
                file_pth = os.path.join(dir, pth)
                file_pths.append(file_pth)
                make_file_directory(file_pth)
                df.to_csv(file_pth, index=False)
                pbar.update(1)

    zip_file_pth = os.path.join(dir, zip_name)
    save_files_to_zip(zip_file_pth, file_pths)
    print(f'removing {len(file_pths)} intermediate files')
    remove_files(file_pths)


def extract_to_dir(num_sample, data_extractor, dataset_dir='dataset', sub_dir='train'):
    dir = os.path.join(dataset_dir, sub_dir)
    file_pths = []
    with tqdm(total=num_sample) as pbar:
        for i in range(num_sample):
            data = data_extractor.get_named_data()

            if data is None:
                print(f'data extractor return none, possibly no more data to process.')
                return
            else:
                pth, df = data
                file_pth = os.path.join(dir, pth)
                file_pths.append(file_pth)
                make_file_directory(file_pth)
                df.to_csv(file_pth, index=False)
                pbar.update(1)


def store_to_zip(data_extractor, dataset_dir='dataset', sub_dir='train', zip_name='dataset.zip'):
    dir = os.path.join(dataset_dir, sub_dir)
    file_pths = [os.path.join(dir, pth) for pth in data_extractor.processed_file_pth_lst]
    zip_file_pth = os.path.join(dir, zip_name)

    save_files_to_zip(zip_file_pth, file_pths)
    print(f'removing {len(file_pths)} intermediate files')
    remove_files(file_pths)


def get_file_lst_from_zip(zip_pth, pre_dirs):
    # 'dataset/train/track/2019/08/28/2_%D4%C1BAJ520.txt' => 'track/2019/08/28/2_%D4%C1BAJ520.txt'
    zip_file = ZipFile(zip_pth)
    file_pth_lst = zip_file.namelist()
    # s = 'track/2019/08/28/2_%D4%C1BAJ520.txt'
    off_set = sum([len(pre_dir) for pre_dir in pre_dirs]) + len(pre_dirs)
    return [file_pth[off_set:] for file_pth in file_pth_lst]


def get_df_from_zip_file(zip_file, file_pth) -> pd.DataFrame:
    io_bytes = io.BytesIO(zip_file.read(file_pth))
    return pd.read_csv(io_bytes)


class DataExtractor:
    def __init__(self,
                 file_path_lst: list,
                 pipe_line_fns: list,
                 cache_zip_file=True,
                 zip_file_dict=None,
                 file_path_masks=None,
                 zip_dir='sample'):

        if cache_zip_file:
            if zip_file_dict is not None:
                self.zip_f_dict = zip_file_dict
            else:
                print(f'getting ZipFiles ready ... ', end='')
                self.zip_f_dict = get_zip_f_dict()
                print(f'done!')

        self.file_path_lst = copy.deepcopy(file_path_lst)
        self.processed_file_pth_lst = []
        self.skipped_file_pth_lst = []
        self.current_file_pth = None
        self.zip_dir = zip_dir

        if file_path_masks is not None:
            print(f'removing {len(file_path_masks)} files from candidate pool {len(file_path_lst)}... ', end='')
            self.file_path_lst = [file for file in self.file_path_lst if file not in file_path_masks]
            print(f'now {len(self.file_path_lst)} remained..')

        self.pipe_line_fns = pipe_line_fns

    def get_named_data(self):
        if len(self.file_path_lst) == 0:
            return None

        while True:
            if len(self.file_path_lst) == 0:
                return None
            # pick path and remove path from lst
            file_path = random.choice(self.file_path_lst)
            self.current_file_pth = file_path
            self.file_path_lst.remove(file_path)

            # get df
            df = get_df_from_raw_pth(file_path, default_load_fn,
                                     None if self.zip_f_dict is None else self.zip_f_dict, self.zip_dir)

            # pipe_line
            try:
                for i, pipe_line_fn in enumerate(self.pipe_line_fns, start=1):
                    result = pipe_line_fn(df)
                    if type(result) is bool:
                        if result:
                            print(f'skip file {file_path}')
                            self.skipped_file_pth_lst.append(file_path)
                            break
                    else:
                        df = result

                    if i == len(self.pipe_line_fns):
                        self.processed_file_pth_lst.append(file_path)
                        return file_path, df
            except:
                self.skipped_file_pth_lst.append(file_path)
                print(f'skip file {file_path}')
                print(sys.exc_info())


# -------------  number plate associated io -----------------
def find_associated_trjs(pool_pths, file_pths):
    pool_plate_dict = make_plate_dict(pool_pths)
    existing_plate_dict = make_plate_dict(file_pths)

    return [trj for plate, trjs in existing_plate_dict.items() for trj in pool_plate_dict[plate] if trj not in trjs]


def find_dissociated_trj(pool_pths: list, file_pths):
    plates = [get_plate_from_path(pth) for pth in file_pths]
    pool_pths = copy.deepcopy(pool_pths)
    pool_plates = [get_plate_from_path(pth) for pth in pool_pths]
    return [pth for plate, pth in zip(pool_plates, pool_pths) if plate not in plates]


def collect_files(zip_pths, plate_lst):
    print(f'get file paths from {len(zip_pths)} zip files...')
    plate_dict = {plate: [] for plate in plate_lst}
    for zip_pth in zip_pths:
        print(f'processing {zip_pth}... ')
        pool_files = get_file_lst_from_zip(zip_pth, [])
        pool_plate_dict = make_plate_dict(pool_files)
        plate_dict.update([(plate, plate_dict[plate] + pool_plate_dict.get(plate, [])) for plate in plate_dict.keys()])

    return plate_dict


def filter_plate_dict(plate_dict, zip_f_dict, zip_dir, filter_fn):
    plate_dict = copy.deepcopy(plate_dict)
    for key in plate_dict.keys():
        print(f'filtering plate {key} ...')
        plate_dict[key] = [pth for pth in tqdm(plate_dict[key])
                           if filter_fn(get_df_from_raw_pth(pth, default_load_fn, zip_f_dict, zip_dir))]
    return plate_dict


if __name__ == '__main__':
    with open('filtered_files.json', 'r') as f:
        filtered_file_pth = json.load(f)
    zip_f_dict = get_zip_f_dict(ZIP_FILE_PATHS_LOCAL)
    find_prior_file('sample', 'track/2019/08/24/2_%D4%C1BFM675.txt', zip_f_dict)

    raw_pth = 'track/2019/08/25/2_%D4%C1BFM675.txt'
    get_df_from_raw_pth(raw_pth, default_load_fn, zip_f_dict)

    plate_dict = make_plate_dict([pth for pths in filtered_file_pth.values() for pth in pths])
    plate_trjs = list(plate_dict.values())[1]
    df = get_df_from_raw_pth(plate_trjs[0], default_load_fn, zip_f_dict)
    dfs = [get_df_from_raw_pth(pth, default_load_fn, zip_f_dict) for pth in plate_trjs]

    from graphutil import plot_trajectory, plot_trajectory_loop, plot_trajectories_loop, plot_bundle_loop

    # # fig, ax = plt.subplots(figsize=(8, 6))
    plot_iter = (plot_trajectory(get_df_from_raw_pth(raw_pth, default_load_fn, zip_f_dict)) for raw_pth in plate_trjs)
    plot_trajectory_loop(df, speedup=50)
    plot_trajectories_loop(dfs, speedup=50)
    plot_bundle_loop(
        ([get_df_from_raw_pth(pth, default_load_fn, zip_f_dict) for pth in trjs] for trjs in plate_dict.values()),
        clear=False)

    # zip file test
    zip_file1 = ZipFile("archive/my_file.zip")
    zip_file = ZipFile("archive/my_file.zip", "w")
    zip_file.write('archive/stop_heat_map_2_4_2020.py')
    zip_file.close()
    zip_file.printdir()
    zip_file1.printdir()
    del zip_file1
    remove_file("archive/my_file.zip")

    # find plate related dataset
    data_dir = '/Volumes/Elements SE/'
    zip_pths = [os.path.join(data_dir, pth) for pth in ZIP_FILE_PATHS]
    zip_f_dict = get_zip_f_dict(zip_pths)
    d = collect_files(zip_pths, [])
    filtered_d = filter_plate_dict(d, zip_f_dict, data_dir, lambda x: True)
