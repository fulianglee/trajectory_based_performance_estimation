"""tabulate zip file to training and target data
input: zip_path, file_name
output: data_frame of the following format

data format:
| gps_long | gps_lat | minute_of_hour | hour_of_day | day_of_week | week_of_year | target_minute |

option:
    remove_duplicate: reduce more than three of consecutive pairs of gps_long and gps_lat that are the same
                        to less than two pairs

"""
from functools import reduce
import numpy as np
import pandas as pd
import numpy
import io
from zipfile import ZipFile
import dateutil.parser as date_parser
import datetime


# --------------------- extraction --------------------------
def get_df_with_name_from_zip(zip_file: ZipFile, file_name: str, col_names):
    io_bytes = io.BytesIO(zip_file.read(file_name))
    df = pd.read_csv(io_bytes, ':', index_col=False, names=col_names)
    return df


def get_numpy_with_keys_from_df(df: pd.DataFrame, keys) -> numpy.ndarray:
    return df[keys].to_numpy()


def get_gps_numpy(arch: ZipFile, file: str, col_names):
    df = get_df_with_name_from_zip(arch, file, col_names)
    return get_numpy_with_keys_from_df(df, ['gps_long', 'gps_lat', 'gps_time'])


# --------------------- helper functions -------------------------------


def remove_zero_batch(dfs, keys=('gps_long', 'gps_lat')):
    import operator
    return [df[reduce(lambda x, y: operator.and_(x, y),
                      [df[key] != 0 for key in keys])]
            for df in dfs]


def remove_gps_zero_batch(dfs):
    return [df[(df.gps_long != 0) & (df.gps_lat != 0)] for df in dfs]


def remove_gps_zero(df: pd.DataFrame):
    return df[(df.gps_long != 0) & (df.gps_lat != 0)]


def get_year_month_day(time_str: str):
    d = date_parser.parse(time_str)
    return d.year, d.month, d.day


def get_hour_minute_second(time_str: str):
    d = date_parser.parse(time_str)
    return d.hour, d.minute, d.second


def get_week_of_year(time_str: str):
    year, month, day = get_year_month_day(time_str)
    return int(datetime.date(year, month, day).strftime("%V"))


def get_day_of_week(time_str: str):
    year, month, day = get_year_month_day(time_str)
    return int(datetime.date(year, month, day).strftime("%w"))


def get_minute_of_hour(time_str: str):
    return date_parser.parse(time_str).minute


def get_minute_of_day(time_str: str):
    parsed = date_parser.parse(time_str)
    return parsed.hour * 60 + parsed.minute


def get_second_of_day(time_str: str):
    parsed = date_parser.parse(time_str)
    return parsed.hour * 60 * 60 + parsed.minute * 60 + parsed.second


def get_hour_of_day(time_str: str):
    return date_parser.parse(time_str).hour


def get_earliest_time(df: pd.DataFrame, datetime_col='gps_time'):
    return date_parser.parse(df[datetime_col].iloc[0])


def get_latest_time(df: pd.DataFrame, datetime_col='gps_time'):
    return date_parser.parse(df[datetime_col].iloc[-1])


def get_duration_minute(df: pd.DataFrame):
    begin_t, end_t = get_earliest_time(df), get_latest_time(df)
    return (end_t - begin_t).seconds // 60


def get_minute_of_trj(df: pd.DataFrame):
    time_start_str = df['gps_time'][0]
    return df.assign(minute_of_trj=df['gps_time'].transform(lambda x: minute_differences(time_start_str, x)))


def get_second_of_trj(df: pd.DataFrame):
    time_start_str = df['gps_time'][0]
    return df.assign(second_of_trj=df['gps_time'].transform(lambda x: second_differences(time_start_str, x)))


def minute_differences(time_start: str, time_end: str):
    return second_differences(time_start, time_end) // 60


def second_differences(time_start: str, time_end: str):
    d_0, d_1 = date_parser.parse(time_start), date_parser.parse(time_end)
    return d_1.timestamp() - d_0.timestamp()
    # return (d_1 - d_0).seconds


def minute_diff_slot(time_start: str, time_end: str, interval_size=20, slot_n=16):
    slot = minute_differences(time_start, time_end) // interval_size
    return slot if slot < slot_n else slot_n - 1


def coord_str(gps_long: int, gps_lat: int):
    return f'{gps_long}:{gps_lat}'


def get_log_range(start, end, num=10, endpoint=False):
    return np.logspace(np.log10(start), np.log10(end), num=num, endpoint=endpoint)


# -------------------------- transform functions --------------------------------

# def extract_stops_minute(df: pd.DataFrame, threshold):
#     """return minute lapsed of consecutive gps location, showing results only above a threshold"""
#     # removing in-yantian and invalid gps location
#     truncated_df = exclude_yantian_and_remove_zero(df)
#
#     # group and loop to calculate lapsed time by second
#     grouped_df = truncated_df.groupby(['gps_long', 'gps_lat'])
#     dfs = []
#     for name, group in grouped_df:
#         if len(group) <= 1:
#             continue
#
#         lapsed_time = minute_differences(group.gps_time.iloc[0], group.gps_time.iloc[-1])
#         if lapsed_time > threshold:
#             dfs.append(pd.DataFrame({
#                 'gps_long': [name[0] / 600000],
#                 'gps_lat': [name[1] / 600000],
#                 'lapsed_minute': [lapsed_time]
#             }))
#
#     final_df = pd.concat(dfs).reset_index(drop=True)
#     return final_df

def ll_int_2_float(df: pd.DataFrame):
    return df.assign(gps_long=df['gps_long'] / 600000, gps_lat=df['gps_lat'] / 600000)


def ll_float_2_int(df: pd.DataFrame):
    return df.assign(gps_long=(df['gps_long'] * 600000).astype(int), gps_lat=(df['gps_lat'] * 600000).astype(int))


def extract_stop_bool(df: pd.DataFrame, threshold=1):
    if 'minute_of_day' not in df.columns:
        df['minute_of_day'] = df['gps_time'].apply(get_minute_of_day)
    # index each stop group
    stop_bool = (df['gps_long'].ne(df['gps_long'].shift()) | df['gps_lat'].ne(df['gps_lat'].shift())).cumsum()
    # followed by df.apply(lambda row: qtree.find_tree(row['gps_long'], row['gps_lat'])
    #                       if row['stop_bool'] else -1, axis=1)
    return df.assign(stop_bool=df['minute_of_day']
                     .groupby(stop_bool).transform(lambda x: True if x.max() - x.min() >= threshold else False))


def get_stop_bool(df: pd.DataFrame, threshold=1):
    if 'minute_of_day' in df.columns:
        minute_of_day = df['minute_of_day']
    else:
        minute_of_day = df['gps_time'].transform(get_minute_of_day)

    stop_bool = (df['gps_long'].ne(df['gps_long'].shift()) | df['gps_lat'].ne(df['gps_lat'].shift())).cumsum()
    grp = minute_of_day.groupby(stop_bool)
    return pd.concat([pd.Series([
                                    True if g_max - g_min >= threshold else False
                                ] * len(grp.groups[i])) for i, (g_max, g_min) in
                      enumerate(zip(grp.max(), grp.min()), 1)],
                     ignore_index=True)


def drop_duplicate_node(df: pd.DataFrame, keep_first=False):
    shift_direction = 1 if keep_first else -1
    repeat_bool = df['node'].ne(df['node'].shift(shift_direction))
    return df[repeat_bool]


def get_second_diff(df: pd.DataFrame, col_name='second_of_trj'):
    if col_name in df.columns:
        second_of_trj = df[col_name]
    else:
        second_of_trj = get_second_of_trj(df)[col_name]

    return (second_of_trj - second_of_trj.shift()).fillna(0)


def apply_to_df_row(df: pd.DataFrame, fns: dict):
    new_dfs = [df.apply(fn, axis=1) for fn in fns.values()]
    return df.assign(**dict([(col_name, col) for col_name, col in zip(fns.keys(), new_dfs)]))


def extract_mileage(df: pd.DataFrame, step, dropna=False):
    df['mileage_diff'] = df.mileage.diff(step)
    if dropna:
        return df.dropna(subset=['mileage_diff'])
    else:
        return df


def get_mean_from_dist(dist: dict):
    xs = np.array(list(dist.keys()))
    counts = np.array(list(dist.values()))
    mean = np.sum(xs * counts) / np.sum(counts)
    return mean


def get_mode_from_dist(dist: dict):
    x_counts = list(dist.items())
    x_counts = sorted(x_counts, key=lambda x: x[1], reverse=True)
    return x_counts[0][0]


def get_median_from_dist(dist: dict):
    x_counts = list(dist.items())
    x_counts = sorted(x_counts, key=lambda x: x[0])
    total = np.sum(list(dist.values()))
    acc = 0
    for x, counts in x_counts:
        acc = acc + counts
        if acc >= total / 2:
            return x


def get_std_from_dist(dist: dict):
    xs = np.array(list(dist.keys()))
    counts = np.array(list(dist.values()))
    N = np.sum(counts)
    mean = np.sum(xs * counts) / np.sum(counts)
    return np.sqrt(np.sum((xs - mean) ** 2 * counts) / N)


if __name__ == '__main__':
    pass
