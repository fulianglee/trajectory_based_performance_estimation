import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn


def range_scale(x, factor):
    if x == 0:
        return 0
    if x > 0:
        return (x - 0.5) * factor
    if x < 0:
        return (x + 0.5) * factor


def p_mse(pre_y, y, factor=1):
    error = pre_y - y
    mse = np.square([*map(lambda x: range_scale(x, factor), error)]).mean()
    # mse = np.square(pre_y - y).mean()
    print(f'Mean square error is {mse:.3f}, scaled factor={factor}')


def p_rmse(pre_y, y, factor=1):
    error = pre_y - y
    rmse = np.sqrt(np.square([*map(lambda x: range_scale(x, factor), error)]).mean())
    # rmse = np.sqrt(np.square(pre_y - y).mean())
    print(f'Root mean square error is {rmse:.3f}, scaled factor={factor}')


def p_ame(pre_y: np.ndarray, y: np.ndarray, factor=1):
    error = pre_y - y
    ame = np.abs([*map(lambda x: range_scale(x, factor), error)]).mean()
    print(f'Absolute Mean error is {ame:.3f}, scaled factor={factor}')


def p_accuracy(pre_y: np.ndarray, y: np.ndarray):
    accuracy = sum(pre_y == y) / len(pre_y)
    print(f'Accuracy is {accuracy * 100:.2f}%')


def p_cm(pre_y: np.ndarray, y: np.ndarray):
    y_actu = pd.Series(y, name='Actual')
    y_pred = pd.Series(pre_y, name='Predicted')
    df_confusion = pd.crosstab(y_actu, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)
    print(df_confusion.to_markdown())


def p_errors(pre_y: np.ndarray, y: np.ndarray, factor=1):
    p_mse(pre_y, y, factor)
    p_rmse(pre_y, y, factor)
    p_ame(pre_y, y, factor)
    p_accuracy(pre_y, y)
    p_cm(pre_y, y)


def accuracy_under_minutes(c_m: np.ndarray, slice_interval):
    for i in range(len(c_m) - 1):
        true_total = c_m[0:i + 1].sum()
        pred_total = c_m[:, 0:i + 1].sum()
        in_slot = c_m[0:i + 1, 0:i + 1].sum()
        print(f'for under {(i + 1) * slice_interval} minute, '
              f'recall={in_slot / true_total * 100:.2f}%, precision={in_slot / pred_total * 100:.2f}%')


# for ETA under variable minute, percentage of trucks never arrive after 75 minutes
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def eta_delay_threshold(c_m: np.ndarray, slice_interval, slice_count, chunk_size=1):
    chks = chunks(range(len(c_m) - 1), chunk_size)
    if chunk_size != 1:
        chks = [[0]] + [*chks]
    for chunk in chks:
        i = chunk[-1]
        eta_acc = c_m[:, 0:i + 1].sum()
        threshold_count = c_m[-1, 0:i + 1].sum()
        print(f'for ETA under {(i + 1) * slice_interval} minute, '
              f'{threshold_count / eta_acc * 100:.2f}% of trucks never arrive within '
              f'{slice_interval * (slice_count - 1)} minutes')


def eta_delay_threshold_C(c_m: np.ndarray, slice_interval, slice_count, chunk_size=1):
    chks = chunks(range(len(c_m) - 1), chunk_size)
    if chunk_size != 1:
        chks = [[0]] + [*chks]
    for chunk in chks:
        i = chunk[-1]
        eta_acc = c_m[:, 0:i + 1].sum()
        threshold_count = c_m[-1, 0:i + 1].sum()
        print(f'预测{(i + 1) * slice_interval}分钟内到达的车辆中有{threshold_count / eta_acc * 100:.2f}%'
              f'最终在{slice_interval * (slice_count - 1)}分钟后到达。')


# for ETA over 75 minute, percentage of trucks arrive within under variable minutes
def eta_early_threshold(c_m: np.ndarray, slice_interval, slice_count, chunk_size=1):
    chks = chunks(range(len(c_m) - 1), chunk_size)
    if chunk_size != 1:
        chks = [[0]] + [*chks]
    for chunk in chks:
        i = chunk[-1]
        eta_acc = c_m[:, -1].sum()
        threshold_count = c_m[0:i + 1, -1].sum()
        print(f'for ETA over {slice_interval * (slice_count - 1)} minute, '
              f'{threshold_count / eta_acc * 100:.2f}% of trucks arrive within {(i + 1) * slice_interval} minutes')


def eta_early_threshold_C(c_m: np.ndarray, slice_interval, slice_count, chunk_size=1):
    chks = chunks(range(len(c_m) - 1), chunk_size)
    if chunk_size != 1:
        chks = [[0]] + [*chks]
    for chunk in chks:
        i = chunk[-1]
        eta_acc = c_m[:, -1].sum()
        threshold_count = c_m[0:i + 1, -1].sum()
        print(f'预测在{slice_interval * (slice_count - 1)}分钟后到达的车辆有{threshold_count / eta_acc * 100:.2f}%'
              f'提早在{(i + 1) * slice_interval}分钟内到达。')


def draw_confusion_grid(confusion_matrix: np.ndarray, smear):
    pred_count_slot = confusion_matrix.sum(0)
    true_count_slot = confusion_matrix.sum(-1)

    if smear == 'pred':
        adjusted_cm = confusion_matrix / pred_count_slot
    elif smear == 'true':
        adjusted_cm = (confusion_matrix.transpose() / true_count_slot).transpose()
    else:
        raise Exception(f'smear by {smear} is not defined!')

    index = [f'{i} ({count})' for i, count in enumerate(true_count_slot)]
    columns = [f'{i} ({count})' for i, count in enumerate(pred_count_slot)]

    df_cm = pd.DataFrame(adjusted_cm, index=index, columns=columns).round(1)

    f = plt.figure(figsize=(10, 7))
    f.suptitle(f'Normalized Confusion Matrix smeared along {smear} axis')
    sn.heatmap(df_cm, annot=True, cmap="YlGnBu")
    plt.gca().set(ylabel='ground truth axis', xlabel='prediction axis')
    plt.gca().tick_params(axis='both', which='major', labelsize=8)
