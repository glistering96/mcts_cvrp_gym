import json
import logging
import os
import shutil
import sys
import time
from copy import deepcopy
from dataclasses import fields
from datetime import datetime

import numpy as np
import pytz
from torch.utils.tensorboard.summary import hparams

from src.common.dataclass import StepState


def get_result_folder(desc, result_dir, date_prefix=True):
    process_start_time = datetime.now(pytz.timezone("Asia/Seoul"))

    if date_prefix is True:
        _date_prefix = process_start_time.strftime("%Y%m%d_%H%M%S")
        result_folder = f'{result_dir}/{_date_prefix}-{desc}'

    else:
        result_folder = f'{result_dir}/{desc}'

    return result_folder


def set_result_folder(folder):
    global result_folder
    result_folder = folder


def create_logger(log_file=None, result_dir='./result', **kwargs):
    # print(log_file)
    if 'result_dir' in log_file:
        result_dir = log_file['result_dir']

    if 'filepath' not in log_file:
        log_file['filepath'] = get_result_folder(log_file['desc'], result_dir=result_dir,
                                                 date_prefix=log_file['date_prefix'])

    if 'desc' in log_file:
        log_file['filepath'] = log_file['filepath'].format(desc='_' + log_file['desc'])
    else:
        log_file['filepath'] = log_file['filepath'].format(desc='')

    if 'filename' in log_file:
        filename = log_file['filepath'] + '/' + log_file['filename']
    else:
        filename = log_file['filepath'] + '/' + 'log.txt'

    if not os.path.exists(log_file['filepath']):
        os.makedirs(log_file['filepath'])

    file_mode = 'a' if os.path.isfile(filename) else 'w'

    root_logger = logging.getLogger()
    root_logger.setLevel(level=logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] %(filename)s(%(lineno)d) : %(message)s", "%Y-%m-%d %H:%M:%S")

    for hdlr in root_logger.handlers[:]:
        root_logger.removeHandler(hdlr)

    # write to file
    fileout = logging.FileHandler(filename, mode=file_mode)
    fileout.setLevel(logging.INFO)
    fileout.setFormatter(formatter)
    root_logger.addHandler(fileout)

    # write to console
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    root_logger.addHandler(console)


def deepcopy_state(state):
    to = StepState()

    for field in fields(StepState):
        setattr(to, field.name, deepcopy(getattr(state, field.name)))

    return to


def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """
    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y-ypred] / Var[y]

    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero

    :param y_pred: the prediction
    :param y_true: the expected value
    :return: explained variance of ypred and y
    """
    if isinstance(y_pred, list):
        y_pred = np.array(y_pred)

    if isinstance(y_true, list):
        y_true = np.array(y_true)

    assert y_true.ndim == 1 and y_pred.ndim == 1
    var_y = np.var(y_true)
    return np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y


def cal_distance(xy, visiting_seq):
    """
    :param xy: coordinates of nodes
    :param visiting_seq: sequence of visiting node idx
    :return:

    1. Gather coordinates on a given sequence of nodes
    2. roll by -1
    3. calculate the distance
    4. return distance

    """
    desired_shape = tuple(list(visiting_seq.shape) + [2])
    gather_idx = np.broadcast_to(visiting_seq[:, :, None], desired_shape)

    original_seq = np.take_along_axis(xy, gather_idx, 1)
    rolled_seq = np.roll(original_seq, -1, 1)

    segments = np.sqrt(((original_seq - rolled_seq) ** 2).sum(-1))
    distance = segments.sum(1)
    return distance


class TimeEstimator:
    def __init__(self):
        self.logger = logging.getLogger('TimeEstimator')
        self.start_time = time.time()
        self.count_zero = 0

    def reset(self, count=1):
        self.start_time = time.time()
        self.count_zero = count - 1

    def get_est(self, count, total):
        curr_time = time.time()
        elapsed_time = curr_time - self.start_time
        remain = total - count
        remain_time = elapsed_time * remain / (count - self.count_zero)

        elapsed_time /= 3600.0
        remain_time /= 3600.0

        return elapsed_time, remain_time

    def get_est_string(self, count, total):
        elapsed_time, remain_time = self.get_est(count, total)

        elapsed_time_str = "{:.2f}h".format(elapsed_time) if elapsed_time > 1.0 else "{:.2f}m".format(elapsed_time * 60)
        remain_time_str = "{:.2f}h".format(remain_time) if remain_time > 1.0 else "{:.2f}m".format(remain_time * 60)

        return elapsed_time_str, remain_time_str

    def print_est_time(self, count, total):
        elapsed_time_str, remain_time_str = self.get_est_string(count, total)

        self.logger.info("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(
            count, total, elapsed_time_str, remain_time_str))


def copy_all_src(dst_root):
    # execution dir
    if os.path.basename(sys.argv[0]).startswith('ipykernel_launcher'):
        execution_path = os.getcwd()
    else:
        execution_path = os.path.dirname(sys.argv[0])

    # home dir setting
    tmp_dir1 = os.path.abspath(os.path.join(execution_path, sys.path[0]))
    tmp_dir2 = os.path.abspath(os.path.join(execution_path, sys.path[1]))

    if len(tmp_dir1) > len(tmp_dir2) and os.path.exists(tmp_dir2):
        home_dir = tmp_dir2
    else:
        home_dir = tmp_dir1

    if 'src' not in home_dir:
        home_dir = os.path.join(home_dir, 'src')

    # make target directory
    dst_path = os.path.join(dst_root, 'src')
    #
    # if not os.path.exists(dst_path):
    #     os.makedirs(dst_path)

    shutil.copytree(home_dir, dst_path, dirs_exist_ok=True)


def check_debug():
    import sys

    eq = sys.gettrace() is None

    if eq is False:
        return True
    else:
        return False


def concat_key_val(*args):
    result = deepcopy(args[0])

    for param_group in args[1:]:

        for k, v in param_group.items():
            result[k] = v

    if 'device' in result:
        del result['device']

    return result


def add_hparams(writer, param_dict, metrics_dict, step=None):
    exp, ssi, sei = hparams(param_dict, metrics_dict)

    writer.file_writer.add_summary(exp)
    writer.file_writer.add_summary(ssi)
    writer.file_writer.add_summary(sei)

    if step is not None:
        for k, v in metrics_dict.items():
            writer.add_scalar(k, v, step)


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)