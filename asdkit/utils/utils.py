#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2023/9/21 17:01
# @Author: ZhaoKe
# @File : data_utils.py
# @Software: PyCharm

"""
functional functions
"""
import os
import shutil
import glob

import librosa
import yaml
import csv
import logging
import random
import numpy as np
import distutils.util
from tqdm import tqdm
import torch
from asdkit.utils.logger import setup_logger

logger = setup_logger(__name__)


def get_logger(filename):
    logging.basicConfig(filename=filename, level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler())
    return logger


def print_arguments(args=None, configs=None):
    if args:
        logger.info("----------- 额外配置参数 -----------")
        for arg, value in sorted(vars(args).items()):
            logger.info("%s: %s" % (arg, value))
        logger.info("------------------------------------------------")
    if configs:
        logger.info("----------- 配置文件参数 -----------")
        for arg, value in sorted(configs.items()):
            if isinstance(value, dict):
                logger.info(f"{arg}:")
                for a, v in sorted(value.items()):
                    if isinstance(v, dict):
                        logger.info(f"\t{a}:")
                        for a1, v1 in sorted(v.items()):
                            logger.info("\t\t%s: %s" % (a1, v1))
                    else:
                        logger.info("\t%s: %s" % (a, v))
            else:
                logger.info("%s: %s" % (arg, value))
        logger.info("------------------------------------------------")


def add_arguments(argname, type, default, help, argparser, **kwargs):
    type = distutils.util.strtobool if type == bool else type
    argparser.add_argument("--" + argname,
                           default=default,
                           type=type,
                           help=help + ' 默认: %(default)s.',
                           **kwargs)


class Dict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


def dict_to_object(dict_obj):
    if not isinstance(dict_obj, dict):
        return dict_obj
    inst = Dict()
    for k, v in dict_obj.items():
        inst[k] = dict_to_object(v)
    return inst


sep = os.sep


def print_dict(_dict):
    for key in _dict:
        print(key, '\t', _dict[key])


def load_ckpt(model, resume_model, m_type=None, load_epoch=None):
    if m_type is None:
        state_dict = torch.load(os.path.join(resume_model, f'model_{load_epoch}.pth'))
    else:
        if load_epoch:
            state_dict = torch.load(os.path.join(resume_model, f'model_{m_type}_{load_epoch}.pth'))
        else:
            state_dict = torch.load(os.path.join(resume_model, f'model_{m_type}.pth'))
    model.load_state_dict(state_dict)


def load_yaml(file_path='./config.yaml'):
    with open(file_path, encoding='utf_8') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    return params


def save_yaml_file(file_path, data: dict):
    with open(file_path, "w") as f:
        yaml.safe_dump(data, f, encoding='utf-8', allow_unicode=True)


def save_load_version_files(path, file_patterns, pass_dirs=None):
    #    save latest version files
    if pass_dirs is None:
        pass_dirs = ['.', '_', 'runs', 'results']
    copy_files(f'.{sep}', 'runs/latest_project', file_patterns, pass_dirs)
    copy_files(f'.{sep}', os.path.join(path, 'project'), file_patterns, pass_dirs)


def save_csv(file_path, data: list):
    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(data)
    print("save " + file_path)


# 复制目标文件到目标路径
def copy_files(root_dir, target_dir, file_patterns, pass_dirs=['.git']):
    # print(root_dir, root_dir.split(sep), [name for name in root_dir.split(sep) if name != ''])
    os.makedirs(target_dir, exist_ok=True)
    len_root = len([name for name in root_dir.split(sep) if name != ''])
    for root, _, _ in os.walk(root_dir):
        cur_dir = sep.join(root.split(sep)[len_root:])
        first_dir_name = cur_dir.split(sep)[0]
        if first_dir_name != '':
            if (first_dir_name in pass_dirs) or (first_dir_name[0] in pass_dirs): continue
        # print(len_root, root, cur_dir)
        target_path = os.path.join(target_dir, cur_dir)
        os.makedirs(target_path, exist_ok=True)
        files = []
        for file_pattern in file_patterns:
            file_path_pattern = os.path.join(root, file_pattern)
            files += sorted(glob.glob(file_path_pattern))
        for file in files:
            target_path_file = os.path.join(target_path, os.path.split(file)[-1])
            shutil.copyfile(file, target_path_file)


def save_model_state_dict(file_path, epoch=None, net=None, optimizer=None):
    import torch
    state_dict = {
        'epoch': epoch,
        'optimizer': optimizer.state_dict() if optimizer else None,
        'model': net.state_dict() if net else None,
    }
    torch.save(state_dict, file_path)


def set_type(value):
    if value.lower() == 'true':
        return True
    elif value.lower() == 'false':
        return False
    else:
        return value


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_filename_list(dir_path, mode="*", pattern='*', ext='*'):
    """
    find all extention files under directory
    :param mode: mode in ['train', 'test']
    :param dir_path: directory path
    :param ext: extention name, like wav, png...
    :param pattern: filename pattern for searching
    :return: files path list
    """
    filename_list = []
    for root, _, _ in os.walk(dir_path):
        # if not "test" == root[-4:]:
        #     continue
        # print(root)
        file_path_pattern = root + '/' + mode + "/" + f'{pattern}.{ext}'
        files = sorted(glob.glob(file_path_pattern))
        # for file in files:
        #     print(file)
        # print("==========")
        filename_list += files
    return filename_list


# def get_machine_id_list(data_dir, mode='*'):
#     res_list = get_filename_list(data_dir, mode=mode, ext='wav')
#     machine_id_list = sorted(list(set(
#         itertools.chain.from_iterable([re.findall('id_[0-9][0-9]', ext_id) for ext_id in res_list])
#     )))
#     return machine_id_list


def get_machine_id_list(data_dir):
    id_set = set()
    for item in os.listdir(data_dir):
        id_set.add(item.split('_')[2])
    return list(id_set)


def metadata_to_label(data_dirs):
    meta2label = {}
    label2meta = {}
    with open(data_dirs, 'r') as f:
        line = f.readline().strip()
        mids = set()
        while line:
            mname = line.split('/')[-3]
            mid = line.split('\t')[2]
            # print(line.split('\t')[2])
            mids.add(mname + "-id_" + mid)
            line = f.readline().strip()
    machine_id_list = sorted(list(mids))
    # print(machine_id_list)
    for idx, meta in enumerate(machine_id_list):
        meta2label[meta] = idx
        label2meta[idx] = meta
    return meta2label, label2meta


def create_test_file_list(target_dir,
                          id_name,
                          dir_name='test',
                          prefix_normal='normal',
                          prefix_anomaly='anomaly',
                          ext='wav'):
    normal_files_path = f'{target_dir}/{prefix_normal}_{id_name}*.{ext}'
    # print(normal_files_path)
    normal_files = sorted(glob.glob(normal_files_path))
    # print(normal_files)
    normal_labels = np.zeros(len(normal_files))

    anomaly_files_path = f'{target_dir}/{prefix_anomaly}_{id_name}*.{ext}'
    anomaly_files = sorted(glob.glob(anomaly_files_path))
    anomaly_labels = np.ones(len(anomaly_files))

    files = np.concatenate((normal_files, anomaly_files), axis=0)
    labels = np.concatenate((normal_labels, anomaly_labels), axis=0)
    return files, labels


# 根据对角余弦值计算准确率和最优的阈值
def cal_accuracy_threshold(y_score, y_true):
    y_score = np.asarray(y_score)
    y_true = np.asarray(y_true)
    best_accuracy = 0
    best_threshold = 0
    for i in tqdm(range(0, 100)):
        threshold = i * 0.01
        y_test = (y_score >= threshold)
        acc = np.mean((y_test == y_true).astype(int))
        if acc > best_accuracy:
            best_accuracy = acc
            best_threshold = threshold

    return best_accuracy, best_threshold


# 根据对角余弦值计算准确率
def cal_accuracy(y_score, y_true, threshold=0.5):
    y_score = np.asarray(y_score)
    y_true = np.asarray(y_true)
    y_test = (y_score >= threshold)
    accuracy = np.mean((y_test == y_true).astype(int))
    return accuracy


# 计算对角余弦值
def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))


# modules from ARGAN-AD
def get_clip_addr(clip_dir, ext='wav'):
    """ get files in a directory"""
    clip_addr = []
    for f in os.listdir(clip_dir):
        if f.split('.')[-1] == ext:
            clip_addr.append(os.path.join(clip_dir, f).replace('\\', '/'))
    clip_addr = sorted(clip_addr)  # 0nor -> 0ano -> 1nor -> 1ano -> ...
    return clip_addr


def extract_mid(clip_addr, set_type, data_type):
    """ get machine id from all the files in a directory"""
    # train：normal_id_01_00000791.wav
    # set：anomaly_id_01_00000181.wav(dev), id_05_00000252.wav(eval)
    all_mid = np.zeros((len(clip_addr)), dtype=int)  # [machine id]
    for i, clip_name in enumerate(clip_addr):
        file_name = os.path.basename(clip_name)[:os.path.basename(clip_name).index('.wav')]
        if set_type == 'eval' and data_type == 'test':
            mid = int(file_name.split('_')[1][1])
        else:
            mid = int(file_name.split('_')[2][1])
        all_mid[i] = mid
    return all_mid


def generate_label(clip_addr, set_type, data_type):
    """ get label vector of all files in a directory"""
    label = np.zeros((len(clip_addr)), dtype=int)  # a label for each clip

    for idx in range(len(clip_addr)):
        # train：normal_id_01_00000791.wav
        # test：anomaly_id_01_00000181.wav(dev), id_05_00000252.wav(eval)
        if set_type == 'dev' and data_type == 'test':
            status_note = clip_addr[idx].split('/')[-1].split('_')[0]
            assert status_note in ['normal', 'anomaly']
            status = 0 if status_note == 'normal' else 1
        elif data_type == 'train':
            status = 0
        else:  # for eval test
            status = -1
        label[idx] = status
    return label


def generate_spec(clip_addr, spec, fft_num, mel_bin, frame_hop, top_dir,
                  mt, data_type, setn, rescale_ctl=True):
    all_clip_spec = None

    for st in clip_addr.keys():  # 'dev', 'eval'
        save_dir = os.path.join(top_dir, st, mt)
        os.makedirs(save_dir, exist_ok=True)
        raw_data_file = os.path.join(save_dir,
                                     f'{data_type}_raw_{spec}_{mel_bin}_{fft_num}_{frame_hop}_1.npy')

        if not os.path.exists(raw_data_file):
            for idx in tqdm(range(len(clip_addr[st]))):
                clip, sr = librosa.load(clip_addr[st][idx], sr=None, mono=True)
                if spec == 'mel':
                    mel = librosa.feature.melspectrogram(y=clip, sr=sr, n_fft=fft_num,
                                                         hop_length=frame_hop, n_mels=mel_bin)
                    mel_db = librosa.power_to_db(mel, ref=1)  # log-mel, (mel_bin, frame_num)
                    if idx == 0:
                        set_clip_spec = np.zeros((len(clip_addr[st]) * mel_bin, mel.shape[1]), dtype=np.float32)
                    set_clip_spec[idx * mel_bin:(idx + 1) * mel_bin, :] = mel_db
                elif spec == 'stft':
                    stft = librosa.stft(y=clip, n_fft=fft_num, hop_length=frame_hop)
                    stabs = np.abs(stft)
                    if idx == 0:
                        set_clip_spec = np.zeros((len(clip_addr[st]) * stabs.shape[0], stabs.shape[1]),
                                                 dtype=np.float32)
            np.save(raw_data_file, set_clip_spec)
        else:
            print("spec file "+raw_data_file+" exists, loading...")
            set_clip_spec = np.load(raw_data_file)
        if all_clip_spec is None:
            all_clip_spec = set_clip_spec
        else:
            all_clip_spec = np.vstack((all_clip_spec, set_clip_spec))

    frame_num_per_clip = all_clip_spec.shape[-1]
    save_dir = os.path.join(top_dir, setn, mt)
    os.makedirs(save_dir, exist_ok=True)
    scale_data_file = os.path.join(save_dir,
                                   f'train_scale_mel_{mel_bin}_{fft_num}_{frame_hop}_1.npy')
    if data_type == 'train' and rescale_ctl:  # scale to [-1,1]
        max_v = np.max(all_clip_spec)
        min_v = np.min(all_clip_spec)
        np.save(scale_data_file, [max_v, min_v])
    else:
        maxmin = np.load(scale_data_file)
        max_v, min_v = maxmin[0], maxmin[1]

    mean = (max_v + min_v) / 2
    scale = (max_v - min_v) / 2
    all_clip_spec = (all_clip_spec - mean) / scale

    all_clip_spec = all_clip_spec.reshape(-1, mel_bin, frame_num_per_clip)
    return all_clip_spec


def config_summary(param):
    summary = []
    # print(param.feat)
    summary.append({'fft_num': param.feat['fft_num'],
                    'mel_bin': param.feat['mel_bin'],
                    'frame_hop': param.feat['frame_hop'],
                    'graph_hop_f': param.feat['graph_hop_f']
                    })
    summary.append({'datasets': param.train_set})
    # print(param.train)
    summary.append({'act': param.net['act'],
                   'normalize': param.net['normalize'],
                   'nz': param.net['nz'],
                   'ndf': param.net['ndf'],
                   'ngf': param.net['ngf']})
    # print(param.net)
    summary.append({'lrD': param.train['lrD'],
                     'lrG': param.train['lrG'],
                     'beta1': param.train['beta1'],
                     'batch_size': param.train['bs'],
                     'epoch': param.train['epoch']})
    # print(param.train['wgan'])
    summary.append(param.train['wgan'])
    return summary


if __name__ == '__main__':
    # res_list = get_filename_list('E:/DATAS/DCASE2020Task2ASD/datasets', ext='wav')
    # for item in res_list:
    #     print(item)
    # mid_list = get_machine_id_list('E:/DATAS/DCASE2020Task2ASD/datasets')
    # for item in mid_list:
    #     print(item)

    # KeyError: 'ToyCar-id_00'
    # dir_path = "C:/Program Files (zk)/data/dcase2020-asd-task2/data/eval_dataset"
    # dir_list = os.walk(dir_path)
    # for item in dir_list:
    #     print(item[0])
    # print(get_machine_id_list("C:/Program Files (zk)/data/dcase2020-asd-task2/data/eval_dataset"))
    # files, labels = create_test_file_list("E:/DATAS/DCASE2020Task2ASD/eval_dataset/eval_data_test_slider/slider/test",
    #                                       id_name="05", mode="test")
    # print(files)
    # print(labels)
    # files = get_filename_list("G:/DATAS-DCASE-ASD/DCASE2020Task2ASD/dataset/", mode="test")
    # for item in files:
    #     print(item)
    files = create_test_file_list("G:/DATAS-DCASE-ASD/DCASE2020Task2ASD/dataset/dev_data_fan/fan/test", id_name="id_00")
    for item in files:
        print(item)
