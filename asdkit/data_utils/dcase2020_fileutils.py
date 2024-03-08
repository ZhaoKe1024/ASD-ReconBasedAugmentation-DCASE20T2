#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2023/12/30 18:38
# @Author: ZhaoKe
# @File : dcase2020_fileutils.py
# @Software: PyCharm
import itertools
import os
import sys
import re
import json
import glob
import argparse

import torch
from tqdm import tqdm
import numpy as np
import librosa
from asdkit.data_utils.audio import wav_slice_padding
from asdkit.utils.utils import logger, get_filename_list
from asdkit.data_utils.featurizer import Wave2Mel


def get_file_mtid_list(file_root="G:/DATAS-DCASE-ASD/DCASE2020Task2ASD/", is_dev=True, train_mode="train",
                       m_type="pump", mid=None):
    if is_dev:
        file_root = file_root + f"dataset/dev_data_{m_type}/{m_type}/{train_mode}/"
    else:
        file_root = file_root + f"eval_dataset/eval_data_test_{m_type}/{m_type}/test/"
    file_list = []
    for item in os.listdir(file_root):
        # print(item.split('_'))
        if item.split('_')[2] == mid:
            # print(item)
            file_list.append(item)
    return file_root, file_list


def get_idlist_for_mtype(file_root="G:/DATAS-DCASE-ASD/DCASE2020Task2ASD/", is_dev=True, train_mode="train",
                         m_type="pump", ):
    id_list = set()
    if is_dev:
        file_root = file_root + f"dataset/dev_data_{m_type}/{m_type}/{train_mode}/"
    else:
        file_root = file_root + f"eval_dataset/eval_data_test_{m_type}/{m_type}/test/"
    for item in os.listdir(file_root):
        id_list.add(item.split('_')[2])
    return list(id_list)


def get_train_file_list(file_root="G:/DATAS-DCASE-ASD/DCASE2020Task2ASD/", m_type="pump", is_dev=True,
                        train_mode="train"):
    if is_dev:
        file_root = file_root + f"dataset/dev_data_{m_type}/{m_type}/{train_mode}/"
    else:
        file_root = file_root + f"eval_dataset/eval_data_test_{m_type}/{m_type}/test/"
    return file_root, os.listdir(file_root)


def get_machine_id_list(data_dir):
    machine_id_list = sorted(list(set(
        itertools.chain.from_iterable([re.findall('id_[0-9][0-9]', ext_id) for ext_id in get_filename_list(data_dir)])
    )))
    return machine_id_list


def metadata_to_label(data_dirs):
    id_set = set()
    for data_dir in data_dirs:
        machine = data_dir.split('/')[-3]
        for item in os.listdir(data_dir):
            id_set.add(machine + '_' + '_'.join(item.split('_')[1:3]))
    machine_id_list = sorted(list(id_set))
    meta2label = {}
    label2meta = {}
    label = 0
    for meta in machine_id_list:
        meta2label[meta] = label
        label2meta[label] = meta
        label += 1
    if not os.path.exists("../datasets/d2020_metadata2label.json"):
        with open("../datasets/d2020_metadata2label.json", 'w', encoding='utf_8') as fp:
            json.dump(meta2label, fp, ensure_ascii=False)
    if not os.path.exists("../datasets/label2metadata.json"):
        with open("../datasets/label2metadata.json", 'w', encoding='utf_8') as fp:
            json.dump(label2meta, fp, ensure_ascii=False)
    return meta2label, label2meta


def get_train_all_file_list(file_root="G:/DATAS-DCASE-ASD/DCASE2020Task2ASD/", is_dev=True, train_mode="train"):
    m_types = ["fan", "pump", "slider", "valve", "ToyCar", "ToyConveyor"]
    m2l, l2m = None, None
    if is_dev:
        train_dirs = [file_root + f"dataset/dev_data_{m}/{m}/{train_mode}/" for m in m_types]
        m2l, l2m = metadata_to_label(train_dirs)
    else:
        train_dirs = [file_root + f"eval_dataset/eval_data_test_{m}/{m}/test/" for m in m_types]
    allfile_list = []
    for train_dir in train_dirs:
        print(train_dir)
        allfile_list.extend([os.path.join(train_dir, item) for item in os.listdir(train_dir)])
    return allfile_list, m2l, l2m


########################################################################
def test_file_list_generator(target_dir,
                             id_name,
                             dir_name="test",
                             mode=True,
                             prefix_normal="normal",
                             prefix_anomaly="anomaly",
                             ext="wav"):
    """
    target_dir : str
        base directory path of the dev_data or eval_data
    id_name : str
        id of wav file in <<test_dir_name>> directory
    dir_name : str (default="test")
        directory containing test data
    prefix_normal : str (default="normal")
        normal directory name
    prefix_anomaly : str (default="anomaly")
        anomaly directory name
    ext : str (default="wav")
        file extension of audio files

    return :
        if the mode is "development":
            test_files : list [ str ]
                file list for test
            test_labels : list [ boolean ]
                label info. list for test
                * normal/anomaly = 0/1
        if the mode is "evaluation":
            test_files : list [ str ]
                file list for test
    """
    # logger.info("target_dir : {}".format(target_dir + "_" + id_name))

    # development
    if mode:
        normal_files = sorted(
            glob.glob("{dir}/{dir_name}/{prefix_normal}_{id_name}*.{ext}".format(dir=target_dir,
                                                                                 dir_name=dir_name,
                                                                                 prefix_normal=prefix_normal,
                                                                                 id_name=id_name,
                                                                                 ext=ext)))
        normal_labels = np.zeros(len(normal_files))
        anomaly_files = sorted(
            glob.glob("{dir}/{dir_name}/{prefix_anomaly}_{id_name}*.{ext}".format(dir=target_dir,
                                                                                  dir_name=dir_name,
                                                                                  prefix_anomaly=prefix_anomaly,
                                                                                  id_name=id_name,
                                                                                  ext=ext)))
        anomaly_labels = np.ones(len(anomaly_files))
        files = np.concatenate((normal_files, anomaly_files), axis=0)
        labels = np.concatenate((normal_labels, anomaly_labels), axis=0)
        # logger.info("test_file  num : {num}".format(num=len(files)))
        if len(files) == 0:
            logger.exception("no_wav_file!!")
        # print("\n========================================")

    # evaluation
    else:
        files = sorted(
            glob.glob("{dir}/{dir_name}/*{id_name}*.{ext}".format(dir=target_dir,
                                                                  dir_name=dir_name,
                                                                  id_name=id_name,
                                                                  ext=ext)))
        labels = None
        # logger.info("test_file  num : {num}".format(num=len(files)))
        if len(files) == 0:
            logger.exception("no_wav_file!!")
        # print("\n=========================================")

    return files, labels


def get_machine_id_list_for_test(target_dir,
                                 dir_name="test",
                                 ext="wav"):
    """
    target_dir : str
        base directory path of "dev_data" or "eval_data"
    test_dir_name : str (default="test")
        directory containing test data
    ext : str (default="wav)
        file extension of audio files

    return :
        machine_id_list : list [ str ]
            list of machine IDs extracted from the names of test files
    """
    # create test files
    dir_path = os.path.abspath("{dir}/{dir_name}/*.{ext}".format(dir=target_dir, dir_name=dir_name, ext=ext))
    file_paths = sorted(glob.glob(dir_path))
    # extract id
    machine_id_list = sorted(list(set(itertools.chain.from_iterable(
        [re.findall('id_[0-9][0-9]', ext_id) for ext_id in file_paths]))))
    return machine_id_list


########################################################################
# get directory paths according to mode
########################################################################
def select_dirs(param, mode):
    """
    param : dict
        baseline.yaml data

    return :
        if active type the development :
            dirs :  list [ str ]
                load base directory list of dev_data
        if active type the evaluation :
            dirs : list [ str ]
                load base directory list of eval_data
    """
    if mode:
        logger.info("load_directory <- development")
        query = os.path.abspath("{base}/*".format(base=param["dev_directory"]))
    else:
        logger.info("load_directory <- evaluation")
        query = os.path.abspath("{base}/*".format(base=param["eval_directory"]))
    dirs = sorted(glob.glob(query))
    dirs = [f for f in dirs if os.path.isdir(f)]
    print("----train dirs:----")
    for dir_id, dir_name in enumerate(dirs):
        print(dir_name, ':', dir_id)
    # G:\DATAS-DCASE-ASD\DCASE2020Task2ASD\dataset\dev_data_ToyCar
    # G:\DATAS-DCASE-ASD\DCASE2020Task2ASD\dataset\dev_data_ToyConveyor
    # G:\DATAS-DCASE-ASD\DCASE2020Task2ASD\dataset\dev_data_fan
    # G:\DATAS-DCASE-ASD\DCASE2020Task2ASD\dataset\dev_data_pump
    # G:\DATAS-DCASE-ASD\DCASE2020Task2ASD\dataset\dev_data_slider
    # G:\DATAS-DCASE-ASD\DCASE2020Task2ASD\dataset\dev_data_valve
    return dirs


def file_list_generator(target_dir,
                        dir_name="train",
                        ext="wav"):
    """
    target_dir : str
        base directory path of the dev_data or eval_data
    dir_name : str (default="train")
        directory name containing training data
    ext : str (default="wav")
        file extension of audio files

    return :
        train_files : list [ str ]
            file list for training
    """
    logger.info("target_dir : {}".format(target_dir))

    # generate training list
    # G:\DATAS-DCASE-ASD\DCASE2020Task2ASD\dataset\dev_data_valve
    training_list_path = os.path.abspath("{dir}/{dir_name}/*.{ext}".format(dir=target_dir, dir_name=dir_name, ext=ext))
    if dir_name == "train":
        y_true = None
    if dir_name == "test":
        y_true = []
    files = sorted(glob.glob(training_list_path))
    if len(files) == 0:
        logger.exception("no_wav_file!!")
    mtid_list = []
    for fi in files:
        fpart = fi.split('\\')
        # print(fpart)
        mtype = fpart[-3]
        mtid_list.append(mtype + "-id_" + fpart[-1].split('_')[2])
        if dir_name == "test":
            if fpart[0][0] == "a":
                y_true.append(1)
            else:
                y_true.append(0)
    logger.info("train_file num : {num}".format(num=len(files)))
    return files, mtid_list, y_true


def list_to_vector_array(file_list,
                         msg="calc...",
                         n_mels=64,
                         frames=5,
                         n_fft=1024,
                         hop_length=512,
                         power=2.0):
    """
    convert the file_list to a vector array.
    file_to_vector_array() is iterated, and the output vector array is concatenated.

    file_list : list [ str ]
        .wav filename list of dataset
    msg : str ( default = "calc..." )
        description for tqdm.
        this parameter will be input into "desc" param at tqdm.

    return : numpy.array( numpy.array( float ) )
        vector array for training (this function is not used for test.)
        * dataset.shape = (number of feature vectors, dimensions of feature vectors)
    """
    # calculate the number of dimensions
    dims = n_mels * frames

    # iterate file_to_vector_array()
    for idx in tqdm(range(len(file_list)), desc=msg):
        # size=(309, 320)
        vector_array = file_to_vector_array(file_list[idx],
                                            n_mels=n_mels,
                                            frames=frames,
                                            n_fft=n_fft,
                                            hop_length=hop_length,
                                            power=power)
        if idx == 0:
            # (309*length, 320)
            dataset = np.zeros((vector_array.shape[0] * len(file_list), dims), float)
        dataset[vector_array.shape[0] * idx: vector_array.shape[0] * (idx + 1), :] = vector_array

    return dataset


def files_to_mel_list(file_list, mtids, y_true=None,
                      msg="calc...", configs=None, wav_length=147000):
    """
    convert the file_list to a vector array.
    file_to_vector_array() is iterated, and the output vector array is concatenated.

    file_list : list [ str ]
        .wav filename list of dataset
    msg : str ( default = "calc..." )
        description for tqdm.
        this parameter will be input into "desc" param at tqdm.

    return : numpy.array( numpy.array( float ) )
        vector array for training (this function is not used for test.)
        * dataset.shape = (number of feature vectors, dimensions of feature vectors)
    """
    # calculate the number of dimensions
    wav2mel = Wave2Mel(16000)
    dataset = []
    new_mtids = []
    new_y_true = [] if y_true else None
    # iterate file_to_vector_array()
    for idx in tqdm(range(len(file_list)), desc=msg):
        y, sr = librosa.load(file_list[idx], sr=None, mono=False)
        y = wav_slice_padding(y, save_len=wav_length)

        vector_array = wav2mel(torch.from_numpy(y.T))
        # --------------------
        # ------原始信号-------
        # --------------------
        dataset.append(vector_array)
        new_mtids.append(mtids[idx])
        if y_true:
            new_y_true.append(y_true[idx])

        # --------------------
        # ------降噪扩充-------
        # --------------------
        # if configs["augment"]["time_shift"]:

    return dataset, new_mtids, new_y_true


########################################################################
# feature extractor
########################################################################
def file_to_vector_array(file_name,
                         n_mels=64,
                         frames=5,
                         n_fft=1024,
                         hop_length=512,
                         power=2.0):
    """
    convert file_name to a vector array.

    file_name : str
        target .wav file

    return : numpy.array( numpy.array( float ) )
        vector array
        * dataset.shape = (dataset_size, feature_vector_length)
    """
    # 01 calculate the number of dimensions
    dims = n_mels * frames

    # 02 generate melspectrogram using librosa
    y, sr = librosa.load(file_name, sr=None, mono=False)
    mel_spectrogram = librosa.feature.melspectrogram(y=y,
                                                     sr=sr,
                                                     n_fft=n_fft,
                                                     hop_length=hop_length,
                                                     n_mels=n_mels,
                                                     power=power)
    # print("shape of mel_spectrogram:", mel_spectrogram.shape)
    # 03 convert melspectrogram to log mel energy (64, 313)
    log_mel_spectrogram = 20.0 / power * np.log10(mel_spectrogram + sys.float_info.epsilon)

    # 04 calculate total vector size  309
    vector_array_size = len(log_mel_spectrogram[0, :]) - frames + 1
    # print("vector_array_size:", vector_array_size)
    # 05 skip too short clips
    if vector_array_size < 1:
        return np.empty((0, dims))

    # 06 generate feature vectors by concatenating multiframes
    vector_array = np.zeros((vector_array_size, dims))  # (309, 320)
    for t in range(frames):
        # [0, 64], [64, 128], ..., [256, 320]
        # [0, 309] [1, 310] [2, 311], [3, 312] [4, 313]
        vector_array[:, n_mels * t: n_mels * (t + 1)] = log_mel_spectrogram[:, t: t + vector_array_size].T
    # 309, 320
    return vector_array


__versions__ = "1.0.0"


def command_line_chk():
    parser = argparse.ArgumentParser(description='Without option argument, it will not run properly.')
    parser.add_argument('-v', '--version', action='store_true', help="show application version")
    parser.add_argument('-e', '--eval', action='store_true', help="run mode Evaluation")
    parser.add_argument('-d', '--dev', action='store_true', help="run mode Development")
    args = parser.parse_args()
    if args.version:
        print("===============================")
        print("DCASE 2020 task 2 baseline\nversion {}".format(__versions__))
        print("===============================\n")
    if args.eval ^ args.dev:
        if args.dev:
            flag = True
        else:
            flag = False
    else:
        flag = None
        print("incorrect argument")
        print("please set option argument '--dev' or '--eval'")
    return flag

