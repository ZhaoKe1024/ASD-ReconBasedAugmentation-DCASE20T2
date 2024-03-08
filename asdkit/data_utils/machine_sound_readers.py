#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2023/12/29 9:59
# @Author: ZhaoKe
# @File : machine_sound_readers.py
# @Software: PyCharm
import os
import random
import json
import numpy as np
import cv2
import librosa
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader

from asdkit.data_utils.audio import wav_slice_padding
from asdkit.data_utils.featurizer import Wave2Mel
from asdkit.data_utils.dcase2020_fileutils import get_train_file_list
# from asdkit.modules.spec_aug import edge_augmented


def get_former_loader(istrain=True, istest=False, configs=None, meta2label=None, isdemo=False, shuffles=(True, False), exclude=None):
    # generate dataset
    # generate dataset
    if shuffles is None:
        shuffles = [True, False]
    print("============== DATASET_GENERATOR ==============")
    ma_id_map = {5: "valve", 4: "slider", 3: "pump", 2: "fan", 1: "ToyConveyor", 0: "ToyCar"}
    train_loader, test_loader = None, None
    if istrain:
        print("---------------train dataset-------------")
        file_paths = []
        mtid_list = []
        mtype_list = []
        with open("../datasets/train_list.txt", 'r') as fin:
            train_path_list = fin.readlines()
            if isdemo:
                train_path_list = random.choices(train_path_list, k=200)
            for item in train_path_list:
                parts = item.strip().split('\t')
                machine_type_id = int(parts[1])
                if exclude:
                    if machine_type_id in exclude:
                        continue
                file_paths.append(parts[0])
                mtype_list.append(machine_type_id)
                machine_id_id = parts[2]
                meta = ma_id_map[machine_type_id] + '-id_' + machine_id_id
                mtid_list.append(meta2label[meta])
        train_dataset = FormerReader(file_paths=file_paths, mtype_list=mtype_list, mtid_list=mtid_list,
                                     y_true_list=None,
                                     configs=configs,
                                     istrain=True, istest=not istest)
        train_loader = DataLoader(train_dataset, batch_size=configs["fit"]["batch_size"], shuffle=shuffles[0])
    if istest:
        print("---------------test dataset-------------")
        file_paths = []
        mtid_list = []
        mtype_list = []
        y_true_list = []
        with open("../datasets/test_list.txt", 'r') as fin:
            test_path_list = fin.readlines()
            if isdemo:
                test_path_list = random.choices(test_path_list, k=20)
            for item in test_path_list:
                parts = item.strip().split('\t')
                machine_type_id = int(parts[1])
                if exclude:
                    if machine_type_id in exclude:
                        continue
                mtype_list.append(machine_type_id)
                file_paths.append(parts[0])
                machine_id_id = parts[2]
                y_true_list.append(int(parts[3]))
                meta = ma_id_map[machine_type_id] + '-id_' + machine_id_id
                mtid_list.append(meta2label[meta])
        test_dataset = FormerReader(file_paths=file_paths, mtype_list=mtype_list, mtid_list=mtid_list,
                                    y_true_list=y_true_list,
                                    configs=configs, istrain=False, istest=not istest)
        test_loader = DataLoader(test_dataset, batch_size=configs["fit"]["batch_size"], shuffle=shuffles[1])
    return train_loader, test_loader


def get_loader_list(train_dirs, m2l_map, configs=None, root="train", mode="train", demo_test=False):
    loaders = []
    id_ma_map = {"valve": 5, "slider": 4, "pump": 3, "fan": 2, "ToyConveyor": 1, "ToyCar": 0}
    ano_id_map = {"normal": 0, "anomal": 1}
    # ma_id_map = {5: "valve", 4: "slider", 3: "pump", 2: "fan", 1: "ToyConveyor", 0: "ToyCar"}
    for machine_dir in train_dirs:
        mt = machine_dir.split('\\')[-1].split('_')[-1]
        machine_root_path = machine_dir + '/' + mt + '/' + root
        file_paths = [machine_root_path + '/' + item for item in os.listdir(machine_root_path)]
        if demo_test:
            file_paths = random.choices(file_paths, k=200)
        mtype_list = [id_ma_map[mt]] * len(file_paths)
        mtid_list = [m2l_map[mt + "-id_" + item.split('_')[2]] for item in os.listdir(machine_root_path)]
        y_true_list = [ano_id_map[item[:6]] for item in os.listdir(machine_root_path)]
        if demo_test:
            mtid_list = random.choices(mtid_list, k=200)

        if mode == "train":
            istrain, istest = True, False
        elif mode == "test":
            istrain, istest = False, False
        else:
            istrain, istest = False, False
        dataset = FormerReader(file_paths=file_paths, mtype_list=mtype_list, mtid_list=mtid_list,
                               y_true_list=y_true_list,
                               configs=configs,
                               istrain=istrain, istest=istest)

        loader = DataLoader(dataset, batch_size=configs["fit"]["batch_size"],
                            shuffle=True)
        loaders.append(loader)

    return loaders


class FormerReader(Dataset):
    def __init__(self, file_paths, mtype_list, mtid_list, y_true_list, configs, istrain=True, istest=False):
        self.files = file_paths
        self.mtids = mtid_list
        self.mtype_list = mtype_list
        self.y_true = y_true_list
        self.configs = configs
        self.w2m = Wave2Mel(16000)
        self.device = torch.device("cuda")
        self.mel_specs = []
        if file_paths:
            for fi in tqdm(file_paths, desc=f"build Set..."):
                self.mel_specs.append(self.load_wav_2mel(fi))
        self.istrain = istrain
        self.istest = istest

    def __getitem__(self, ind):
        if not self.istest:
            if self.istrain:
                # print("***")
                return self.mel_specs[ind], self.mtype_list[ind], self.mtids[ind]
            else:
                # print("????")
                return self.mel_specs[ind], self.mtype_list[ind], self.mtids[ind], self.y_true[ind], self.files[ind]
            # if self.istrain:
            #     return self.mel_specs[ind] / self.mel_specs[ind].abs().max(), self.mtype_list[ind], self.mtids[ind]
            # else:
            #     return self.mel_specs[ind] / self.mel_specs[ind].abs().max(), self.mtype_list[ind], self.mtids[ind], self.y_true[ind], self.files[ind]
        else:
            # print("!!!")
            # print(self.mel_specs[ind].shape, self.mtype_list[ind], self.mtids[ind], self.y_true[ind], self.files[ind])
            return self.mel_specs[ind], self.mtype_list[ind], self.mtids[ind], self.y_true[ind], self.files[ind]
            # return self.mel_specs[ind] / self.mel_specs[ind].abs().max(), self.y_true[ind], self.files[ind]

    def __len__(self):
        return len(self.files)

    def load_wav_2mel(self, wav_path):
        # print(wav_path)
        y, sr = librosa.core.load(wav_path, sr=16000)
        # print("....\n", self.configs)
        y = wav_slice_padding(y, save_len=self.configs["feature"]["wav_length"])
        x_mel = self.w2m(torch.from_numpy(y.T))
        return torch.tensor(x_mel, device=self.device).transpose(0, 1).to(torch.float32)

    def get_mel_from_filepath(self, file_path):
        y, sr = librosa.core.load(file_path, sr=16000)
        # print("....\n", self.configs)
        x_wav = wav_slice_padding(y, save_len=self.configs["feature"]["wav_length"])
        x_mel = self.w2m(torch.from_numpy(x_wav.T))
        return x_wav, x_mel


class ASD_dataset(Dataset):
    def __init__(self, data_path, cfg):
        self.features = []
        for filename in tqdm(os.listdir(data_path)):
            data, sr = librosa.load(data_path + filename, sr=cfg["sr"])
            ft = np.abs(librosa.stft(data, n_fft=cfg["n_fft"],
                                     win_length=cfg["win_length"], hop_length=cfg["hop_length"]))
            mel = librosa.feature.melspectrogram(
                S=ft, sr=cfg["sr"], n_mels=cfg["n_mels"], hop_length=cfg["hop_length"], win_length=cfg["win_length"])

            m_mel = mel.mean(axis=1)
            self.features.append(m_mel)
        # self.x = data

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.features[idx])

        return x

    def __len__(self):
        return len(self.features)


class SpecReader(Dataset):
    def __init__(self, m_type="pump", is_dev=True, train_mode="train"):
        # data_config_path = "../configs/coughvid_dataset.yaml"
        self.file_root, self.wav_name_list = get_train_file_list(m_type=m_type, is_dev=is_dev, train_mode=train_mode)
        self.wav2mel = Wave2Mel(sr=16000)
        self.train_mode = train_mode
        self.is_dev = is_dev
        self.labels = []  # 0 normal 1 anomaly
        if (not is_dev) or (not train_mode == "train"):
            for item in self.wav_name_list:
                self.labels.append(0 if item.split('_')[0] == "normal" else 1)

    def __getitem__(self, item):
        x, _ = librosa.core.load(self.file_root + self.wav_name_list[item], sr=16000, mono=True)
        x_wav = torch.from_numpy(x)
        x_mel = self.wav2mel(x_wav)
        # print("------shape (wav, mel)--------")
        # print(x_wav.shape, x_mel.shape)  # ([wav_len]) ([mel_dim=128, mel_len])
        if self.is_dev:
            if self.train_mode == "train":
                # training dataset, only contains normal sounds.
                return x_mel
            else:
                # test dataset in developed set, contains both normal and anomaly sounds.
                return x_mel, self.labels[item], self.wav_name_list[item]
        else:
            # evaluate dataset, not specified normal or anomaly, need give filename for record.
            # print(self.labels[item])
            # print(self.wav_name_list[item])
            return x_mel, self.wav_name_list[item]
        # return x_wav, x_mel

    def __len__(self):
        return len(self.wav_name_list)


# 这个假定了不存在跨域问题
class SpecAllReader(Dataset):
    def __init__(self, file_paths, mtype_list, mtid_list, y_true_list, configs, istrain=True, istest=False):
        self.files = file_paths
        self.mtids = mtid_list
        self.mtype_list = mtype_list
        self.y_true = y_true_list
        self.configs = configs
        self.w2m = Wave2Mel(16000)
        self.device = torch.device("cuda")
        self.wav_form = []
        self.mel_specs = []
        if file_paths:
            for fi in tqdm(file_paths, desc=f"Building DataSet"):
                self.mel_specs.append(self.load_wav_2mel(fi))
        self.istrain = istrain
        self.istest = istest

    def __getitem__(self, ind):
        if self.istrain:
            if not self.istest:
                # print(self.wav_form[ind], self.mel_specs[ind], self.mtype_list[ind], self.mtids[ind], self.y_true[ind])
                return self.wav_form[ind], self.mel_specs[ind], self.mtype_list[ind], self.mtids[ind], self.y_true[ind]
            else:
                # print("????")
                return self.wav_form[ind], self.mel_specs[ind], self.mtype_list[ind], self.mtids[ind], self.y_true[ind], self.files[ind]
            # if self.istrain:
            #     return self.mel_specs[ind] / self.mel_specs[ind].abs().max(), self.mtype_list[ind], self.mtids[ind]
            # else:
            #     return self.mel_specs[ind] / self.mel_specs[ind].abs().max(), self.mtype_list[ind], self.mtids[ind], self.y_true[ind], self.files[ind]
        else:
            # print("!!!")
            # print(self.mel_specs[ind].shape, self.mtype_list[ind], self.mtids[ind], self.y_true[ind], self.files[ind])
            return self.wav_form[ind], self.mel_specs[ind], self.mtype_list[ind], self.mtids[ind], self.y_true[ind], self.files[ind]
            # return self.mel_specs[ind] / self.mel_specs[ind].abs().max(), self.y_true[ind], self.files[ind]

    def __len__(self):
        return len(self.files)

    def load_wav_2mel(self, wav_path):
        # print(wav_path)
        y, sr = librosa.core.load(wav_path, sr=16000)
        # print("....\n", self.configs)
        y = wav_slice_padding(y, save_len=self.configs["feature"]["wav_length"])
        self.wav_form.append(torch.tensor(y, device=self.device).to(torch.float32))
        x_mel = self.w2m(torch.from_numpy(y.T))
        return torch.tensor(x_mel, device=self.device).transpose(0, 1).to(torch.float32)

    def get_mel_from_filepath(self, file_path):
        y, sr = librosa.core.load(file_path, sr=16000)
        # print("....\n", self.configs)
        x_wav = wav_slice_padding(y, save_len=self.configs["feature"]["wav_length"])
        x_mel = self.w2m(torch.from_numpy(x_wav.T))
        return x_wav, x_mel


def collate_fn_min(batch):
    # wav mel label
    # batch shape: (batch_size, tuple([mel_dim=128, mel_len]))
    # 找出mel长度最长的
    batch = sorted(batch, key=lambda sample: sample[1].shape[1], reverse=False)
    mel_dim, min_mel_length = batch[0][1].shape
    wav_min_len = len(batch[0][0])
    batch_size = len(batch)
    # 以最大的长度创建0张量
    input_mels = np.zeros((batch_size, mel_dim, min_mel_length), dtype='float32')
    input_wavs = np.zeros((batch_size, wav_min_len), dtype='float32')
    labels = []
    input_lens_bias = []
    # labels = []
    for x in range(batch_size):
        sample = batch[x]
        mel_tensor = sample[1]
        wav_tensor = sample[0]
        labels.append(sample[2])
        seq_length = mel_tensor.shape[1]
        # wav_len = len(wav_tensor)
        # 将数据插入都0张量中，实现了padding
        input_mels[x, :, :] = mel_tensor[:, :min_mel_length]
        input_wavs[x, :] = wav_tensor[:wav_min_len]
        input_lens_bias.append(seq_length)
    input_lens_bias = np.array(input_lens_bias, dtype='int64')
    labels = np.array(labels, dtype='float32')
    return torch.tensor(input_wavs), torch.tensor(input_mels), torch.tensor(labels), wav_min_len, min_mel_length


def collate_fn_max(batch):
    # wav mel label
    # batch shape: (batch_size, tuple([mel_dim=128, mel_len]))
    # 找出mel长度最长的
    batch = sorted(batch, key=lambda sample: sample[1].shape[1], reverse=True)
    mel_dim, max_mel_length = batch[0][1].shape
    wav_max_len = len(batch[0][0])
    batch_size = len(batch)
    # 以最大的长度创建0张量
    input_mels = np.zeros((batch_size, mel_dim, max_mel_length), dtype='float32')
    input_wavs = np.zeros((batch_size, wav_max_len), dtype='float32')
    labels = []
    input_lens_bias = []
    # labels = []
    for x in range(batch_size):
        sample = batch[x]
        mel_tensor = sample[1]
        wav_tensor = sample[0]
        labels.append(sample[2])
        seq_length = mel_tensor.shape[1]
        wav_len = len(wav_tensor)
        # 将数据插入都0张量中，实现了padding
        input_mels[x, :, :seq_length] = mel_tensor[:]
        input_wavs[x, :wav_len] = wav_tensor[:]
        input_lens_bias.append(seq_length)
    input_lens_bias = np.array(input_lens_bias, dtype='int64')
    labels = np.array(labels, dtype='float32')
    return torch.tensor(input_wavs), torch.tensor(input_mels), torch.tensor(labels), wav_max_len, max_mel_length


def collate_fn_img(batch):
    # batch shape: (batch_size, tuple([mel_dim=128, mel_len]))
    # 找出mel长度最长的
    print(len(batch))
    print(batch[0].shape)
    batch = sorted(batch, key=lambda sample: sample[0].shape[1], reverse=True)
    mel_dim, max_mel_length = batch[0][0].shape
    batch_size = len(batch)
    # 以最大的长度创建0张量
    inputs = np.zeros((batch_size, mel_dim, max_mel_length), dtype='float32')
    labels = []
    input_lens_bias = []
    # labels = []
    for x in range(batch_size):
        sample = batch[x]
        tensor = sample[0]
        labels.append(sample[1])
        seq_length = tensor.shape[1]
        # 将数据插入都0张量中，实现了padding
        inputs[x, :, :seq_length] = tensor[:]
        input_lens_bias.append(seq_length)
    input_lens_bias = np.array(input_lens_bias, dtype='int64')
    labels = np.array(labels, dtype='float32')
    return torch.tensor(inputs), torch.tensor(labels), max_mel_length


def test_show_spec_dcase2020():
    # ms_reader = SpecReader("pump", True, "train")
    ms_reader = SpecReader("pump", False, "eval")
    idx, cnt = 0, 4
    import matplotlib.pyplot as plt
    plt.figure(0)
    for (img, ano, filename) in ms_reader:
        print(filename, ano, img.shape)
        if idx == cnt:
            break
        plt.subplot(4, 1, idx + 1)
        plt.imshow(img)
        idx += 1
    plt.show()


def get_a_wavmel_sample(test_wav_path):
    # test_wav_path = f"G:/DATAS-DCASE-ASD/DCASE2020Task2ASD/dataset/dev_data_{mt}/{mt}/train/normal_id_00_00000016.wav"
    y, sr = librosa.core.load(test_wav_path, sr=16000)
    y = wav_slice_padding(y, 147000)
    w2m = Wave2Mel(16000)
    x_mel = w2m(torch.from_numpy(y.T))
    x_input = x_mel.unsqueeze(0).unsqueeze(0).to(torch.device("cuda")).transpose(2, 3)
    return y, x_input


if __name__ == '__main__':
    test_show_spec_dcase2020()
