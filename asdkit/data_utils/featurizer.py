#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2023/9/21 16:54
# @Author: ZhaoKe
# @File : featurizer.py
# @Software: PyCharm
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
import librosa
import torch
from torch import nn
import torchaudio
import torchaudio.compliance.kaldi as Kaldi
from torchaudio.transforms import MelSpectrogram, Spectrogram, MFCC


class Wave2Mel(object):
    def __init__(self, sr,
                 n_fft=1024,
                 n_mels=128,
                 win_length=1024,
                 hop_length=512,
                 power=2.0
                 ):
        self.mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=sr,
                                                                  win_length=win_length,
                                                                  hop_length=hop_length,
                                                                  n_fft=n_fft,
                                                                  n_mels=n_mels,
                                                                  power=power)
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype='power')

    def __call__(self, x):
        return self.amplitude_to_db(self.mel_transform(x))


class AudioFeaturizer(nn.Module):
    """音频特征器

    :param feature_method: 所使用的预处理方法
    :type feature_method: str
    :param method_args: 预处理方法的参数
    :type method_args: dict
    """

    def __init__(self, feature_method='MelSpectrogram', method_args={}):
        super().__init__()
        self._method_args = method_args
        self._feature_method = feature_method
        if feature_method == 'MelSpectrogram':
            self.feat_fun = MelSpectrogram(**method_args)
        elif feature_method == 'Spectrogram':
            self.feat_fun = Spectrogram(**method_args)
        elif feature_method == 'MFCC':
            self.feat_fun = MFCC(**method_args)
        elif feature_method == 'Fbank':
            self.feat_fun = KaldiFbank(**method_args)
        else:
            raise Exception(f'预处理方法 {self._feature_method} 不存在!')

    def forward(self, waveforms, input_lens_ratio):
        """从AudioSegment中提取音频特征

        :param waveforms: Audio segment to extract features from.
        :type waveforms: AudioSegment
        :param input_lens_ratio: input length ratio
        :type input_lens_ratio: tensor
        :return: Spectrogram audio feature in 2darray.
        :rtype: ndarray
        """
        feature = self.feat_fun(waveforms)
        feature = feature.transpose(2, 1)
        # 归一化
        feature = feature - feature.mean(1, keepdim=True)
        # 对掩码比例进行扩展
        input_lens = (input_lens_ratio * feature.shape[1])
        mask_lens = torch.round(input_lens).long()
        mask_lens = mask_lens.unsqueeze(1)
        input_lens = input_lens.int()
        # 生成掩码张量
        idxs = torch.arange(feature.shape[1], device=feature.device).repeat(feature.shape[0], 1)
        mask = idxs < mask_lens
        mask = mask.unsqueeze(-1)
        # 对特征进行掩码操作
        feature_masked = torch.where(mask, feature, torch.zeros_like(feature))
        return feature_masked, input_lens

    @property
    def feature_dim(self):
        """返回特征大小

        :return: 特征大小
        :rtype: int
        """
        if self._feature_method == 'MelSpectrogram':
            return self._method_args.get('n_mels', 128)
        elif self._feature_method == 'Spectrogram':
            return self._method_args.get('n_fft', 400) // 2 + 1
        elif self._feature_method == 'MFCC':
            return self._method_args.get('n_mfcc', 40)
        elif self._feature_method == 'Fbank':
            return self._method_args.get('num_mel_bins', 23)
        else:
            raise Exception('没有{}预处理方法'.format(self._feature_method))


class KaldiFbank(nn.Module):
    def __init__(self, **kwargs):
        super(KaldiFbank, self).__init__()
        self.kwargs = kwargs

    def forward(self, waveforms):
        """
        :param waveforms: [Batch, Length]
        :return: [Batch, Length, Feature]
        """
        log_fbanks = []
        for waveform in waveforms:
            if len(waveform.shape) == 1:
                waveform = waveform.unsqueeze(0)
            log_fbank = Kaldi.fbank(waveform, **self.kwargs)
            log_fbank = log_fbank.transpose(0, 1)
            log_fbanks.append(log_fbank)
        log_fbank = torch.stack(log_fbanks)
        return log_fbank


def wav2mel(file_name):
    # # DCASE2020里面的做法
    # spec =  self.amplitude_to_db(self.mel_transform(x)).squeeze().transpose(-1,-2)
    # 02 generate melspectrogram using librosa
    y, sr = librosa.load(file_name, sr=None, mono=False)
    mel_spectrogram = librosa.feature.melspectrogram(y=y,
                                                     sr=sr,
                                                     n_fft=1024,
                                                     hop_length=512,
                                                     n_mels=128,
                                                     power=2.0)
    # print("shape of mel_spectrogram:", mel_spectrogram.shape)
    # 03 convert melspectrogram to log mel energy (64, 313)
    power = 2.0
    log_mel_spectrogram = 20.0 / power * np.log10(mel_spectrogram + sys.float_info.epsilon)
    return log_mel_spectrogram


def read_spect_opts(root_path, target_path, dataset_mode="", opt_name=None):
    opt = opt_name
    dataset_path = root_path + dataset_mode + '/'
    for item in os.listdir(dataset_path):
        img = cv2.imread(dataset_path+item)
        # print(img.shape)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = None
        kernel = np.ones((3, 3), np.uint8)
        if opt == 'erosion':
            res = cv2.erode(img, kernel, iterations=1)
        if opt == 'dilation':
            res = cv2.dilate(img, kernel, iterations=1)
        if opt == 'gradient':
            res = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
        if opt == 'opening':
            res = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        if opt == 'closing':
            res = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        if opt == 'top_hat':
            res = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
        if opt == 'black_hat':
            res = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
        # res = cv2.erode(img, kernel, iterations=1)
        # res = cv2.dilate(img, kernel, iterations=1)
        plt.figure(0)
        plt.imshow(res)
        os.makedirs(target_path + dataset_mode + '/', exist_ok=True)
        # plt.savefig(f"../useless_programs/imgs/{opt}-asdvalve09-" + time.strftime("%Y%m%d%H%M", time.localtime()))
        plt.savefig(target_path + '/' + dataset_mode + '/' + item + ".png")
        plt.close()


def bgr2rgb_test():
    machine_name = "gearbox"
    ms_reader = f"G:/DATAS-DCASE-ASD/DCASE2023Task2ASD-spec/dev_{machine_name}/{machine_name}/train/"
    img_path_list = [
        "section_00_source_train_normal_0014_volt_3.0_wt_0.png",
        "section_00_source_train_normal_0073_volt_1.5_wt_0.png"
    ]
    plt.figure(0)
    idx = 0
    for img_path in img_path_list:
        pa_str = ms_reader + img_path
        # print(pa_str)
        img = cv2.imread(pa_str)
        r, g, b = cv2.split(img)
        img2 = cv2.merge([b, g, r])
        idx += 1
        plt.subplot(2, 2, idx)
        plt.imshow(img)
        idx += 1
        plt.subplot(2, 2, idx)
        plt.imshow(img2)
    plt.show()


def img_trans(img_path, opt="opening"):
    w2m = Wave2Mel(sr=16000)
    x, sr = librosa.core.load(img_path, sr=16000, mono=True)
    x = torch.from_numpy(x)
    img = w2m(x)
    print(img.shape)
    img = np.asarray(img.data.numpy(), dtype=np.uint8)
    # img = wav2mel(img_path)
    # img = np.asarray(img, dtype=np.uint8)
    # print(img.shape)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = None
    kernel = np.ones((3, 3), np.uint8)
    if opt == 'erosion':
        res = cv2.erode(img, kernel, iterations=1)
    if opt == 'dilation':
        res = cv2.dilate(img, kernel, iterations=1)
    if opt == 'gradient':
        res = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
    if opt == 'opening':
        res = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    if opt == 'closing':
        res = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    if opt == 'top_hat':
        res = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
    if opt == 'black_hat':
        res = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
    # res = cv2.erode(img, kernel, iterations=1)
    # res = cv2.dilate(img, kernel, iterations=1)

    plt.figure(0)
    plt.subplot(2,1,1)
    plt.imshow(img)
    plt.subplot(2,1,2)
    plt.imshow(res)
    plt.show()


if __name__ == '__main__':
    # opt_name = "closing"
    # machine_name = ["fan", "pump", "slider", "valve", "ToyCar", "ToyConveyor"]
    # for name in machine_name:
    #     root_path = f"G:/DATAS-mini/dcase2020-asd-task2-specbgr/data/dataset/{name}/"
    #     target_path = f"G:/DATAS-mini/dcase2020-asd-task2-{opt_name}/{name}/"
    #     read_spect_opts(root_path, target_path, "test", opt_name)
    mt = ["fan", "pump", "slider", "valve", "ToyCar", "ToyConveyor"]
    # test_img_path = f"G:/DATAS-DCASE-ASD/DCASE2020Task2ASD/dataset/dev_data_{mt[0]}/{mt[0]}/train/normal_id_00_00000004.wav"
    test_img_path = f"G:/DATAS-DCASE-ASD/DCASE2020Task2ASD/dataset/dev_data_{mt[0]}/{mt[0]}/test/normal_id_00_00000004.wav"
    img_trans(test_img_path, opt="erosion")
    # bgr2rgb_test()

