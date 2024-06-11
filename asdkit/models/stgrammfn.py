#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2023/9/21 17:03
# @Author: ZhaoKe
# @File : stgrammfn.py
# @Software: PyCharm
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import Parameter

from asdkit.models.tdnn import TDNN_Extractor
from asdkit.modules.loss import ArcMarginProduct
from asdkit.modules.mobilefacenet import MobileFaceNet


# original mobilefacenet setting
# Mobilefacenet_bottleneck_setting = [
#     # t, c , n ,s
#     [2, 64, 5, 2],
#     [4, 128, 1, 2],
#     [2, 128, 6, 1],
#     [4, 128, 1, 2],
#     [2, 128, 2, 1]
# ]

# Mobilenetv2_bottleneck_setting = [
#     # t, c, n, s
#     [1, 16, 1, 1],
#     [6, 24, 2, 2],
#     [6, 32, 3, 2],
#     [6, 64, 4, 2],
#     [6, 96, 3, 1],
#     [6, 160, 3, 2],
#     [6, 320, 1, 1],
# ]

# refer to DCASE2022 Task2 Top-1
# https://dcase.community/documents/challenge2022/technical_reports/DCASE2022_Liu_8_t2.pdf
Mobilefacenet_bottleneck_setting = [
    # t, c , n ,s
    [2, 128, 2, 2],
    [4, 128, 2, 2],
    [4, 128, 2, 2],
]


class TgramNet(nn.Module):
    def __init__(self, num_layer=3, mel_bins=128, win_len=1024, hop_len=512):
        super(TgramNet, self).__init__()
        # if "center=True" of stft, padding = win_len / 2
        self.conv_extrctor = nn.Conv1d(1, mel_bins, win_len, hop_len, win_len // 2, bias=False)
        self.conv_encoder = nn.Sequential(
            *[nn.Sequential(
                # 313(10) , 63(2), 126(4)
                nn.LayerNorm(288),  # 313, but maxlen 344
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv1d(mel_bins, mel_bins, 3, 1, 1, bias=False),
            ) for idx in range(num_layer)])

    def forward(self, x):
        out = self.conv_extrctor(x)
        out = self.conv_encoder(out)
        return out


class STgramMFN(nn.Module):
    def __init__(self, num_classes,
                 c_dim=128,
                 win_len=1024,
                 hop_len=512,
                 bottleneck_setting=Mobilefacenet_bottleneck_setting,
                 use_arcface=False, m=0.5, s=30, sub=1):
        super(STgramMFN, self).__init__()
        # self.arcface = ArcMarginProduct(in_features=128, out_features=num_classes,
        #                                 m=m, s=s, sub=sub) if use_arcface else use_arcface
        self.tgramnet = TDNN_Extractor(input_size=128, hidden_size=128, channels=128, embd_dim=128)
        # self.tgramnet = TgramNet(mel_bins=c_dim, win_len=win_len, hop_len=hop_len)
        self.mobilefacenet = MobileFaceNet(inp_c=2, num_class=num_classes,
                                           bottleneck_setting=bottleneck_setting, inp=2)

    def get_tgram(self, x_wav):
        return self.tgramnet(x_wav)

    def forward(self, x_wav, x_mel, label=None):
        # print("shape wav mel label:", x_wav.shape, x_mel.shape, label.shape)
        # torch.Size([1, 160000]) torch.Size([1, 128, 344]) torch.Size([1])
        # shape wav mel label: torch.Size([1, 147000]) torch.Size([1, 128, 288]) torch.Size([1])
        # shape wav mel label: torch.Size([1, 147000]) torch.Size([1, 128, 288]) torch.Size([1])
        # print("shape:", x_wav.shape, x_mel.shape)
        x_wav, x_mel = x_wav.unsqueeze(1), x_mel.unsqueeze(1)
        x_t = self.tgramnet(x_wav).unsqueeze(1)
        # print("shape:", x_t.shape, x_mel.shape)
        x = torch.cat((x_mel, x_t), dim=1)
        # print("shape:", x.shape)
        out, feature = self.mobilefacenet(x, label)
        # if self.arcface:
        #     out = self.arcface(feature, label)
        return out, feature


if __name__ == '__main__':
    net = STgramMFN(num_classes=23, use_arcface=True)
    x_wav = torch.rand(2, 147000)
    x_mel = torch.rand(2, 128, 288)
    # labels = torch.randint(0, 23, size=(1,))
    labels = torch.tensor([3, 4]).long()
    pred, feat = net(x_wav, x_mel, label=labels)
    print("shape:", pred.shape, feat.shape)
