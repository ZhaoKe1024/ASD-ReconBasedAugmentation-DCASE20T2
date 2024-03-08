#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2024/1/16 18:41
# @Author: ZhaoKe
# @File : tdnn.py
# @Software: PyCharm
import torch
import torch.nn as nn


class TDNN_Extractor(nn.Module):
    def __init__(self, input_size=80, hidden_size=512, channels=512, embd_dim=192):
        super(TDNN_Extractor, self).__init__()
        self.emb_size = embd_dim
        self.wav2mel = nn.Conv1d(in_channels=1, out_channels=128, kernel_size=1024, stride=488,
                                 padding=1024 // 2, bias=False)
        # self.wav2mel = nn.Conv1d(in_channels=1, out_channels=128, kernel_size=1024, stride=512, padding=1024 // 2, bias=False)
        self.td_layer1 = torch.nn.Conv1d(in_channels=input_size, out_channels=hidden_size, dilation=1, kernel_size=5,
                                         stride=1)  # IW-5+1
        self.bn1 = nn.LayerNorm(302)
        self.td_layer2 = torch.nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, dilation=2, kernel_size=3,
                                         stride=1, groups=hidden_size)  # IW-4+1
        self.bn2 = nn.LayerNorm(298)
        self.td_layer3 = torch.nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, dilation=3, kernel_size=3,
                                         stride=1, groups=hidden_size)  # IW-6+1
        self.bn3 = nn.LayerNorm(294)
        self.td_layer4 = torch.nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, dilation=1, kernel_size=1,
                                         stride=1, groups=hidden_size)  # IW+1
        self.bn4 = nn.LayerNorm(288)
        self.td_layer5 = torch.nn.Conv1d(in_channels=hidden_size, out_channels=channels, dilation=1, kernel_size=1,
                                         stride=1, groups=hidden_size)  # IW+1
        self.leakyrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, waveform):
        # x = x.transpose(2, 1)
        x = self.leakyrelu(self.bn1(self.wav2mel(waveform)))
        # print("shape of x as a wave:", x.shape)
        x = self.leakyrelu(self.bn2(self.td_layer1(x)))
        # print("shape of x in layer 1:", x.shape)
        x = self.leakyrelu(self.bn3(self.td_layer2(x)))
        # print("shape of x in layer 2:", x.shape)
        x = self.leakyrelu(self.bn4(self.td_layer3(x)))
        # print("shape of x in layer 3:", x.shape)
        x = self.leakyrelu(self.bn4(self.td_layer4(x)))
        # print("shape of x in layer 4:", x.shape)
        x = self.td_layer5(x)
        # print("shape of x in layer 5:", x.shape)
        return x


if __name__ == '__main__':
    input_wav = torch.rand(16, 1, 147000)
    tdnn_model = TDNN_Extractor(input_size=128, hidden_size=128, channels=128, embd_dim=128)
    feat = tdnn_model(input_wav)
    print("feat shape:", feat.shape)

    # scale = 2
    # kernel_size = 1024
    # wav2mel = nn.Conv1d(in_channels=1, out_channels=128, kernel_size=kernel_size, stride=488, padding=kernel_size // 2,
    #                     bias=False)
    # print(wav2mel(input_wav).shape)
    # input_mel = torch.rand(16, 128, 288)
    # tdnn_model = TDNN(num_class=23, input_size=128, hidden_size=512, channels=512, embd_dim=128)
    # pred, feat = tdnn_model(wav2mel(input_wav))
