#!/user/zhao/miniconda3/envs/torch-0
# -*- coding: utf_8 -*-
# @Time : 2023/10/25 9:43
# @Author: ZhaoKe
# @File : autoencoder.py
# @Software: PyCharm
import torch
import torch.nn as nn

from asdkit.modules.loss import ArcMarginProduct


class ConvEncoder(nn.Module):
    def __init__(self, input_channel=1, input_length=288, input_dim=128, class_num=23, class_num1=6):
        super(ConvEncoder, self).__init__()
        self.input_dim = input_channel
        self.max_cha = 256
        es = [input_channel, 32, 64, 128, self.max_cha]  # , 128]
        self.encoder_layers = nn.Sequential()
        kernel_size, stride, padding = 4, 2, 1
        for i in range(len(es) - 2):
            self.encoder_layers.append(nn.Conv2d(es[i], es[i + 1], kernel_size=kernel_size, stride=stride, padding=padding, bias=False))
            self.encoder_layers.append(nn.LayerNorm((es[i + 1], input_length // (2 ** (i + 1)), input_dim // (2 ** (i + 1)))))
            self.encoder_layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.encoder_layers.append(nn.Conv2d(es[-2], es[-1], kernel_size=kernel_size, stride=1, padding=0, bias=False))
        # z_len = input_length // 8
        # z_dim = input_dim // 8
        self.class_num23 = class_num
        self.arcface = ArcMarginProduct(in_features=input_dim * 2, out_features=class_num)
        # self.fc_out = nn.Linear(in_features=input_dim * 2, out_features=class_num1)

    def forward(self, input_mel, class_vec, coarse_cls=False, fine_cls=True):
        z = self.encoder_layers(input_mel)
        if fine_cls:
            latent_pred = self.arcface(z.mean(axis=3).mean(axis=2), class_vec)
            if coarse_cls:
                coarse_pred = self.fc_out(z.mean(axis=3).mean(axis=2))
                return z, coarse_pred, latent_pred
            return z, latent_pred
        elif coarse_cls:
            coarse_pred = self.fc_out(z.mean(axis=3).mean(axis=2))
            return z, coarse_pred
        else:
            return z


class ConvDecoder(nn.Module):
    def __init__(self, input_channel=1, input_length=288, input_dim=128):
        super(ConvDecoder, self).__init__()
        self.max_cha = 256
        kernel_size, stride, padding = 4, 2, 1
        z_len = input_length // 8
        z_dim = input_dim // 8
        ds = [self.max_cha, 128, 64, 32, input_channel]

        self.decoder_layers = nn.Sequential()
        self.decoder_layers.append(
            nn.ConvTranspose2d(ds[0], ds[1], kernel_size=kernel_size, stride=1, padding=0, bias=False))
        self.decoder_layers.append(nn.LayerNorm((ds[1], z_len, z_dim)))
        self.decoder_layers.append(nn.ReLU(inplace=True))
        for i in range(1, len(ds) - 2):
            self.decoder_layers.append(
                nn.ConvTranspose2d(ds[i], ds[i + 1], kernel_size=kernel_size, stride=stride, padding=padding,
                                   bias=False))
            self.decoder_layers.append(nn.LayerNorm((ds[i + 1], z_len * 2 ** i, z_dim * 2 ** i)))
            self.decoder_layers.append(nn.ReLU(inplace=True))
        self.decoder_layers.append(
            nn.ConvTranspose2d(ds[-2], ds[-1], kernel_size=kernel_size, stride=stride, padding=padding, bias=False))
        self.decoder_layers.append(nn.Tanh())

    def forward(self, latent_map):
        """
        :param latent_map: shape (33, 13)
        :return:
        """
        d = self.decoder_layers(latent_map)
        return d


class ConvAENet(nn.Module):
    def __init__(self, input_channel=1, input_length=288, input_dim=128, class_num=23):
        super(ConvAENet, self).__init__()
        self.input_dim = input_channel
        # self.cov_source = nn.Parameter(torch.zeros(block_size, block_size), requires_grad=False)
        # self.cov_target = nn.Parameter(torch.zeros(block_size, block_size), requires_grad=False)
        self.max_cha = 256
        es = [input_channel, 32, 64, 128, self.max_cha]  # , 128]
        ds = [self.max_cha, 128, 64, 32, input_channel]

        # 这里导致了无法放到cuda上面，不必要这么做
        # self.encoder_layers = []
        # self.decoder_layers = []
        # 这里导致了没有forward函数的报错
        # self.encoder_layers = nn.ModuleList()
        # self.decoder_layers = nn.ModuleList()
        # 正确实现
        self.encoder_layers = nn.Sequential()
        self.decoder_layers = nn.Sequential()
        kernel_size, stride, padding = 4, 2, 1

        for i in range(len(es) - 2):
            self.encoder_layers.append(
                nn.Conv2d(es[i], es[i + 1], kernel_size=kernel_size, stride=stride, padding=padding, bias=False))
            self.encoder_layers.append(nn.LayerNorm((es[i + 1], input_length // (2 ** (i + 1)), input_dim // (2 ** (i + 1)))))
            self.encoder_layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.encoder_layers.append(nn.Conv2d(es[-2], es[-1], kernel_size=kernel_size, stride=1, padding=0, bias=False))
        z_len = input_length // 8
        z_dim = input_dim // 8
        self.decoder_layers.append(
            nn.ConvTranspose2d(ds[0], ds[1], kernel_size=kernel_size, stride=1, padding=0, bias=False))
        self.decoder_layers.append(nn.LayerNorm((ds[1], z_len, z_dim)))
        self.decoder_layers.append(nn.ReLU(inplace=True))
        for i in range(1, len(ds) - 2):
            self.decoder_layers.append(
                nn.ConvTranspose2d(ds[i], ds[i + 1], kernel_size=kernel_size, stride=stride, padding=padding,
                                   bias=False))
            self.decoder_layers.append(nn.LayerNorm((ds[i + 1], z_len*2 ** i, z_dim*2 **i)))
            self.decoder_layers.append(nn.ReLU(inplace=True))
        self.decoder_layers.append(
            nn.ConvTranspose2d(ds[-2], ds[-1], kernel_size=kernel_size, stride=stride, padding=padding, bias=False))
        self.decoder_layers.append(nn.Tanh())
        self.class_num23 = class_num
        self.arcface = ArcMarginProduct(in_features=input_dim*2, out_features=class_num)

    def forward(self, x, frozen_encoder=False, ifcls=False):
        # for m in self.encoder_layers:
        #     x = m(x)
        #     print("shape of x:", x.shape)
        # z = x
        # for m in self.decoder_layers:
        #     z = m(z)
        #     print("shape of z:", z.shape)
        # if frozen_encoder:
            # with no.g
        z = self.encoder_layers(x)
        d = self.decoder_layers(z)
        if ifcls:
            latent_pred = self.arcface1(z.mean(axis=3).mean(axis=2), self.class_num23)
            return d, z, latent_pred
        else:
            return d, z


class ConvLinearAENet(nn.Module):
    """reference from https://github.com/rajanisamir/anomaly_usd/blob/main/cnn_autoencoder.py"""

    def __init__(self, input_dim, mel_shape=None, class_num=23):
        super(ConvLinearAENet, self).__init__()
        self.input_dim = input_dim
        # self.cov_source = nn.Parameter(torch.zeros(block_size, block_size), requires_grad=False)
        # self.cov_target = nn.Parameter(torch.zeros(block_size, block_size), requires_grad=False)
        max_cha = 128
        es = [input_dim, 16, 32, 64, max_cha]  # , 128]
        ds = [max_cha, 64, 32, 16, input_dim]

        # 这里导致了无法放到cuda上面，不必要这么做
        # self.encoder_layers = []
        # self.decoder_layers = []
        self.encoder_layers = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()
        kernel_size, stride, padding = 3, 1, 2
        for i in range(len(es) - 1):
            self.encoder_layers.append(nn.Conv2d(es[i], es[i + 1], kernel_size=3, stride=stride, padding=padding))

        for i in range(len(ds) - 1):
            self.decoder_layers.append(
                nn.ConvTranspose2d(ds[i], ds[i + 1], kernel_size=3, stride=stride, padding=padding))

        self.maxpool = nn.MaxPool2d(kernel_size=2, return_indices=True)
        self.maxunpool = nn.MaxUnpool2d(kernel_size=2)

        # self.relu = nn.ReLU()
        # self.tanh = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.tanh = nn.Tanh()
        self.flatten = nn.Flatten()
        lin_h, lin_w = mel_shape[0], mel_shape[1]
        for i in range(len(es) - 1):
            lin_h = ((lin_h - kernel_size + 2 * padding) // stride + 1) // 2
            lin_w = ((lin_w - kernel_size + 2 * padding) // stride + 1) // 2
            # print("shape of linear input", lin_h, lin_w)
        self.linear1 = nn.Linear(max_cha * lin_h * lin_w, class_num)
        # self.softmax = nn.Softmax(dim=1)
        self.linear2 = nn.Linear(class_num, max_cha * lin_h * lin_w)
        self.unflatten = nn.Unflatten(1, (max_cha, lin_h, lin_w))

    def forward(self, x):
        # print("shape of input:", x.shape)
        sizes = []
        indices = []
        encoder_out = x
        for i in range(len(self.encoder_layers)):
            # print("cur", i, end=', ')
            encoder_out = self.encoder_layers[i](encoder_out)
            encoder_out = self.tanh(encoder_out)
            sizes.append(encoder_out.size())
            encoder_out, indices_tmp = self.maxpool(encoder_out)
            indices.append(indices_tmp)
        #     print("shape of cur encoder layer:", encoder_out.shape)
        # print("shape after encoder:", encoder_out.shape)

        encoder_out = self.flatten(encoder_out)
        # print("shape after flatten:", encoder_out.shape)
        encoder_out = self.linear1(encoder_out)
        # encoder_out = self.softmax(encoder_out)
        # shape of input: torch.Size([32, 1, 340, 639])
        # cur 0, shape of cur encoder layer: torch.Size([32, 16, 84, 159])
        # cur 1, shape of cur encoder layer: torch.Size([32, 32, 20, 39])
        # cur 2, shape of cur encoder layer: torch.Size([32, 64, 4, 9])

        # shape after encoder: torch.Size([32, 64, 4, 9])
        # shape after flatten: torch.Size([32, 2304])

        # decoder
        decoder_out = self.linear2(encoder_out)
        # print("shape after decoder_out linear", decoder_out.shape)
        decoder_out = self.unflatten(decoder_out)
        # print("shape after unflatten:", decoder_out.shape)
        layer_num = len(self.encoder_layers)
        for i in range(len(self.decoder_layers)):
            # print("cur", i, end=', ')
            decoder_out = self.maxunpool(decoder_out, indices=indices[layer_num - 1 - i],
                                         output_size=sizes[layer_num - 1 - i])
            decoder_out = self.tanh(decoder_out)
            decoder_out = self.decoder_layers[i](decoder_out)
        return decoder_out, encoder_out


class LinearAENet(nn.Module):
    def __init__(self, input_dim, block_size=128):
        super(LinearAENet, self).__init__()
        self.input_dim = input_dim
        self.cov = nn.Parameter(torch.zeros(block_size, block_size), requires_grad=False)
        # self.cov_source = nn.Parameter(torch.zeros(block_size, block_size), requires_grad=False)
        # self.cov_target = nn.Parameter(torch.zeros(block_size, block_size), requires_grad=False)

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.BatchNorm1d(128, momentum=0.01, eps=1e-03),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.BatchNorm1d(128, momentum=0.01, eps=1e-03),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.BatchNorm1d(128, momentum=0.01, eps=1e-03),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.BatchNorm1d(128, momentum=0.01, eps=1e-03),
            nn.ReLU(),

            nn.Linear(128, 8),
            nn.BatchNorm1d(8, momentum=0.01, eps=1e-03),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(8, 128),
            nn.BatchNorm1d(128, momentum=0.01, eps=1e-03),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.BatchNorm1d(128, momentum=0.01, eps=1e-03),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.BatchNorm1d(128, momentum=0.01, eps=1e-03),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.BatchNorm1d(128, momentum=0.01, eps=1e-03),
            nn.ReLU(),

            nn.Linear(128, self.input_dim)
        )

    def forward(self, x):
        # z = self.encoder(x.reshape(-1, self.input_dim))
        z = self.encoder(x.view(-1, self.input_dim))
        print(z.shape)
        return self.decoder(z), z


def find_suit_size():
    k, s = 3, 2

    for h in range(300, 341):
        x = ((h - k) / 2 + 1) / 2
        print(f"h: {h}, after: {x}")
        x = ((x - k) / 2 + 1) / 2
        print(f"h: {h}, after: {x}")
        x = ((x - k) / 2 + 1) / 2
        print(f"h: {h}, after: {x}")


def testConvAE():
    batch_size = 64
    # cae_model = ConvAENet(input_channel=1, input_length=288, input_dim=128)  # , mel_shape=[309, 640])
    enc_model = ConvEncoder(input_channel=1, input_length=288, input_dim=128, class_num=23)
    # dec_model = ConvDecoder(input_channel=1, input_length=288, input_dim=128)
    from asdkit.modules.func import weight_init
    # cae_model.apply(weight_init)
    # print(cae_model)
    x = torch.randn(batch_size, 288, 128)
    # model = ConvAENet(input_dim=1)
    # # x = torch.randn(batch_size, 340, 640)
    print("shape of input:", x.unsqueeze(1).shape)
    # dout, eout = cae_model(x.unsqueeze(1))
    # print("output:")
    # print(dout.shape)
    # print("latent feature")
    # print(eout.shape)
    eout = enc_model(x.unsqueeze(1), class_vec=None, coarse_cls=False, fine_cls=False)
    # dout = dec_model(eout)
    # print("output:")
    # print(dout.shape)
    print("latent feature")
    print(eout.shape)  # [64, 256, 33, 13]
    # # find_suit_size()


if __name__ == '__main__':
    model = LinearAENet(128)
    x_input = torch.rand(32, 1, 288, 128)
    recon, z = model(x_input)
    print(recon.shape, z.shape)

