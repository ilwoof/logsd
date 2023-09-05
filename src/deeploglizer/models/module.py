#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 17/05/2023 8:16 pm

@author : Yongzheng Xie
@email : ilwoof@gmail.com
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    """
    model_type(str): 'cnn'
    """
    def __init__(self,
                 encoder_type,
                 decoder_type,
                 pooling_mode,
                 feature_transpose=False,
                 window_size=500,
                 input_dim=128,
                 hidden_dim=64,
                 kernel_sizes=[2, 3, 4],
                 num_layers=1,
                 ):
        super(Encoder, self).__init__()
        self.encoder_type = encoder_type
        self.decoder_type = decoder_type
        self.kernel_sizes = kernel_sizes
        self.act_function = nn.LeakyReLU(inplace=True, negative_slope=2.5e-1)
        self.pooling_mode = pooling_mode
        self.feature_transpose = feature_transpose
        self.window_size = window_size

        if encoder_type == 'cnn':
            self.convs = nn.ModuleList([nn.Conv2d(1, hidden_dim, (K, input_dim)) for K in kernel_sizes])
            self.norm1d = nn.BatchNorm1d(hidden_dim)
            self.norm2d = nn.BatchNorm2d(hidden_dim)
        else:
            RuntimeError(f"The specified model is not supported")

    def forward(self, x):
        if self.encoder_type == 'cnn':
            output = [conv(F.pad(x[:, i, :, :].unsqueeze(1), (0, 0, 0, K - 1))).squeeze(3) for i, (conv, K) in enumerate(zip(self.convs, self.kernel_sizes))]
            if self.pooling_mode == 'max':
                pooled_z = torch.cat([F.max_pool1d(l, l.size(2)).squeeze(2) for l in output], 1)
            elif self.pooling_mode == 'mean':
                pooled_z = torch.cat([F.avg_pool1d(l, l.size(2)).squeeze(2) for l in output], 1)
        else:
            raise RuntimeError("The specified the model type is not supported!")

        return output, x, pooled_z


class Decoder(nn.Module):
    def __init__(self,
                 encoder_type='cnn',
                 decoder_type='deconv',
                 feature_transpose=False,
                 window_size=500,
                 input_dim=128,
                 output_dim=32,
                 kernel_sizes=[2, 3, 4],
                 num_layers=1,
                 ):
        super(Decoder, self).__init__()

        self.encoder_type = encoder_type
        self.decoder_type = decoder_type
        self.act_function = nn.ReLU()
        self.feature_transpose = feature_transpose
        self.window_size = window_size
        self.kernel_sizes = kernel_sizes

        if encoder_type == 'cnn':
            self.de_convs = nn.ModuleList([nn.ConvTranspose2d(input_dim, 1, (K, output_dim)) for K in kernel_sizes])
        else:
            RuntimeError(f"The specified model is not supported")

    def forward(self, x):
        if self.encoder_type == 'cnn':
            z_internal = [de_conv(F.pad(x[i].unsqueeze(3), (0, 0, 0, K - 1))) for i, (de_conv, K) in enumerate(zip(self.de_convs, self.kernel_sizes))]
            z_internal = [z[:, :, :self.window_size, :] for z in z_internal]
            output = torch.cat(z_internal, 1)
        else:
            raise RuntimeError("The specified the model type is not supported!")

        return output
