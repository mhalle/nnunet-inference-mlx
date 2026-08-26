"""SynthStrip's network, vendored verbatim from FreeSurfer's ``mri_synthstrip``
(StripModel + ConvBlock). A plain 3D UNet - torch + numpy only, no surfa / no
FreeSurfer. Kept in its own module so :mod:`nnseg.engines.synthstrip` imports
without torch (the model is loaded lazily). The bundled ``synthstrip.1.pt`` is a
``model_state_dict`` that loads into ``StripModel()`` with zero missing keys
(verified in the spike, 2026-08-26)."""
import numpy as np
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, ndims, in_channels, out_channels, stride=1, activation="leaky"):
        super().__init__()
        Conv = getattr(nn, "Conv%dd" % ndims)
        self.conv = Conv(in_channels, out_channels, 3, stride, 1)
        if activation == "leaky":
            self.activation = nn.LeakyReLU(0.2)
        elif activation is None:
            self.activation = None
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def forward(self, x):
        out = self.conv(x)
        if self.activation is not None:
            out = self.activation(out)
        return out


class StripModel(nn.Module):
    def __init__(self, nb_features=16, nb_levels=7, feat_mult=2, max_features=64,
                 nb_conv_per_level=2, max_pool=2, return_mask=False):
        super().__init__()
        ndims = 3
        feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
        feats = np.clip(feats, 1, max_features)
        nb_features = [np.repeat(feats[:-1], nb_conv_per_level),
                       np.repeat(np.flip(feats), nb_conv_per_level)]
        enc_nf, dec_nf = nb_features
        nb_dec_convs = len(enc_nf)
        final_convs = dec_nf[nb_dec_convs:]
        dec_nf = dec_nf[:nb_dec_convs]
        self.nb_levels = int(nb_dec_convs / nb_conv_per_level) + 1
        max_pool = [max_pool] * self.nb_levels
        MaxPooling = getattr(nn, "MaxPool%dd" % ndims)
        self.pooling = [MaxPooling(s) for s in max_pool]
        self.upsampling = [nn.Upsample(scale_factor=s, mode="nearest") for s in max_pool]
        prev_nf = 1
        encoder_nfs = [prev_nf]
        self.encoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = enc_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf)); prev_nf = nf
            self.encoder.append(convs); encoder_nfs.append(prev_nf)
        encoder_nfs = np.flip(encoder_nfs)
        self.decoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = dec_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf)); prev_nf = nf
            self.decoder.append(convs)
            if level < (self.nb_levels - 1):
                prev_nf += encoder_nfs[level]
        self.remaining = nn.ModuleList()
        for num, nf in enumerate(final_convs):
            self.remaining.append(ConvBlock(ndims, prev_nf, nf)); prev_nf = nf
        if return_mask:
            self.remaining.append(ConvBlock(ndims, prev_nf, 2, activation=None))
            self.remaining.append(nn.Softmax(dim=1))
        else:
            self.remaining.append(ConvBlock(ndims, prev_nf, 1, activation=None))

    def forward(self, x):
        x_history = [x]
        for level, convs in enumerate(self.encoder):
            for conv in convs:
                x = conv(x)
            x_history.append(x)
            x = self.pooling[level](x)
        for level, convs in enumerate(self.decoder):
            for conv in convs:
                x = conv(x)
            if level < (self.nb_levels - 1):
                x = self.upsampling[level](x)
                x = torch.cat([x, x_history.pop()], dim=1)
        for conv in self.remaining:
            x = conv(x)
        return x
