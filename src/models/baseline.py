# -----------------------------------------------------------------------------
# Portions of this code are adapted from SpokenLanguageClassifiers
# Original repository: https://github.com/RicherMans/SpokenLanguageClassifiers
# Copyright (c) Heinrich Dinkel, 2020
# Licensed under the MIT License (https://opensource.org/licenses/MIT)
# -----------------------------------------------------------------------------

import os
import urllib

import librosa
import numpy as np
import torch
from torch import nn

from .._config import SAMPLE_RATE
from .base import BaseFeatureExtractor

BASELINE_CKPTS = {
    "CNN10": {
        "url": "https://zenodo.org/record/4436037/files/CNN10.pth?download=1",
        "md5": "ca56e5003b5025eff6f991e47ba87b06",
    },
    "CNN6": {
        "url": "https://zenodo.org/record/4436037/files/CNN6.pth?download=1",
        "md5": "b0ae5a1bce63fa5522939fa123d3f0a3",
    },
}

NUM_CLASSES_VOXLINGUA107 = 107


class CNN6(nn.Module):
    def __init__(self, inputdim, outputdim, **kwargs):
        super().__init__()
        self.embed = kwargs.get("embed", 256)

        def _block(in_channel, out_channel):
            return nn.Sequential(
                nn.Conv2d(
                    in_channel, out_channel, kernel_size=3, bias=False, padding=1
                ),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(True),
                nn.Conv2d(
                    out_channel, out_channel, kernel_size=3, bias=False, padding=1
                ),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(True),
            )

        self.init_bn = nn.BatchNorm1d(inputdim)
        self.features = nn.Sequential(
            _block(1, 64),
            nn.MaxPool2d(2),
            _block(64, 128),
            nn.MaxPool2d(2),
            _block(128, self.embed),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((None, 1)),
            nn.Dropout(0.2, True),
        )
        self.outputlayer = nn.Linear(self.embed, outputdim)
        self.attention = nn.Linear(self.embed, self.embed)

    def forward(self, x):
        x = self.init_bn(x.transpose(1, 2)).transpose(1, 2)
        x = x.unsqueeze(1)  # B x 1 x T x D
        x = self.features(x)
        x = x.transpose(1, 2).contiguous().flatten(-2)
        # Attention
        w = torch.softmax(self.attention(x), dim=1)
        x = (x * w).sum(1) / (w.sum(1) + 1e-7)
        return self.outputlayer(x)


class CNN10(nn.Module):
    def __init__(self, inputdim, outputdim, **kwargs):
        super().__init__()
        self.embed = kwargs.get("embed", 512)

        def _block(in_channel, out_channel):
            return nn.Sequential(
                nn.Conv2d(
                    in_channel, out_channel, kernel_size=3, bias=False, padding=1
                ),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(True),
                nn.Conv2d(
                    out_channel, out_channel, kernel_size=3, bias=False, padding=1
                ),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(True),
            )

        self.init_bn = nn.BatchNorm1d(inputdim)
        self.features = nn.Sequential(
            _block(1, 64),
            nn.AvgPool2d((2, 2)),
            nn.Dropout(0.2, True),
            _block(64, 128),
            nn.AvgPool2d((2, 2)),
            nn.Dropout(0.2, True),
            _block(128, 256),
            nn.AvgPool2d((2, 2)),
            nn.Dropout(0.2, True),
            _block(256, self.embed),
            nn.AvgPool2d((2, 2)),
            nn.Dropout(0.2, True),
            nn.AdaptiveAvgPool2d((None, 1)),
        )
        self.outputlayer = nn.Linear(self.embed, outputdim)

    def forward(self, x):
        x = self.init_bn(x.transpose(1, 2)).transpose(1, 2)
        x = x.unsqueeze(1)  # B x 1 x T x D
        x = self.features(x)
        x = x.transpose(1, 2).contiguous().flatten(-2)
        x = x.mean(1) + x.max(1)[0]
        return self.outputlayer(x)


class BaselineFeatureExtractor(BaseFeatureExtractor):

    def __init__(
        self,
        model_id,
        cache_dir=None,
        n_mels=64,
        n_fft=2048,
        hop_length=320,
        win_length=640,
        **kwargs,
    ):
        super().__init__(model_id)

        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

        self.load(cache_dir)

    def load(self, cache_dir=None):
        if cache_dir is not None:
            cache_dir = f"{cache_dir}/baselines"

        os.makedirs(cache_dir, exist_ok=True)

        _, model_name = self.model_id.split("/")

        if model_name == "CNN6":
            feature_extractor = CNN6(
                inputdim=self.n_mels, outputdim=NUM_CLASSES_VOXLINGUA107
            )
        elif model_name == "CNN10":
            feature_extractor = CNN10(
                inputdim=self.n_mels, outputdim=NUM_CLASSES_VOXLINGUA107
            )
        else:
            raise ValueError(f"Unknown model ID: {self.model_id}")

        save_path = f"{cache_dir}/{model_name}.pth"

        if not os.path.exists(save_path):
            url = BASELINE_CKPTS[model_name]["url"]
            urllib.request.urlretrieve(url, save_path)

        ckpt = torch.load(save_path)

        feature_extractor.load_state_dict(ckpt)

        # Remove final layer
        feature_extractor.outputlayer = torch.nn.Identity()

        # Freeze parameters
        for param in feature_extractor.parameters():
            param.requires_grad = False

        feature_extractor.eval()

        self.feature_extractor = feature_extractor
        self.emb_dim = feature_extractor.embed

    def forward(self, x):
        lms = np.log(
            librosa.feature.melspectrogram(
                y=x.cpu().numpy(),
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                n_mels=self.n_mels,
                sr=SAMPLE_RATE,
            )
            + np.spacing(1)
        )

        lms = torch.from_numpy(lms).float().to(x.device).permute(0, 2, 1)

        out = self.feature_extractor(lms)

        return out
