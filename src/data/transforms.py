"""Data transforms"""

import random

import torch

from torchaudio.transforms import Vad


class Pad(torch.nn.Module):
    """
    Audio zero-padding

    Parameters
    ----------
    max_length : int
        Max number of audio frames
    direction : str, optional
        Padding direction, by default "right"
    """

    def __init__(self, max_length, direction="right"):
        super().__init__()
        self.max_length = max_length

        # TODO: add bidirectional padding mode
        assert direction in ("left", "right", "random")
        self.direction = direction

    def forward(self, x):
        """Forward pass

        Parameters
        ----------
        x : torch.Tensor
            Input of size (B, T)
            B: batch size
            T: sample length

        Returns
        -------
        x : torch.Tensor
            Padded tensor of size (B, T*)
            T* >= T
        """
        sample_length = len(x)

        pad_length = self.max_length - sample_length

        if pad_length > 0:
            padding = torch.zeros(pad_length, dtype=x.dtype)

            if self.direction == "random":
                direction = "left" if random.random() < 0.5 else "right"
            else:
                direction = self.direction

            if direction == "left":
                x = torch.cat((padding, x), dim=0)
            else:
                x = torch.cat((x, padding), dim=0)
        return x


class Trim(torch.nn.Module):
    """
    Audio trimming

    Parameters
    ----------
    max_length : int
        Max number of audio frames
    direction : str, optional
        Trimming direction, by default "random"
    """

    def __init__(self, max_length, direction="right"):
        super().__init__()
        self.max_length = max_length

        # TODO: add bidirectional trimming mode
        assert direction in ("left", "right", "random")
        self.direction = direction

    def forward(self, x):
        """Forward pass

        Parameters
        ----------
        x : torch.Tensor
            Input of size (B, T)
            B: batch size
            T: sample length

        Returns
        -------
        x : torch.Tensor
            Trimmed tensor of size (B, T*)
            T* <= T
        """
        sample_length = len(x)

        trim_length = sample_length - self.max_length - 1

        if trim_length > 0:
            if self.direction == "random":
                start = random.randint(0, trim_length)
                stop = start + self.max_length
            elif self.direction == "right":
                start = 0
                stop = self.max_length
            else:
                start = trim_length + 1
                stop = sample_length

            x = x[start:stop]

        return x


def load_transforms(
    sr,
    max_length,
    with_vad=False,
    trim_direction="right",
    pad_direction="right",
):
    transforms = []

    if with_vad:
        transforms.append(Vad(sample_rate=sr))

    transforms.extend(
        [
            Trim(max_length=max_length, direction=trim_direction),
            Pad(max_length=max_length, direction=pad_direction),
        ]
    )

    return torch.nn.Sequential(*transforms)
