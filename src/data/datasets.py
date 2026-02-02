"""Dataset classes"""

import csv
import json
from abc import abstractmethod
from glob import glob
from pathlib import Path

import pandas as pd
import torch
from speechbrain.dataio.encoder import CategoricalEncoder
from torch.utils.data import Dataset
from tqdm import tqdm

from .._config import NONE_TENSOR
from ..models.transformers import TransformersAudioProcessor
from .glottolog import filter_languages_from_glottocode
from .transforms import load_transforms


class BaseDataset(Dataset):
    """Base dataset for this project

    Parameters
    ----------
    dataset : str
        Name of a dataset. Example: `fleurs`.
    subset : str
        Data subset (`train`, `dev`, `test`).
    root_dir : str
        Root folder path for analyses
    transform : callable, optional
        Optional transform to be applied on a sample, by default None
    target_transform : callable, optional
        Optional transform to be applied on a label, by default None
    """

    # pylint: disable=unused-argument
    def __init__(
        self,
        dataset,
        subset,
        root_dir,
        transform=None,
        target_transform=None,
        **kwargs,  # avoids argument errors when passing kwargs to BaseDataset
    ):
        self.dataset = dataset
        self.subset = subset
        self.root_dir = root_dir

        self.data_dir = f"{self.root_dir}/datasets/{self.dataset}"
        self.meta_dir = f"{self.root_dir}/metadata/{self.dataset}"

        with open(f"{self.meta_dir}/languages.json", "r", encoding="utf-8") as f:
            self.languages = json.load(f)

        self.label_encoder = CategoricalEncoder()
        self.label_encoder.load_or_create(
            path=f"{self.meta_dir}/labels.txt",
            from_iterables=[list(self.languages.keys())],
            output_key="lang_id",
        )
        self.label_encoder.expect_len(len(self.languages))

        self.transform = transform
        self.target_transform = target_transform

        pattern = f"{self.data_dir}/*/*/{subset or '*'}.tsv"

        language_transcripts = sorted(glob(pattern))

        if len(language_transcripts) == 0:
            raise FileNotFoundError(f"No files found for {pattern}")

        dfs = []

        for file in tqdm(language_transcripts, desc="(datasets) Reading data"):
            language = Path(file).parents[0].stem

            subset = Path(file).stem

            df = pd.read_csv(
                file,
                sep="\t",
                names=[
                    "sentence_index",
                    "fname",
                    "sentence",
                    "sentence_lower",
                    "chars",
                    "num_samples",
                    "gender",
                ],
                quoting=csv.QUOTE_NONE,
            )

            df["sentence_index"] = df["sentence_index"].apply(
                lambda x, subset=subset: f"{subset}_{x}"
            )

            df["subset"] = subset
            df["language"] = self.label_encoder.encode_label(language)

            dfs.append(df)

        self.data = pd.concat(dfs, axis=0).dropna()

    # pylint: enable=unused-argument

    def __len__(self):
        return self.data.shape[0]

    @abstractmethod
    def __getitem__(self, idx):
        raise NotImplementedError


class AudioDataset(BaseDataset):
    """Dataset for audio samples

    Parameters
    ----------
    processor : core.upstream.AudioProcessor
        an object with a ```process``` function
    ext : str, optional
        Audio file extension, by default "wav"
    """

    def __init__(
        self,
        processor,
        # ext="wav",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.processor = processor

    @staticmethod
    def read_audio(processor, pattern):  # self, basename, subset):
        """Read audio from a filename

        Parameters
        ----------
        basename : str
            Basename of an audio file (no path)
        subset : str
            Train/dev/test

        Returns
        -------
        x : torch.Tensor
            Audio sample, loaded as a tensor
        attention_mask : torch.Tensor
            Attention mask (None if not a ```transformers``` processor)
        """
        # pattern = f"{self.data_dir}/*/*/audio/{subset}/{basename}"
        wav_path = glob(pattern)

        if len(wav_path) != 1:
            raise ValueError(f"Expected 1 file for {pattern}, got {len(wav_path)}")

        x, attention_mask = processor(wav_path[0])

        return x, attention_mask

    def __getitem__(self, idx):
        # Path: data_dir/language/language/audio/subset/*.wav
        _, basename, *_, subset, label = self.data.iloc[idx]

        x, attention_mask = self.read_audio(
            processor=self.processor,
            pattern=f"{self.data_dir}/*/*/audio/{subset}/{basename}",
        )

        # NOTE: might change the duration/length of x!
        if self.transform:
            x = self.transform(x)

        if self.target_transform:
            label = self.target_transform(label)

        return {"input": x, "attention_mask": attention_mask, "label": label}


class FleursParallelDataset(BaseDataset):
    """A torch.utils.data.Dataset to load FLEURS sentences in parallel

    Parameters
    ----------
    processor : core.upstream.AudioProcessor or core.upstream.TextProcessor
        an object with a ```process``` function
    glottocode : str, optional
        A glottocode from which to select a subset of languages, by default None
        Example: ```indo1319``` (Indo-European)
    min_speakers : float, optional
        Min. number of speakers to select a subset of languages, by default 0.0
        (In millions)
    subset : str, optional
        Data subset (`train`, `dev`, `test`), by default None
    kwargs
        All other keyword arguments are passed to BaseDataset
    """

    # pylint: disable=unused-variable
    def __init__(
        self,
        dtype,
        processor=None,
        # sentence_index_col="sentence_index",
        glottocode=None,
        min_speakers=0.0,
        subset=None,
        # unique=False, # Only keep one sample per language (i.e., one speaker)
        **kwargs,
    ):
        kwargs["subset"] = subset
        super().__init__(**kwargs)

        self.processor = processor

        self.dtype = dtype

        self.glottocode = glottocode

        if self.glottocode is not None:
            glottolog_path = f"{self.meta_dir}/glottolog.csv"
            languages_to_keep = filter_languages_from_glottocode(
                self.dataset,
                glottocode=glottocode,
                min_speakers=min_speakers,
            )
            labels_to_keep = self.label_encoder.encode_sequence(  # noqa: F841
                languages_to_keep
            )
            print(
                f"(datasets) Using {self.glottocode} to select {len(languages_to_keep)} languages."
            )

            self.data = self.data.query("language in @labels_to_keep")

        self.sentence_idxs = self.data.sentence_index.unique()

    # pylint: enable=unused-variable

    def __len__(self):
        return self.sentence_idxs.shape[0]

    def __getitem__(self, idx):
        sentence_index = self.sentence_idxs[idx]
        sentence_data = self.data.query("sentence_index == @sentence_index")

        # Avoids duplicating similar transcripts --> saves time
        # FIXME: does not work with pdist_multimodel!!
        # if self.dtype == "text":
        #     sentence_data.drop_duplicates("language", inplace=True)

        inputs = []
        attention_masks = []
        labels = torch.zeros((sentence_data.shape[0],)).long()

        for i, (_, row) in enumerate(sentence_data.iterrows()):
            _, basename, sentence, *_, subset, label = row

            if self.dtype == "audio":
                input_, attention_mask = AudioDataset.read_audio(
                    processor=self.processor,
                    pattern=f"{self.data_dir}/*/*/audio/{subset}/{basename}",
                )
            else:
                input_ = sentence

                if self.processor is not None:
                    input_, attention_mask = self.processor(input_)
                else:
                    attention_mask = NONE_TENSOR

            if self.transform:
                input_ = self.transform(input_)

            if self.target_transform:
                label = self.target_transform(label)

            inputs.append(input_)
            labels[i] = label
            attention_masks.append(attention_mask)

        attention_masks = torch.stack(attention_masks)

        return {
            "input": inputs,
            "label": labels,
            "attention_mask": attention_masks,
            "sentence_index": sentence_index,
        }


def load_dataset(dtype, dataset, root_dir, split=True, fleurs_parallel=False, **kwargs):
    """Load train/dev/test audio or text datasets

    Parameters
    ----------
    dataset : str
        Name of a dataset. Example: `fleurs`.
    root_dir : str
        Root folder path for analyses
    fleurs_parallel : bool, optional
        If True, load a FleursParallelDataset, by default False

    Returns
    -------
    tuple
        If split is True, return
        train_dataset : torch.utils.data.Dataset object
            Train dataset
        valid_dataset : torch.utils.data.Dataset object
            Validation dataset
        test_dataset : torch.utils.data.Dataset object
            Test dataset

        Else, return
        dataset : torch.utils.data.Dataset object
            Un-split dataset (train+dev+test combined)
    """
    if fleurs_parallel:
        cls = FleursParallelDataset
        kwargs["dtype"] = dtype
    elif dtype == "audio":
        cls = AudioDataset
    else:
        raise NotImplementedError

    processor = kwargs["processor"]
    # For transformer models, trimming/padding is built in
    # For other models, need to trim/pad for non-Transformer models
    # Such that input shape is `max_length` for all samples and models
    if not isinstance(processor, TransformersAudioProcessor):
        kwargs["transform"] = load_transforms(
            sr=processor.sr,
            max_length=processor.max_length,
            with_vad=kwargs["with_vad"],
        )

    positional_args = {"dataset": dataset, "root_dir": root_dir}

    if split:
        train_dataset = cls(
            subset="train",
            **positional_args,
            **kwargs,
        )

        valid_dataset = cls(
            subset="dev",
            **positional_args,
            **kwargs,
        )

        test_dataset = cls(
            subset="test",
            **positional_args,
            **kwargs,
        )

        return train_dataset, valid_dataset, test_dataset

    return (
        cls(
            subset=None,
            **positional_args,
            **kwargs,
        ),
    )
