import argparse
import os
import Config
import random
from pathlib import Path
from typing import Optional, Union, Tuple, List, Any, Callable

import torch
import torch.utils.data
import torchaudio
import tqdm

test_path = Config.test_DIR_2

def load_info(path: str) -> dict:
    """Load audio metadata

    this is a backend_independent wrapper around torchaudio.info

    Args:
        path: Path of filename
    Returns:
        Dict: Metadata with
        `samplerate`, `samples` and `duration` in seconds

    """
    # get length of file in samples
    if torchaudio.get_audio_backend() == "sox":
        raise RuntimeError("Deprecated backend is not supported")

    info = {}
    si = torchaudio.info(str(path))
    info["samplerate"] = si.sample_rate
    info["samples"] = si.num_frames
    info["channels"] = si.num_channels
    info["duration"] = info["samples"] / info["samplerate"]
    return info


def load_audio(
    path: str,
    start: float = 0.0,
    dur: Optional[float] = None,
    info: Optional[dict] = None,
):
    """Load audio file

    Args:
        path: Path of audio file
        start: start position in seconds, defaults on the beginning.
        dur: end position in seconds, defaults to `None` (full file).
        info: metadata object as called from `load_info`.

    Returns:
        Tensor: torch tensor waveform of shape `(num_channels, num_samples)`
    """
    # loads the full track duration
    if dur is None:
        # we ignore the case where start!=0 and dur=None
        # since we have to deal with fixed length audio
        sig, rate = torchaudio.load(path, normalize=True)
        return sig, rate
    else:
        if info is None:
            info = load_info(path)
        num_frames = int(dur * info["samplerate"])
        frame_offset = int(start * info["samplerate"])
        sig, rate = torchaudio.load(path, num_frames=num_frames, frame_offset=frame_offset, normalize=True)
        return sig, rate


def aug_from_str(list_of_function_names: list):
    if list_of_function_names:
        return Compose([globals()["_augment_" + aug] for aug in list_of_function_names])
    else:
        return lambda audio: audio


class Compose(object):
    """Composes several augmentation transforms.
    Args:
        augmentations: list of augmentations to compose.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            audio = t(audio)
        return audio


def _augment_gain(audio: torch.Tensor, low: float = 0.25, high: float = 1.25) -> torch.Tensor:
    """Applies a random gain between `low` and `high`"""
    g = low + torch.rand(1) * (high - low)
    return audio * g


def _augment_channelswap(audio: torch.Tensor) -> torch.Tensor:
    """Swap channels of stereo signals with a probability of p=0.5"""
    if audio.shape[0] == 2 and torch.tensor(1.0).uniform_() < 0.5:
        return torch.flip(audio, [0])
    else:
        return audio


def _augment_force_stereo(audio: torch.Tensor) -> torch.Tensor:
    # for multichannel > 2, we drop the other channels
    if audio.shape[0] > 2:
        audio = audio[:2, ...]

    if audio.shape[0] == 1:
        # if we have mono, we duplicate it to get stereo
        audio = torch.repeat_interleave(audio, 2, dim=0)

    return audio


class UnmixDataset(torch.utils.data.Dataset):
    _repr_indent = 4

    def __init__(
        self,
        root: Union[Path, str],
        sample_rate: float,
        seq_duration: Optional[float] = None,
        source_augmentations: Optional[Callable] = None,
    ) -> None:
        self.root = Path(root).expanduser()
        self.sample_rate = sample_rate
        self.seq_duration = seq_duration
        self.source_augmentations = source_augmentations

    def __getitem__(self, index: int) -> Any:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = ["Number of datapoints: {}".format(self.__len__())]
        body += self.extra_repr().splitlines()
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return "\n".join(lines)

    def extra_repr(self) -> str:
        return ""


def load_datasets(seq_dur,seed,root,interferer_dirs,target_dir,ext,source_augmentations,nb_train_samples,
                  nb_valid_samples,input_file, output_file,root_a):
    """Loads the specified dataset from commandline arguments

    Returns:
        train_dataset, validation_dataset
    """


    source_augmentations = aug_from_str(source_augmentations)

    # 对齐数据集加载
    # dataset_kwargs = {
    #     "root": root_a,
    #     "input_file": input_file,
    #     "output_file": output_file,
    # }
    # train_dataset_1 = AlignedDataset(
    #     split="train", **dataset_kwargs
    # )  # type: UnmixDataset
    # valid_dataset_1 = AlignedDataset(
    #     split="valid",
    #     **dataset_kwargs)

    # 原数据集加载法
    dataset_kwargs = {
        "root": Path(root),
        "interferer_dirs": interferer_dirs,
        "target_dir": target_dir,
        "ext": ext,
    }
    train_dataset = SourceFolderDataset(
        split="train",
        source_augmentations=source_augmentations,
        random_chunks=True,
        nb_samples=nb_train_samples,
        seq_duration=seq_dur,
        **dataset_kwargs,
    )

    valid_dataset = SourceFolderDataset(
        split="valid",
        random_chunks=True,
        seq_duration=seq_dur,
        nb_samples=nb_valid_samples,
        seed=None,
        **dataset_kwargs,
    )

    # train_dataset = train_dataset_1+train_dataset_2
    # valid_dataset = valid_dataset_1+valid_dataset_2

    return train_dataset, valid_dataset


class AlignedDataset(UnmixDataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        input_file: str = "mixture.wav",
        output_file: str = "y.wav",
        sample_rate: float = 44100.0,
        source_augmentations: Optional[Callable] = None,
    ) -> None:
        """A dataset of that assumes multiple track folders
        where each track includes and input and an output file
        which directly corresponds to the the input and the
        output of the model. This dataset is the most basic of
        all datasets provided here, due to the least amount of
        preprocessing, it is also the fastest option, however,
        it lacks any kind of source augmentations or custum mixing.

        Typical use cases:

        * Source Separation (Mixture -> Target)
        * Denoising (Noisy -> Clean)
        * Bandwidth Extension (Low Bandwidth -> High Bandwidth)

        Example
        =======
        data/train/01/mixture.wav --> input
        data/train/01/vocals.wav ---> output

        """
        self.root = Path(root).expanduser()
        self.split = split
        self.sample_rate = sample_rate
        # set the input and output files (accept glob)
        self.input_file = input_file
        self.output_file = output_file
        self.tuple_paths = list(self._get_paths())
        if not self.tuple_paths:
            raise RuntimeError("Dataset is empty, please check parameters")

    def __getitem__(self, index):
        input_path, output_path = self.tuple_paths[index]
        start = 0
        X_audio, _ = load_audio(input_path, start=start)
        Y_audio, _ = load_audio(output_path, start=start)
        # return torch tensors
        return X_audio, Y_audio

    def __len__(self):
        return len(self.tuple_paths)

    def _get_paths(self):
        """Loads input and output tracks"""
        p = Path(self.root, self.split)
        for track_path in tqdm.tqdm(p.iterdir()):
            if track_path.is_dir():
                input_path = list(track_path.glob(self.input_file))
                output_path = list(track_path.glob(self.output_file))
                if input_path and output_path:
                    yield input_path[0], output_path[0]



class SourceFolderDataset(UnmixDataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        target_dir: str = "vocals",
        interferer_dirs: List[str] = ["bass", "drums"],
        ext: str = ".wav",
        nb_samples: int = 10,
        seq_duration: Optional[float] = 6.0,
        random_chunks: bool = True,
        sample_rate: int = 44100,
        source_augmentations: Optional[Callable] = lambda audio: audio,
        seed: int = 60,
    ) -> None:
        """A dataset that assumes folders of sources,
        instead of track folders. This is a common
        format for speech and environmental sound datasets
        such das DCASE. For each source a variable number of
        tracks/sounds is available, therefore the dataset
        is unaligned by design.
        By default, for each sample, sources from random track are drawn
        to assemble the mixture.

        Example
        =======
        train/vocals/track11.wav -----------------\
        train/drums/track202.wav  (interferer1) ---+--> input
        train/bass/track007a.wav  (interferer2) --/

        train/vocals/track11.wav ---------------------> output

        """
        self.root = Path(root).expanduser()
        self.split = split
        self.sample_rate = sample_rate
        self.seq_duration = seq_duration
        self.ext = ext
        self.random_chunks = random_chunks
        self.source_augmentations = source_augmentations
        self.target_dir = target_dir
        self.interferer_dirs = interferer_dirs
        self.source_folders = self.interferer_dirs + [self.target_dir]
        self.source_tracks = self.get_tracks()
        self.nb_samples = nb_samples
        self.seed = seed
        # random.seed(self.seed) # 每次这个文件生成的训练集都是一样的

    def __getitem__(self, index):
        # For each source draw a random sound and mix them together
        audio_sources = []
        for source in self.source_folders:
            # if self.split == "valid":
                # provide deterministic behaviour for validation so that
                # each epoch, the same tracks are yielded 确保每次的验证集都是一样的
                # random.seed(index)

            # select a random track for each source
            source_path = random.choice(self.source_tracks[source])
            duration = load_info(source_path)["duration"]
            # set random start position 此处duration都大于seq_dur
            if self.random_chunks:
                # for each source, select a random chunk
                start = random.uniform(0, duration - self.seq_duration)
            else:
                # use center segment 如果不用随机切割，就从中间切割
                start = max(duration // 2 - self.seq_duration // 2, 0)

            # load source audio and apply time domain source_augmentations
            audio, _ = load_audio(source_path, start=start, dur=self.seq_duration)
            if source == "percussions" or source=="zhudi"or source=="sheng":  # 打击乐和竹笛调整音量
                vol = torchaudio.transforms.Vol(gain=0.2, gain_type="amplitude")
            #     vol = torchaudio.transforms.Vol(gain=-5.9, gain_type="db")
                audio = vol(audio)
            audio = self.source_augmentations(audio)
            audio_sources.append(audio)

        # create stem tensor of shape (source, channel, samples)
        stems = torch.stack(audio_sources)
        # # apply linear mix over source index=0
        x = stems.sum(0)
        # target is always the last element in the list
        y = stems[-1]
        return x, y

    def __len__(self):
        return self.nb_samples

    def get_tracks(self):
        """Loads input and output tracks"""
        p = Path(self.root, self.split)
        source_tracks = {}
        for source_folder in self.source_folders:
            tracks = []
            source_path = p / source_folder
            for source_track_path in sorted(source_path.glob("*" + self.ext)):
                if self.seq_duration is not None:
                    info = load_info(source_track_path)
                    # get minimum duration of track
                    if info["duration"] > self.seq_duration:
                        tracks.append(source_track_path)
                else:
                    tracks.append(source_track_path)
            source_tracks[source_folder] = tracks
        return source_tracks



if __name__ == "__main__":


    audio_backend = "soundfile"
    batch_size = 16
    save = True
    source_augmentations = None
    # sourcefile的参数
    seq_dur = 5.0
    seed = 60
    root = "./data"

    target = "zhudi"
    interferer_dirs = ["guzheng", "yangqin", "pipa", "daruan", "konghou", "sheng", "zhonghu", "xiao", "percussions"]
    target_dir = "zhudi"
    ext = ".wav"
    nb_valid_samples = 340
    nb_train_samples = 1
    # aligned的参数
    root_a = "./data2"
    input_file = 'mixture.wav'
    output_file ='y.wav'
    torchaudio.set_audio_backend(audio_backend)

    train_dataset, valid_dataset = load_datasets(seq_dur, seed, root, interferer_dirs, target_dir, ext,
                                                 source_augmentations, nb_train_samples, nb_valid_samples,
                                                 input_file, output_file,root_a)
    print("Audio Backend: ", torchaudio.get_audio_backend())

    # Iterate over training dataset and compute statistics
    # total_training_duration = 0
    # for k in tqdm.tqdm(range(len(train_dataset))):
    #     x, y = train_dataset[k]
    #     total_training_duration += x.shape[1] / train_dataset.sample_rate
    #     if save:
    #         x_path = os.path.join(test_path, str(k)+"mixture.wav")
    #         print(x_path)
    #         y_path = os.path.join(test_path, str(k) + "y.wav")
    #         torchaudio.save(x_path, x, train_dataset.sample_rate, bits_per_sample=16)
    #         torchaudio.save(y_path, y, train_dataset.sample_rate, bits_per_sample=16)
    total_valid_duration = 0
    for k in tqdm.tqdm(range(len(valid_dataset))):
        x, y = valid_dataset[k]
        total_valid_duration += x.shape[1] / valid_dataset.sample_rate
        if save:
            x_path = os.path.join(test_path, str(k)+"mixture.wav")
            y_path = os.path.join(test_path, str(k) + "y.wav")
            torchaudio.save(x_path, x, valid_dataset.sample_rate, bits_per_sample=16)
            torchaudio.save(y_path, y, valid_dataset.sample_rate, bits_per_sample=16)

    # print("Total training duration (h): ", total_training_duration / 3600)
    print("Total training duration (h): ", total_valid_duration / 3600)
    print("Number of train samples: ", len(train_dataset))
    print("Number of validation samples: ", len(valid_dataset))

    # iterate over dataloader
    train_dataset.seq_duration = seq_dur

    train_sampler = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
    )