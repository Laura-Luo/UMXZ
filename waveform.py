import os

import Config
import torchaudio
from pathlib import Path
import subprocess
from pydub import AudioSegment
import librosa
import soundfile
import matplotlib.pyplot as plt
import numpy as np


FILE_PATH =Config.New_DIR


def print_metadata(metadata, src=None):
  if src:
    print("-" * 10)
    print("Source:", src)
    print("-" * 10)
  print(" - sample_rate:", metadata.sample_rate)
  print(" - num_channels:", metadata.num_channels)
  print(" - num_frames:", metadata.num_frames)
  print(" - encoding:", metadata.encoding)
  print()

if __name__ == '__main__':

    # 打印音频文件信息
    # 遍历所有文件
    # 复制一份wav文件保存audio_ok_name, 利用sox调整参数：通道-1 位-16 采样率-16k
    # dirs = os.listdir(FILE_PATH)
    # for dir in dirs:
    #     dirpath = Path(FILE_PATH, dir)
    #     files = dirpath.glob("*" + ".wav")
    # # dirpath = Path(FILE_PATH, "zhonghu")
    # # files = dirpath.glob("*" + ".wav")
    #
    #     for file in files:
    #         filename = os.path.join(FILE_PATH,file)
    #         # song = AudioSegment.from_wav(filename)
    #         # if song.frame_width != 4:
    #         #     print(filename, song.frame_width)
    #         # if song.channels ==1:
    #         #     print(filename, song.channels)
    #         #     left = AudioSegment.from_wav(filename)
    #         #     right = AudioSegment.from_wav(filename)
    #         #     stereo = AudioSegment.from_mono_audiosegments(left,right)
    #         #     stereo.export(filename, format="wav")
    #         #     metadata = torchaudio.info(filename)
    #         #     print_metadata(metadata, src=filename)
    #         # 显示信息
    #         metadata=torchaudio.info(filename)
    #         dura = metadata.duration
    #         dura_total += dura
            # if metadata.encoding != "PCM_S":
            #     print_metadata(metadata, src=filename)
            # if metadata.sample_rate != 44100:
            #     print_metadata(metadata, src=filename)
            #     print("resample")
            #     y,sr = librosa.load(filename)
            #     y_new = librosa.resample(y,sr,44100)
            #     soundfile.write(filename,y_new,44100)
            #     metadata = torchaudio.info(filename)
            #     print_metadata(metadata, src=filename)
    dir = Path(FILE_PATH)
    file = files = dir.glob("*" + ".wav")
    dura_total = 0
    for file in files:
        filename = os.path.join(FILE_PATH,file)
        metadata = torchaudio.info(filename)
        dura = metadata.num_frames/metadata.sample_rate
        dura_total += dura
    print("Total duration of Dataset: {:.2f}".format(dura_total))


