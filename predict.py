import utils
import torch
import torchaudio
from mir_eval.separation import *
from pathlib import Path
import numpy as np

from Config import TEST_DIR_1, TEST_DIR_2
import os

def main(file_num, dir_path):
    file_num = file_num
    # reference_sources = np.zeros([file_num, time_step, channel])
    # estimated_sources = np.zeros([file_num, time_step, channel])
    target = "zhudi"
    sdrs = []
    # isrs = []
    sirs = []
    sars = []
    # perms = []
    for i in range(0, file_num):
        song_path = os.path.join(dir_path, str(i) + "mixture.wav")
        target_path = os.path.join(dir_path, str(i) + "y.wav")
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")

        # kaggle上新增
        separator = utils.load_separator(
            model_str_or_path="open-unmix_3",  # kaggle上修改
            targets="zhudi",
            niter=1,
            residual=True,
            wiener_win_len=300,
            device=device,
            pretrained=True,
            filterbank="torch",
        )
        separator.freeze()
        separator.to(device)
        audio, rate = torchaudio.load(song_path, normalize=True)  # audio 读取为tensor
        aggregate_dict = None

        if device:
            audio = audio.to(device)

        # audio格式转换为shape(nb_samples, nb_channels, nb_timesteps)
        audio = utils.preprocess(audio, rate, separator.sample_rate)
        estimate = separator(audio)
        estimates = separator.to_dict(estimate, aggregate_dict=aggregate_dict)

        # load target audio
        reference_source, _ = torchaudio.load(target_path, normalize=True)
        reference_source = utils.preprocess(reference_source, rate, separator.sample_rate)
        reference_source = reference_source.permute(1, 0, 2)
        estimated_source = estimates[target]
        # 保存估计音频(sample)
        if i == 2:
            torchaudio.save(str(3) + str(target) + "1.wav", estimated_source[0], rate, bits_per_sample=16)
        estimated_source = estimated_source.permute(1, 0, 2)
        # 转换成narray来evaluate
        reference_source = reference_source[0].numpy()
        estimated_source = estimated_source[0].numpy()
        # reference_sources[i]=reference_source
        # estimated_sources[i]=estimated_source
        # (sdr, isr, sir, sar, perm) = bss_eval_images_framewise(reference_source, estimated_source,window=44100,
        #                                                        hop=22050, compute_permutation=False, )
        try:
            (sdr, sir, sar, perm) = bss_eval_sources(reference_source, estimated_source, compute_permutation=False)
            sdrs.append(sdr)
            # isrs.append(isr)
            sirs.append(sir)
            sars.append(sar)
            # perms.append(perm)
            print("{}:sdr:{},sir:{},sar:{}\n".format(i,sdr, sir, sar))
        except Exception as e:
            print(e)
    return  sdrs, sars

if __name__ == '__main__':
    sdrs, sars = main(741, TEST_DIR_2)
    sdrs2, sars2 = main(340, TEST_DIR_1)
    sdrs.extend(sdrs2)
    sars.extend(sars2)
    # 去掉nan
    sdrs = [x for x in sdrs if np.isnan(x) == False]
    sars = [x for x in sars if np.isnan(x) == False]
    sdr_avg= sum(sdrs)/len(sdrs)
    print("sdr:{}".format(sdr_avg))
    sar_avg = sum(sars) / len(sars)
    print("sar:{}".format(sar_avg))

