import argparse
import torch
import time
from pathlib import Path
import tqdm
import json
import sklearn.preprocessing
import numpy as np
import random
import os
import copy
import torchaudio

import data
import model
import utils
from openunmix import transforms


tqdm.monitor_interval = 0


def train(quiet, unmix, encoder, device, train_sampler, optimizer):
    losses = utils.AverageMeter()
    unmix.train()
    pbar = tqdm.tqdm(train_sampler, disable=quiet) # 进度条
    for x, y in pbar:
        pbar.set_description("Training batch")
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        X = encoder(x)
        Y_hat = unmix(X) # 经过模型输出
        Y = encoder(y)
        loss = torch.nn.functional.mse_loss(Y_hat, Y) #计算loss
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), Y.size(1))
        pbar.set_postfix(loss="{:.3f}".format(losses.avg))
    return losses.avg


def valid(unmix, encoder, device, valid_sampler):
    losses = utils.AverageMeter()
    unmix.eval()
    with torch.no_grad():
        for x, y in valid_sampler:
            x, y = x.to(device), y.to(device)
            X = encoder(x)
            Y_hat = unmix(X)
            Y = encoder(y)
            loss = torch.nn.functional.mse_loss(Y_hat, Y)
            losses.update(loss.item(), Y.size(1))
        return losses.avg


def get_statistics(quiet, encoder, dataset):
    """
        数据标准化
    :param quiet:
    :param encoder:
    :param dataset:
    :return:
    """
    encoder = copy.deepcopy(encoder).to("cpu")
    scaler = sklearn.preprocessing.StandardScaler()

    dataset_scaler = copy.deepcopy(dataset)
    if isinstance(dataset_scaler, data.CombinedDataset):
        dataset_scaler.random_chunks = False
    else:
        dataset_scaler.random_chunks = False
        dataset_scaler.seq_duration = None

    dataset_scaler.samples_per_track = 1
    dataset_scaler.augmentations = None
    dataset_scaler.random_track_mix = False
    dataset_scaler.random_interferer_mix = False

    pbar = tqdm.tqdm(range(len(dataset_scaler)), disable=quiet)
    for ind in pbar:
        x, y = dataset_scaler[ind]
        pbar.set_description("Compute dataset statistics")
        # downmix to mono channel
        X = encoder(x[None, ...]).mean(1, keepdim=False).permute(0, 2, 1)

        scaler.partial_fit(np.squeeze(X))

    # set inital input scaler values
    std = np.maximum(scaler.scale_, 1e-4 * np.max(scaler.scale_))
    return scaler.mean_, std


def main():

    # 设置参数
    # 训练的参数---------
    epochs = 1000
    lr = 0.001
    lr_decay_gamma = 0.3
    lr_decay_patience = 80

    patience = 140 # 训练的最大epoch数
    weight_decay = 0.00001
    unidirectional = False
    # STFT parameters
    nfft = 4096
    nhop = 1024
    hidden_size = 512
    bandwidth = 16000
    nb_channels = 2

    no_cuda = False
    nb_workers = 4
    debug= False
    quiet = False  # disable print and progress bar during training

    # dataset 的参数---------------
    audio_backend = "soundfile"
    seq_dur = 5.0
    batch_size = 16
    seed = 42
    target = "zhudi"
    root = "./data"
    interferer_dirs = ["guzheng", "yangqin", "pipa", "daruan", "konghou", "sheng", "zhonghu", "xiao", "percussions"]
    target_dir = "zhudi"
    ext = ".wav"
    source_augmentations = None
    nb_train_samples = 1000
    nb_valid_samples = 300
    output = "./open-unmix"
    # aligned添加的参数------------------
    root_a = "./data2"
    input_file = 'mixture.wav'
    output_file = 'y.wav'
    # --------------------
    arg_model =""
    checkpoint =""

    # ---------------------------------------
    # 设置backend
    torchaudio.set_audio_backend(audio_backend)
    # 设置gpu使用
    use_cuda = not no_cuda and torch.cuda.is_available()
    print("Using GPU:", use_cuda)
    dataloader_kwargs = {"num_workers": nb_workers, "pin_memory": True} if use_cuda else {}

    # use jpg or npy 设置种子
    torch.manual_seed(seed)
    random.seed(seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    # 加载数据集，数据集中的也需要变一下，参数是命令行传进去的，返回的args也是命令行传的参数原样返回
    train_dataset, valid_dataset = data.load_datasets(seq_dur, seed, root, interferer_dirs, target_dir, ext,
                                                 source_augmentations, nb_train_samples, nb_valid_samples,
                                                 input_file, output_file,root_a)

    # create output dir if not exist 后面的args都是数据集加载中出来的
    target_path = Path(output)
    target_path.mkdir(parents=True, exist_ok=True)

    # 处理数据集
    train_sampler = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, **dataloader_kwargs
    )
    valid_sampler = torch.utils.data.DataLoader(valid_dataset, batch_size=1, **dataloader_kwargs)

    # stft方法加载
    stft, _ = transforms.make_filterbanks(
        n_fft=nfft, n_hop=nhop, sample_rate=train_dataset.sample_rate
    )
    # encoder方法加载
    encoder = torch.nn.Sequential(stft, model.ComplexNorm(mono=nb_channels == 1)).to(device)

    separator_conf = {
        "nfft": nfft,
        "nhop": nhop,
        "sample_rate": train_dataset.sample_rate,
        "nb_channels": nb_channels,
    }

    # 生成json文件
    with open(Path(target_path, "separator.json"), "w") as outfile:
        outfile.write(json.dumps(separator_conf, indent=4, sort_keys=True))

    # 获得参数？？？
    if checkpoint or arg_model or debug:
        scaler_mean = None
        scaler_std = None
    else:
        scaler_mean, scaler_std = get_statistics(quiet, encoder, train_dataset)

    # 将带宽转换为最大 bin 计数
    max_bin = utils.bandwidth_to_max_bin(train_dataset.sample_rate, nfft, bandwidth)

    # 加载模型
    if arg_model:
        # fine tune model 预训练的模型
        print(f"Fine-tuning model from {arg_model}")
        unmix = utils.load_target_models(
            target, model_str_or_path=arg_model, device=device, pretrained=True
        )[target]
        unmix = unmix.to(device)
    else:
        unmix = model.OpenUnmix(
            input_mean=scaler_mean,
            input_scale=scaler_std,
            nb_bins=nfft // 2 + 1,
            nb_channels=nb_channels,
            hidden_size=hidden_size,
            max_bin=max_bin,
        ).to(device)

    # 优化器加载
    optimizer = torch.optim.Adam(unmix.parameters(), lr=lr, weight_decay=weight_decay)

    # 执行器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=lr_decay_gamma,
        patience=lr_decay_patience,
        cooldown=10,
    )

    #
    es = utils.EarlyStopping(patience=patience)

    # if a checkpoint is specified: resume training
    if checkpoint:
        model_path = Path(checkpoint).expanduser()
        with open(Path(model_path, target + ".json"), "r") as stream:
            results = json.load(stream)

        target_model_path = Path(model_path, target + ".chkpnt")
        checkpoint = torch.load(target_model_path, map_location=device)
        unmix.load_state_dict(checkpoint["state_dict"], strict=False)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        # train for another epochs_trained
        t = tqdm.trange(
            results["epochs_trained"],
            results["epochs_trained"] + epochs + 1,
            disable=quiet,
        )
        train_losses = results["train_loss_history"]
        valid_losses = results["valid_loss_history"]
        train_times = results["train_time_history"]
        best_epoch = results["best_epoch"]
        es.best = results["best_loss"]
        es.num_bad_epochs = results["num_bad_epochs"]
    # else start optimizer from scratch
    else:
        t = tqdm.trange(1, epochs + 1, disable=quiet)
        train_losses = []
        valid_losses = []
        train_times = []
        best_epoch = 0

    # 开始训练
    for epoch in t:
        t.set_description("Training epoch")
        end = time.time()
        train_loss = train(quiet, unmix, encoder, device, train_sampler, optimizer)
        valid_loss = valid(unmix, encoder, device, valid_sampler)
        scheduler.step(valid_loss)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        t.set_postfix(train_loss=train_loss, val_loss=valid_loss)

        stop = es.step(valid_loss)

        if valid_loss == es.best:
            best_epoch = epoch

        utils.save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": unmix.state_dict(),
                "best_loss": es.best,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
            is_best=valid_loss == es.best,
            path=target_path,
            target=target,
        )

        # save params
        params = {
            "epochs_trained": epoch,
            "best_loss": es.best,
            "best_epoch": best_epoch,
            "train_loss_history": train_losses,
            "valid_loss_history": valid_losses,
            "train_time_history": train_times,
            "num_bad_epochs": es.num_bad_epochs,
        }

        with open(Path(target_path, target + ".json"), "w") as outfile:
            outfile.write(json.dumps(params, indent=4, sort_keys=True))

        train_times.append(time.time() - end)

        if stop:
            print("Apply Early Stopping")
            break

if __name__ == '__main__':
    main()