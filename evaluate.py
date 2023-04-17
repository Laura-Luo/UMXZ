import musdb
import museval
import torch
from openunmix import utils
from typing import Optional, Union
import functools
import json

# 写进kaggle

def separate_and_evaluate(
    track: musdb.MultiTrack,
    targets: list,
    model_str_or_path: str,
    niter: int,
    output_dir: str,
    eval_dir: str,
    residual: bool,
    mus,
    aggregate_dict: dict = None,
    device: Union[str, torch.device] = "cpu",
    wiener_win_len: Optional[int] = 300,
    filterbank="torch",
) -> str:

    separator = utils.load_separator(
        model_str_or_path=model_str_or_path,
        targets=targets,
        niter=niter,
        residual=residual,
        wiener_win_len=wiener_win_len,
        device=device,
        pretrained=True,
        filterbank=filterbank,
    )

    separator.freeze()
    separator.to(device)

    audio = torch.as_tensor(track.audio, dtype=torch.float32, device=device) # !!
    audio = utils.preprocess(audio, track.rate, separator.sample_rate)

    estimates = separator(audio)
    estimates = separator.to_dict(estimates, aggregate_dict=aggregate_dict)

    for key in estimates:
        estimates[key] = estimates[key][0].cpu().detach().numpy().T
    if output_dir:
        mus.save_estimates(estimates, track, output_dir)

    scores = museval.eval_mus_track(track, estimates, output_dir=eval_dir)
    return scores


root=  # Path to MUSDB18
subset  # MUSDB subset (`train`/`test`)
is_wav= True
targets=
model =  # path to mode base directory of pretrained models
outdir   #Results path where audio evaluation results are stored
evaldir=  # Results path for museval estimates

mus = musdb.DB(
    root=root,
    download=root is None,
    subsets=subset,
    is_wav=is_wav,
)


aggregate_dict = None
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

results = museval.EvalStore()
for track in tqdm.tqdm(mus.tracks):#
    scores = separate_and_evaluate(
        track,
        targets=targets,
        model_str_or_path=model,
        niter=1,
        residual=False,
        mus=mus,
        aggregate_dict=aggregate_dict,
        output_dir=outdir,
        eval_dir=evaldir,
        device=device,
    )
    print(track, "\n", scores)
    results.add_track(scores)

print(results)
method = museval.MethodStore()
method.add_evalstore(results, model)
method.save(model + ".pandas")