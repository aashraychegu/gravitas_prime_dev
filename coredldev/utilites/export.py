import torch
from ..dataset import CoReDataset
import shutil
import pathlib as p


def to_folder(dataset: CoReDataset, folder: p.Path):
    if folder.is_dir():
        shutil.rmtree(folder)
    folder.mkdir()
    for i, dp in enumerate(dataset):
        lparams = list(dp[1])
        path = (
            folder
            / f"{i}__eos_{lparams[0]}__m1_{lparams[1]}__m2_{lparams[2]}__pwr_{lparams[3]}__shft_{lparams[4]}.pt"
        )
        torch.save(
            dp,
            path,
        )
        print(f"saved: {i} at {path} with params {lparams}")
    return folder


def to_pth(dataset: CoReDataset, path: p.Path):
    sgrams = []
    params = []
    for i, (sgram, param) in enumerate(dataset):
        sgrams.append(sgram)
        params.append(param)
        print("saved: ", i)
    torch.save(torch.stack(sgrams), torch.stack(params), path)
    return path
