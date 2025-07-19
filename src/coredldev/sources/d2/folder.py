from typing import Any
from ..._filepaths._filepaths import padded_spectrograms_path
import h5py as h5
import numpy as np
import pathlib as p
import math
from ...utilites._preprocessing import *
from ..._resources.eos_to_index_map import eos_to_index_map
import torch


class FolderSource:
    def __init__(
        self, path=padded_spectrograms_path, device="cpu", eos_to_index_map=None
    ) -> None:
        self.file_list = list(p.Path(path).glob("*.pt"))
        self.length = len(self.file_list)
        self.device = device

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        spectrogram, parameters = torch.load(index[0], map_location=self.device)
        parameters[-1] = index[1]
        return (spectrogram, parameters)
