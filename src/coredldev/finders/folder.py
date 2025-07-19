from typing import Any
from .._filepaths._filepaths import padded_spectrograms_path
import h5py as h5
import numpy as np
import pathlib as p
import math
from ..utilites._preprocessing import *
from .._resources.eos_to_index_map import eos_to_index_map
import torch


class FolderFinder:
    def __init__(
        self,
        folder_path=padded_spectrograms_path,
        device="cpu",
        snrs=list(np.linspace(0.5, 2, 100)),
        eos_to_index_map=eos_to_index_map,
    ):
        self.folder_path = folder_path
        self.device = device
        self.snrs = snrs
        self.eos_to_index_map = eos_to_index_map
        self.datapoints = []
        for i in p.Path(self.folder_path).glob("*.pt"):
            for j in self.snrs:
                self.datapoints.append((i, j))
        self.datapoints = np.array(self.datapoints)

    def get_datapoints(self):
        return np.array(self.datapoints), self.eos_to_index_map, None
