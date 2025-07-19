from typing import Any
from .._filepaths._filepaths import noduplication_path, allextrationradii_path
import h5py as h5
import numpy as np
import pathlib as p
import math
from ..utilites._preprocessing import *
from .._resources.eos_to_index_map import eos_to_index_map
import torch


class FileFinder:
    def __init__(
        self,
        path=noduplication_path,
        device="cpu",
        snrs=list(np.linspace(0.5, 2, 100)),
        eos_to_index_map=eos_to_index_map,
    ) -> None:
        self.spectrograms, self.parameters = torch.load(path, map_location=device)
        self.length = len(self.spectrograms)
        self.eos_to_index_map = eos_to_index_map
        self.datapoints = []
        for i in range(self.length):
            for j in snrs:
                self.datapoints.append((i, j))

    def get_datapoints(self):
        return np.array(self.datapoints), self.eos_to_index_map, None
