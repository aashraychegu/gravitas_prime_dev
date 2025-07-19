import torch
import torch.utils.data as data
import numpy as np
import pathlib as p
from ._filepaths._filepaths import *

# The CoReDatset class is a wrapper around the Source class and the Preprocessor class.
# A Source class implements:
#    the datapoints attribute, which is a list of datapoints
#    the __getitem__ method, which returns a datapoint given an index
# A Preprocessor object is a function or class that implements the __call__ method, which takes a datapoint and returns a processed datapoint.


class CoReDataset(data.Dataset):
    def __init__(
        self,
        source,
        sample_list,
        preprocessor,
    ):
        self.source = source
        self.preprocessor = preprocessor
        self.sample_list = sample_list

    def __getitem__(self, index):
        return self.preprocessor(self.source[self.sample_list[index]])

    def get_raw_item(self, index):
        return self.source[self.sample_list[index]]

    def get_raw_spec(self, index):
        return self.sample_list[index]

    def __len__(self):
        return len(self.sample_list)

    def __repr__(self):
        return f"CoReDataset( source = {self.source}, preprocessor = {self.preprocessor}, len = {len(self)} )"
