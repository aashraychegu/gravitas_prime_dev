# Standard library imports
import warnings
import time
from pprint import pprint

# Third-party library imports
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import cupy as cp
import tqdm
from scipy.interpolate import interp1d
import torchaudio
import torchaudio.functional as AF
import torchaudio.transforms as AT

# Suppress warnings
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")

# Local imports - core functionality
from coredldev.dataset import CoReDataset
from coredldev.utilites.pipeline import pipeline

# Local imports - finders and sources
from coredldev.finders.distance_scaling.h5 import h5Finder
from coredldev.sources.distance_scaling.h5 import h5Source

# Local imports - preprocessing steps
from coredldev.preprocessing.raw_postmerger.detector_angle_mixing import DetectorAngleMixing
from coredldev.preprocessing.raw_postmerger.fast_detector_angle_mixing import detector_angle_mixing as fastdam
from coredldev.preprocessing.raw_postmerger.distance_scale import distance_scale
from coredldev.preprocessing.raw_postmerger.time_shift import time_shift
from coredldev.preprocessing.to_tensor import to_tensor_clean
from coredldev.preprocessing.ligo_noise.inject_noise import NoiseInjection1D as noise_injection_1d
from coredldev.preprocessing.wavelet_transforms.morlet import MorletWaveletTransform
from coredldev.preprocessing.whiten import TimeSeriesWhitener

from pycbc.types import TimeSeries

# Import PyCBC libraries
import pycbc.noise
import pycbc.psd
import pycbc.filter

import scipy
from scipy.signal import welch
# Load frequency values for plotting
freqs = np.genfromtxt("fspace.npy")

# UPDATE CORE DATABASE
import torch.utils.data as data
import os
import h5py

datapoints, eosmap, remaining = h5Finder(shiftpercents=[0],angles=[(0,0,0)],distances = [1]).get_datapoints()
source = h5Source(eos_to_index_map=eosmap)
dataset = CoReDataset(source, datapoints,lambda x: x)
transformed_dataset = CoReDataset(source, datapoints,pipeline([DetectorAngleMixing(),distance_scale(),time_shift(),noise_injection_1d(),TimeSeriesWhitener(4,2),to_tensor_clean()]))

# Determine optimal device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Move Morlet transform to GPU if available
morl = MorletWaveletTransform(freqs).to(device)

# Determine optimal number of workers based on CPU count
num_workers = min(16, os.cpu_count() or 4)

# Create DataLoader with optimized settings
dataloader = data.DataLoader(
    dataset=transformed_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,  # Speeds up host to GPU transfers
    persistent_workers=True if num_workers > 0 else False,  # Keep workers alive between iterations
    prefetch_factor=2 if num_workers > 0 else None  # Prefetch batches for better efficiency
)

h5_file_path = f'./transformed_maps.h5'

# Higher compression level (9 is maximum for gzip)
compression_level = 9
# Check if the file already exists
if os.path.exists(h5_file_path):
    print(f"Warning: {h5_file_path} already exists. It will be overwritten.")
    os.remove(h5_file_path)
h5_file = h5py.File(h5_file_path, 'w')
# Create the HDF5 file

# Estimate total number of samples
total_samples = len(transformed_dataset)
print(f"Total samples to process: {total_samples}")

# Set fixed dimensions for the dataset as specified
data_shape = (total_samples, 400, 400)
metadata_shape = (total_samples, 9)

print(f"Creating main dataset with shape: {data_shape}")
print(f"Creating metadata dataset with shape: {metadata_shape}")

# Create datasets with predefined shapes
main_dataset = h5_file.create_dataset(
    "transformed_maps",
    shape=data_shape,
    dtype=np.float32,
    compression="gzip",
    compression_opts=compression_level,
    chunks=(1, 400, 400)  # Chunk by individual samples for efficient access
)

# Create a dataset to store metadata with exactly 9 values per sample
metadata_dataset = h5_file.create_dataset(
    "metadata",
    shape=metadata_shape,
    dtype=np.float32,
    compression="gzip",
    compression_opts=compression_level
)

# Memory monitoring function
def check_memory():
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        memory_reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        return f"GPU Memory: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved"
    return "GPU not available"

# Process all batches efficiently
sample_counter = 0
with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):  # Use mixed precision if available
    for batch_idx, (batch, datapoint) in enumerate(tqdm.tqdm(dataloader, desc="Processing batches")):
        # Move batch to device
        batch = batch.to(device)
        transformed = morl(batch)
        batch_numpy = transformed.detach().cpu().numpy()
        dp = datapoint.detach().cpu().numpy()
        
        for sgram,metadata in zip(batch_numpy, dp):
            # Get single sample Store in the dataset
            main_dataset[sample_counter] = sgram
            metadata_dataset[sample_counter] = metadata
            sample_counter += 1
        
        # Print memory usage and clear GPU cache periodically
        if batch_idx % 10 == 0:
            print(check_memory())
            # Flush the HDF5 file to disk periodically
            h5_file.flush()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

# Add attributes with information about the dataset
h5_file.attrs["total_samples"] = sample_counter
h5_file.attrs["data_shape"] = str(data_shape)
h5_file.attrs["metadata_shape"] = str(metadata_shape)
h5_file.attrs["compression_level"] = compression_level
h5_file.attrs["creation_date"] = time.strftime("%Y-%m-%d %H:%M:%S")

# Close the HDF5 file
h5_file.close()

print(f"Processed {sample_counter} samples total")
print(f"All data saved to single HDF5 file: {h5_file_path}")
print(f"Dataset layout:")
print(f"  - Main dataset 'transformed_maps' with shape {data_shape}")
print(f"  - Metadata dataset 'metadata' with shape {metadata_shape}")
print(f"Compression level used: {compression_level}")