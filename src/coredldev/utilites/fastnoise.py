
from behavelet import wavelet_transform
from behavelet.morlet import _morlet_fft_convolution, _morlet_fft_convolution_parallel

from .._filepaths._filepaths import freqs_path

import cupy as cp

import numpy as np

import math

freqs = np.genfromtxt(freqs_path)

def wavelet_transform(X, freqs = freqs, fsample = 16392*8, sam_p = None,
                      prob=True, omega0=6.0, log_scale=True,
                      n_jobs=1, gpu=False,device = 0):
    
    if sam_p is not None:
        fsample = 1/sam_p

    with cp.cuda.Device(device):

        if(len(X.shape) == 1):
            X = X[:,np.newaxis]

        if gpu is True and cp is None:
            gpu = False
            print("not using cupy!")

        X = X.astype(np.float32)

        dtime = 1. / fsample

        scales = (omega0 + np.sqrt(2 + omega0**2)) / (4 * np.pi * freqs)
        
        feed_dicts = [{"X": feature,
                    "freqs": freqs,
                    "scales": scales,
                    "dtime": dtime,
                    "omega0": omega0,
                    "gpu": gpu}
                    for feature in X.T]
        
        convolved = list(map(_morlet_fft_convolution_parallel, feed_dicts))

        X_new = np.concatenate(convolved, axis=1)

        if gpu:
            mempool = cp.get_default_memory_pool()
            pinned_mempool = cp.get_default_pinned_memory_pool()
            mempool.free_all_blocks()
            pinned_mempool.free_all_blocks()

        del convolved, scales, feed_dicts, X
        
        return (np.abs(X_new)/np.linalg.norm(X_new)).T

