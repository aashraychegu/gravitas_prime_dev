from ...utilites._preprocessing import power, noisewt
from ..._filepaths._filepaths import ligopsd_path
import pycbc
import torch
import numpy as np

psd = pycbc.types.load_frequencyseries(ligopsd_path)
delta_t = 1.0 / (4096 * 2)
tsamples = 400


def custom_noise(device="cpu"):
    def inner(a, *args):
        tensor, params = a[0], a[1]
        tensor_power = params[3]
        snr = params[-1]
        noise = np.array(pycbc.noise.noise_from_psd(tsamples, delta_t, psd, seed=42))
        noise_power = power(noise)
        if snr == 0:
            return tensor
        noise = noisewt(noise) * (float(tensor_power) / float(noise_power)) / float(snr)
        return tensor.to(device) + torch.tensor(noise).to(device)

    return inner
