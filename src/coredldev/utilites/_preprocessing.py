import sys
import pywt
from os import devnull as print_supression_source
import numpy as np
from scipy.signal import argrelextrema
import math
from behavelet import wavelet_transform
from scipy.ndimage import zoom

scale_min = 1
scale_max = 201
dscale = 0.5
FIN_WIDTH = 400
pad_to_for_planck_window = 90


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(print_supression_source, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def planck_window(j: int, N: int):
    window = np.linspace(0, j - 1, j - 1)
    window[0] = 1
    window = 1.0 / (1.0 + np.exp(j / window - j / (j - window)))
    window[0] = 0
    window = np.concatenate(
        (
            window,
            np.ones((N - (j * 2))),
            np.flip(
                window,
            ),
        )
    )
    return window


def power(signal):
    return np.mean(np.abs(signal) ** 2)


def wt(
    postmerger,
    sam_p,
    getfreqs=False,
    device = None
):
    sam_f = 1 / sam_p
    scales = np.arange(scale_min, scale_max, dscale)

    # CWT on the gwf using the Morlet wavelet
    coefs, freqs = pywt.cwt(postmerger, scales, "morl", sampling_period=sam_p)

    # Normalising the coefficient matrix using the Frobenius norm
    Z = (np.abs(coefs)) / (np.linalg.norm(coefs))
    # Z = Z[:, ::45][:, :400]
    if getfreqs:
        return Z, freqs
    return Z

def noisewt(postmerger, sam_p=1, getfreqs=False):
    postmerger = np.array(postmerger)
    sam_f = 1 / sam_p
    scales = np.arange(1, 201, 0.5)
    coefs, freqs = pywt.cwt(postmerger, scales, "morl", sampling_period=sam_p)
    Z = (np.abs(coefs)) / (np.linalg.norm(coefs))
    if getfreqs:
        return Z, freqs
    return Z


def cut_at_lowest_envelope(
    hplus,
    hcross,
    pm_time,
    getcross=False,
):
    # Cutting inspiral off
    oenv = np.sqrt(hplus**2 + hcross**2)
    cut_point = np.argmax(hplus)
    mhplus = hplus[cut_point:]
    mhcross = hcross[cut_point:]
    mpm_time = pm_time[cut_point:]
    env = oenv[cut_point:]
    envcut = argrelextrema(env, np.less)
    if len(envcut[0]) == 0:
        return mhplus, mhcross, mpm_time
    return mhplus[envcut[0][0] :], mhcross[envcut[0][0] :], mpm_time[envcut[0][0] :]


def numshift_pad_width(Z, l=FIN_WIDTH, shift=0):
    shift = int(shift)
    cwidth = Z.shape[1]
    leftpad = int((l - cwidth) / 2)
    rightpad = int((l - cwidth) / 2)
    fudgepad = rightpad + int(l - (leftpad + rightpad + cwidth))
    assert (abs(shift) <= leftpad) and (
        abs(shift) <= rightpad
    ), "Shift must be less than or equal to the padding on either side"
    padb = np.zeros((l, leftpad + shift))
    pada = np.zeros((l, fudgepad - shift))
    return np.concatenate((padb, Z, pada), axis=1)


def pad_width(Z, l=FIN_WIDTH, percent_shift=0):
    maxshift = int((l - Z.shape[1]) / 2)
    # print(maxshift)
    shift = percent_shift * maxshift if percent_shift != 0 else 0
    # print(shift)
    return numshift_pad_width(Z, shift=shift)


window = lambda ts: planck_window(math.floor(math.log(len(ts) / 2) * 6), len(ts) + 2)

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    a = np.ones((200, 400))
    plt.imshow(a)
    plt.show()
    b = zoom(a, (2, 1))
    plt.imshow(b)
    plt.show()
