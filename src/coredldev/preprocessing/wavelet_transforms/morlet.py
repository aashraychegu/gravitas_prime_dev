
import torch
import torch.nn.functional as F
import numpy as np

class MorletWaveletTransform(torch.nn.Module):

    def __init__(self, freqs, sam_p=1 / (16392 * 8), len_signal=20_000):
        """
        Parameters:
        Freqs: Central frequency of the wavelet.
        """
        super(MorletWaveletTransform, self).__init__()
        t = np.arange(-len_signal // 2, len_signal // 2)
        freqs = np.array(freqs)

        self.wavelets = [
            torch.fft.fft(
                self.generate_morlet_wavelet(t * sam_p, f0=freq),
                n=len_signal,
            )
            for freq in freqs
        ]

    def generate_morlet_wavelet(self, t, f0):
        """
        Generate a Morlet wavelet.

        Parameters:
        t (numpy array): Time array.

        Returns:
        torch.Tensor: Morlet wavelet.
        """
        sigma = 1.0 / (2 * np.pi * (f0))
        wavelet = np.exp(2j * np.pi * f0 * t) * np.exp(-(t**2) / (2 * sigma**2))
        return torch.tensor(wavelet, dtype=torch.complex64).cuda()

    def forward(self, signal):
        """
        This function is called during computation
        """
        with torch.no_grad():
            signal_tensor = signal.to(torch.complex64).cuda()
            sig_fft = torch.fft.fft(signal_tensor, dim=1)
            out = torch.real(
                torch.stack(
                    [
                        torch.fft.ifft(sig_fft * wavelet, dim=1)[::, ::50].squeeze(1)
                        for wavelet in self.wavelets
                    ],
                    dim=1,
                )
            )
            return out

    def _wavelet(self, signal):
        """
        This function is called during computation
        """
        with torch.no_grad():
            signal_tensor = signal.to(torch.complex64).cuda()
            sig_fft = torch.fft.fft(signal_tensor, dim=1)
            out = torch.real(
                torch.stack(
                    [
                        torch.fft.ifft(sig_fft * wavelet, dim=1).squeeze(1)
                        for wavelet in self.wavelets
                    ],
                    dim=1,
                )
            )
            return out

if __name__ == "__main__":
    signal = torch.randn(10, 20_000)
    freqs = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    morlet = MorletWaveletTransform(freqs)
    out = morlet(signal)
    print(out.shape)