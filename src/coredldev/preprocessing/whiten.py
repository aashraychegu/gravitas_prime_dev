import numpy as np
import scipy

class TimeSeriesWhitener:
    """
    A class to whiten time series data stored in numpy arrays.
    Uses only numpy, scipy, and pytorch for implementation.
    """
    
    def __init__(self,segment_duration=4, overlap_duration=2, low_freq_cutoff=30):
        """
        Initialize the whitener with configuration parameters.
        
        Args:
            low_freq_cutoff: Frequency below which to zero out the PSD (Hz)
            segment_duration: Duration of segments for PSD estimation (seconds)
            overlap_duration: Duration of overlap between segments (seconds)
        """
        self.low_freq_cutoff = low_freq_cutoff
        self.segment_duration = segment_duration
        self.overlap_duration = overlap_duration
        
    def _estimate_psd(self, data, sample_rate):
        """
        Estimate PSD using Welch's method from scipy.
        """
        # Calculate NFFT and noverlap based on segment and overlap durations
        nfft = int(self.segment_duration)
        noverlap = int(self.overlap_duration)
        # Use scipy's welch function to estimate PSD
        freqs, psd = scipy.signal.welch(
            data,
            fs=sample_rate, 
            nperseg=nfft,
            noverlap=noverlap,
            return_onesided=True
        )
        
        # Apply low frequency cutoff
        psd[freqs < self.low_freq_cutoff] = np.inf
        
        return freqs, psd
    
    def whiten_numpy(self, data, sample_rate):
        """
        Whiten the signal using numpy/scipy operations
        
        Args:
            data: Input time series as numpy array
            sample_rate: Sample rate in Hz (1/sample_period)
            
        Returns:
            Whitened time series as numpy array
        """
        # Get data length
        n = len(data)
        
        # Estimate PSD
        freqs, psd = self._estimate_psd(data, sample_rate)
        
        # FFT the signal
        fft_data = np.fft.rfft(data)
        
        # Create frequency array for FFT bins
        fft_freqs = np.fft.rfftfreq(n, d=1.0/sample_rate)
        
        # Interpolate PSD to match FFT frequency bins
        psd_interp = np.interp(fft_freqs, freqs, psd)
        
        # Prevent division by zero or very small numbers
        psd_interp = np.where(psd_interp > 0, psd_interp, np.inf)
        
        # Apply whitening
        whitened_fft = fft_data / np.sqrt(psd_interp) / (2*np.pi*(20_000)*sample_rate+1)
        
        # Inverse FFT to get back to time domain
        whitened_data = np.fft.irfft(whitened_fft, n)
        
        return whitened_data


    def symlog_normalize(self, data, linthresh=4):
        """
        Apply symmetric log normalization to data while preserving sign.
        
        Args:
            data: Input signal array
            linthresh: Threshold for linear region around zero
            epsilon: Small constant to avoid log(0)
            
        Returns:
            Normalized signal with values between [-1, 1]
        """
        sign = np.sign(data)
        abs_data = np.abs(data)
        
        # Apply linear scaling for small values, log scaling for larger values
        result = np.zeros_like(abs_data)
        
        # Linear region
        linear_mask = abs_data < linthresh
        result[linear_mask] = abs_data[linear_mask] / linthresh
        
        # Log region
        log_mask = ~linear_mask
        result[log_mask] = 1.0 + np.log10(abs_data[log_mask] / linthresh) / np.log10(abs_data.max() / linthresh)
        
        # Restore sign
        normalized = sign * result
        
        return normalized

    def __call__(self, data_dict, duration=None, overlap=None,linthresh=1e-5):
        """
        Apply whitening to data["signal"]
        
        Args:
            data_dict: Dictionary containing signal and parameters
            
        Returns:
            Updated data dictionary with whitened signal
        """

        # Get sampling period and calculate rate
        sam_p = data_dict["params"]["sam_p"]
        sample_rate = 1.0 / sam_p
        data_dict["signal"] = self.whiten_numpy(data_dict["signal"], sample_rate) # self.symlog_normalize(self.whiten_numpy(data_dict["signal"], sample_rate),linthresh)

        return data_dict
