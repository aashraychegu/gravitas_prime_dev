from ..._filepaths._filepaths import ligopsd_path
from ...utilites._preprocessing import pad_width, window, planck_window
import numpy as np
from scipy.interpolate import interp1d
import functools
import numpy

# Cache for PSD interpolation objects
_psd_interp_cache = {}
# Cache for generated noise
_noise_cache = {}

def create_white_noise(sampling_frequency, duration):
    # logging.info('Generating white noise')
    number_of_samples = int(sampling_frequency * duration)
    # rfftfreq calculates the frequencies corresponding to the positive half of the complex-valued spectrum obtained after applying the real FFT (rfft()) to a real-valued input signal
    frequencies = np.fft.rfftfreq(number_of_samples, 1 / sampling_frequency)
    sigma = np.sqrt(duration / 4)

    white_noise_real = np.random.normal(0, sigma, len(frequencies))
    white_noise_imag = np.random.normal(0, sigma, len(frequencies))
    white_noise = white_noise_real + 1j * white_noise_imag
    
    return white_noise, frequencies

@functools.lru_cache(maxsize=8)
def _get_psd_interpolation(psd_file):
    """
    Load PSD file and create interpolation function, with caching
    """
    frequency_array, psd_array = np.genfromtxt(psd_file).T
    if np.min(psd_array) > 1e-30:
        # logging.warning('You provided an ASD File')
        # logging.info('Generating PSD from ASD')
        psd_array = np.power(psd_array, 2)
        
    return frequency_array, psd_array

def generate_colored_noise(psd_file, sampling_frequency = 4096, duration = 512):
    frequency_array, psd_array = numpy.genfromtxt(psd_file).T
    if numpy.min(psd_array) > 1e-30:
        # logging.warning('You provided an ASD File')
        # logging.info('Generating PSD from ASD')
        psd_array = numpy.power(psd_array, 2)
    elif numpy.min(psd_array) < 1e-30:
        # logging.info('You provided a PSD file. Awesome!')
        pass
    white_noise, frequencies = create_white_noise(sampling_frequency, duration)
    
    # interpolate the PSD so that it can be sampled to any frequency array. 
    psd_interp1d = interp1d(frequency_array, 
                            psd_array,
                            bounds_error=False,
                            fill_value=numpy.inf)
    
    frequency_domain_strain = white_noise * numpy.sqrt(psd_interp1d(frequencies))
    outside = numpy.logical_or(frequencies < numpy.min(frequency_array), frequencies > numpy.max(frequency_array))
    frequency_domain_strain[outside] = 0
    return frequency_domain_strain, frequencies

def get_time_domain_strain(frequency_domain_strain, sampling_frequency=4096):
    '''
    Convert from frequency domain strain to time domain strain
    '''
    time_domain_strain_norm = np.fft.irfft(frequency_domain_strain)
    time_domain_strain = time_domain_strain_norm * sampling_frequency
    return time_domain_strain

def get_frequency_domain_strain(time_domain_strain, sampling_frequency=4096):
    '''
    Convert time domain strain to frequency domain strain
    '''
    frequency_domain_strain = np.fft.rfft(time_domain_strain)
    frequency_domain_strain = frequency_domain_strain / sampling_frequency
    frequency_array = np.linspace(0, sampling_frequency/2, num=len(frequency_domain_strain))
    return frequency_domain_strain, frequency_array


class NoiseInjection1D:
    def __init__(self, debug=False, passthrough=False, cache_size=600,psd_file = ligopsd_path):
        self.debug = debug
        self.passthrough = passthrough
        self.cache_size = cache_size
        self.noise_cache = {}  # Cache for noise samples
        self.psd_file = psd_file

    def __call__(self, inp, size=40 * 500):
        signal = inp["signal"]
        sam_p = inp["params"]["sam_p"]
        
        # Make sure the input signal is of the correct size
        if len(signal) != size:
            if len(signal) < size:
                # Pad the signal if it's too short
                padded_signal = np.zeros(size)
                padded_signal[:len(signal)] = signal
                inp["signal"] = padded_signal
            else:
                # Truncate the signal if it's too long
                inp["signal"] = signal[:size]
        
        # Recalculate duration based on the specified size and sampling parameter
        duration = size * sam_p
        
        # Create cache key based on sampling params and desired size
        cache_key = (sam_p, size)
        
        # Check if we have this noise configuration cached
        if cache_key not in self.noise_cache:
            # If cache is full, remove oldest entry
            if len(self.noise_cache) >= self.cache_size:
                oldest_key = next(iter(self.noise_cache))
                del self.noise_cache[oldest_key]
                
            # Generate new noise and cache it
            fnoise = generate_colored_noise(
                psd_file=self.psd_file,
                sampling_frequency=1 / sam_p,
                duration=duration,
            )[0]
            self.noise_cache[cache_key] = fnoise
        else:
            fnoise = self.noise_cache[cache_key]
        
        tds = get_time_domain_strain(fnoise, sampling_frequency=1 / sam_p)
        
        # Ensure the time domain strain matches the target size
        if len(tds) != size:
            if len(tds) < size:
                tds = np.pad(tds, (0, size - len(tds)))
            else:
                tds = tds[:size]
        
        if not self.passthrough:
            inp["signal"] += tds
            inp["signal"] *= planck_window(50,len(inp["signal"])+2)
        if self.debug:
            return inp, fnoise, tds
        return inp