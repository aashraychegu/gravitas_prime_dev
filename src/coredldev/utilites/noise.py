from .._filepaths._filepaths import *
import matplotlib
import numpy


from scipy.interpolate import interp1d
# import logging
from pycbc.types.timeseries import TimeSeries
from lal import LIGOTimeGPS

# logging.basicConfig(level=logging.INFO,
#                     format='%(asctime)s %(message)s',
#                     datefmt='%Y-%m-%d %H:%M:%S')

def create_white_noise(sampling_frequency, duration):
    # logging.info('Generating white noise')
    number_of_samples = int(sampling_frequency * duration)
    # rfftfreq calculates the frequencies corresponding to the positive half of the complex-valued spectrum obtained after applying the real FFT (rfft()) to a real-valued input signal
    frequencies = numpy.fft.rfftfreq(number_of_samples, 1 / sampling_frequency)
    sigma = numpy.sqrt(duration / 4)

    white_noise_real = numpy.random.normal(0, sigma, len(frequencies))
    white_noise_imag = numpy.random.normal(0, sigma, len(frequencies))
    white_noise = white_noise_real + 1j * white_noise_imag
    
    return white_noise, frequencies

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

def get_time_domain_strain(frequency_domain_strain, sampling_frequency = 4096):
    '''
    Convert from frequency domain strain to time domain strain
    '''
    time_domain_strain_norm = numpy.fft.irfft(frequency_domain_strain)
    time_domain_strain = time_domain_strain_norm * sampling_frequency
    return time_domain_strain

def get_frequency_domain_strain(time_domain_strain, sampling_frequency = 4096):
    '''
    Convert time domain strain to frequency domain strain
    '''
    frequency_domain_strain = numpy.fft.rfft(time_domain_strain)
    frequency_domain_strain = frequency_domain_strain / sampling_frequency
    frequency_array = numpy.linspace(0, sampling_frequency/2, num=len(frequency_domain_strain))

def to_pycbc_timeseries(time_domain_strain, sampling_frequency=4096, start_time = 0):
    '''
    Convert to PyCBC TimeSeries. Easier to add signals.
    '''
    return TimeSeries(time_domain_strain, 
                      epoch=LIGOTimeGPS(start_time),
                      delta_t=1./sampling_frequency)

# Verify the correctness of the code
from pycbc.psd import interpolate, inverse_spectrum_truncation, welch
def get_psd(strain,
            strain_high_pass = 5):
        
    psd_estimation = 'median'
    psd_segment_length = 16
    psd_segment_stride = 8
    psd_inverse_length = 16
    psd_num_segments = 63
    psd_duration = 4
    psd_stride = 2
    

    psd = welch(strain, avg_method=psd_estimation,
                seg_len=int(psd_segment_length * strain.sample_rate + 0.5),
                seg_stride=int(psd_segment_stride * strain.sample_rate + 0.5),
                num_segments=psd_num_segments,
                require_exact_data_fit=False)
    
    psd = interpolate(psd, 1. / strain.duration)
    psd = inverse_spectrum_truncation(psd,
                                      int(psd_inverse_length * strain.sample_rate),
                                      low_frequency_cutoff=strain_high_pass,
                                      trunc_method='hann')
    return psd
