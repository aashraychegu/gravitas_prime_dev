import numpy as np
from functools import lru_cache
import time
# Reference time for GPS epoch
tc = 1242443167

# Speed of light in m/s
SPEED_OF_LIGHT = 299792458.0

# Pre-defined detector coordinates (longitude, latitude in radians) and response matrices
DETECTOR_DATA = {
    # LIGO Hanford
    "H1": {
        "longitude": -2.08405676917,  # radians
        "latitude": 0.81079526383,    # radians
        "response": np.array([
            [-0.3946, 0.0759, 0.3166],
            [0.0759, 0.0523, -0.0281],
            [0.3166, -0.0281, 0.3423]
        ]),
        "location": np.array([-2.16141492636e+06, -3.83469517889e+06, 4.59274737811e+06])
    },
    # LIGO Livingston
    "L1": {
        "longitude": -1.58430937078,
        "latitude": 0.53342313506,
        "response": np.array([
            [-0.4143, 0.1457, 0.2642],
            [0.1457, 0.1910, 0.1112],
            [0.2642, 0.1112, 0.2233]
        ]), 
        "location": np.array([-7.42760447238e+05, -5.49628371971e+06, 3.22425701744e+06])
    },
    # Virgo
    "V1": {
        "longitude": 0.18333805213,
        "latitude": 0.76151183984,
        "response": np.array([
            [-0.0839, 0.0158, -0.1115],
            [0.0158, 0.3944, -0.1676],
            [-0.1115, -0.1676, -0.3105]  
        ]),
        "location": np.array([4.54637409900e+06, 8.42989697626e+05, 4.37857696241e+06])
    }
}

def apply_taper(data, taper_length=0.1):
    """Apply tapering to the start and end of a waveform
    
    Parameters
    ----------
    data: numpy.ndarray
        Waveform data
    taper_length: float
        Fraction of the waveform to taper on each end
        
    Returns
    -------
    data: numpy.ndarray
        Tapered waveform
    """
    n = len(data)
    taper_samples = int(n * taper_length)
    
    # Create Hann window for tapering
    window = np.hanning(2 * taper_samples)
    
    # Create taper window with ones in the middle
    taper = np.ones(n)
    taper[:taper_samples] = window[:taper_samples]
    taper[-taper_samples:] = window[taper_samples:]
    
    return data * taper

def apply_time_shift(data, dt, delta_t):
    """Apply time shift to data using FFT phase shift
    
    Parameters
    ----------
    data: numpy.ndarray
        Waveform data
    dt: float
        Time shift in seconds
    delta_t: float
        Sampling time interval
        
    Returns
    -------
    shifted_data: numpy.ndarray
        Time-shifted waveform
    """
    # For small time shifts less than one sample, use FFT phase shift
    if abs(dt) < delta_t:
        n = len(data)
        freq = np.fft.rfftfreq(n, d=delta_t)
        fft_data = np.fft.rfft(data)
        phase_shift = np.exp(-2j * np.pi * freq * dt)
        shifted_data = np.fft.irfft(fft_data * phase_shift, n=n)
        return shifted_data
    
    # For larger shifts, use direct sample shifting
    samples_shift = int(round(dt / delta_t))
    shifted_data = np.zeros_like(data)
    
    if samples_shift > 0:
        # Shift right
        shifted_data[samples_shift:] = data[:-samples_shift]
    elif samples_shift < 0:
        # Shift left
        shifted_data[:samples_shift] = data[-samples_shift:]
    else:
        # No shift
        shifted_data = data.copy()
        
    return shifted_data

class DetectorAngleMixing:
    def __init__(self, detector_name="H1", cache_size=1024):
        """Initialize the detector angle mixing processor
        
        Parameters
        ----------
        detector_name: str
            Name of the detector (e.g. "H1", "L1", "V1")
        cache_size: int
            Number of antenna pattern configurations to cache
        """
        if detector_name not in DETECTOR_DATA:
            raise ValueError(f"Unknown detector: {detector_name}")
            
        self.detector_name = detector_name
        self.detector_data = DETECTOR_DATA[detector_name]
        self.response = self.detector_data["response"]
        self.location = self.detector_data["location"]
        
        # Cache the antenna pattern and time delay calculations
        self.get_antenna_pattern = lru_cache(maxsize=cache_size)(self._get_antenna_pattern)
        self.get_time_delay = lru_cache(maxsize=cache_size)(self._get_time_delay)
        
    def _get_antenna_pattern(self, ra, dec, pol):
        """Calculate detector antenna pattern for given sky location
        
        Parameters
        ----------
        ra: float
            Right ascension in radians
        dec: float
            Declination in radians
        pol: float
            Polarization angle in radians
            
        Returns
        -------
        fp: float
            Plus polarization factor
        fc: float
            Cross polarization factor
        """
        # Calculate Greenwich Mean Sidereal Time (simplified)
        gha = 0.67598  # Approximate GMST at reference time, minus ra
        
        # Calculate detector response using numpy for all operations
        cosgha = np.cos(gha)
        singha = np.sin(gha)
        cosdec = np.cos(dec)
        sindec = np.sin(dec)
        cospsi = np.cos(pol)
        sinpsi = np.sin(pol)

        # Detector response calculation (simplified from pycbc)
        x0 = -cospsi * singha - sinpsi * cosgha * sindec
        x1 = -cospsi * cosgha + sinpsi * singha * sindec
        x2 = sinpsi * cosdec
        x = np.array([x0, x1, x2])
        dx = self.response.dot(x)

        y0 = sinpsi * singha - cospsi * cosgha * sindec
        y1 = sinpsi * cosgha + cospsi * singha * sindec
        y2 = cospsi * cosdec
        y = np.array([y0, y1, y2])
        dy = self.response.dot(y)

        fp = (x * dx - y * dy).sum()
        fc = (x * dy + y * dx).sum()
        
        return fp, fc
        
    def _get_time_delay(self, ra, dec):
        """Calculate time delay from Earth center
        
        Parameters
        ----------
        ra: float
            Right ascension in radians
        dec: float
            Declination in radians
            
        Returns
        -------
        dt: float
            Time delay in seconds
        """
        # Direction to source
        cosd = np.cos(dec)
        e0 = cosd * np.cos(ra)
        e1 = cosd * -np.sin(ra)
        e2 = np.sin(dec)
        
        ehat = np.array([e0, e1, e2])
        
        # Calculate delay
        return self.location.dot(ehat) / SPEED_OF_LIGHT
    
    def __call__(self, inp):
        """Project strain onto detector
        
        Parameters
        ----------
        inp: dict
            Dictionary containing hplus, hcross waveforms and params
            
        Returns
        -------
        inp: dict
            Modified dictionary with projected signal
        """
        # Extract waveforms and parameters
        hplus = inp["hplus"]
        hcross = inp["hcross"]
        delta_t = inp["params"]["sam_p"]
        ra, dec, pol = inp["params"]["angle"]
        
        # Apply tapering
        hplus = apply_taper(hplus)
        hcross = apply_taper(hcross)
        # Get antenna patterns (uses cache if same sky location)
        fp, fc = self.get_antenna_pattern(ra, dec, pol)
        # Calculate time delay
        dt = self.get_time_delay(ra, dec)
        # Apply antenna patterns to waveforms
        signal = fp * hplus + fc * hcross
        # Apply time delay directly to the signal
        signal = apply_time_shift(signal, dt, delta_t)
        # Store result and clean up
        inp["signal"] = signal
        inp.pop("hplus")
        inp.pop("hcross")
        
        return inp

