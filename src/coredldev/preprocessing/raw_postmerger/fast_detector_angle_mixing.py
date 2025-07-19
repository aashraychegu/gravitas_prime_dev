import numpy as np
from pycbc.detector import Detector
from pycbc.types import TimeSeries
from pycbc.waveform import taper_timeseries

tc = 1242443167


class detector_angle_mixing:
    def __init__(self, detector_name="H1", method="constant", taper_type="startend"):
        """Initialize the detector angle mixing processor
        
        Parameters
        ----------
        detector_name: str
            Name of the detector (e.g. "H1", "L1", "V1")
        method: str
            Method for projecting waveforms ("lal", "constant", or "vary_polarization")
            Default is "constant" which is faster than "lal" for most cases
        taper_type: str
            Type of tapering to apply to waveforms
        """
        self.detector_name = detector_name
        self.detector = Detector(detector_name)
        self.method = method
        self.taper_type = taper_type
        
        # Cache for antenna patterns (key: (ra, dec, pol))
        self._antenna_pattern_cache = {}
        
        # Maximum cache size to prevent memory issues
        self.max_cache_size = 1000
    
    def _get_antenna_patterns(self, ra, dec, pol):
        """Get antenna patterns, using cache if possible"""
        key = (ra, dec, pol)
        if key not in self._antenna_pattern_cache:
            # Limit cache size
            if len(self._antenna_pattern_cache) >= self.max_cache_size:
                self._antenna_pattern_cache.clear()
            
            # Store in cache
            self._antenna_pattern_cache[key] = self.detector.antenna_pattern(
                ra, dec, pol, tc
            )
        
        return self._antenna_pattern_cache[key]
    
    def _fast_project(self, hplus, hcross, ra, dec, pol):
        """Faster projection for 'constant' method"""
        # Get antenna patterns
        fp, fc = self._get_antenna_patterns(ra, dec, pol)
        
        # Get time delay
        dt = self.detector.time_delay_from_earth_center(ra, dec, tc)
        
        # Project directly 
        projected = fp * hplus + fc * hcross
        
        # Create TimeSeries with time shift
        signal = TimeSeries(projected, delta_t=hplus.delta_t, epoch=hplus.start_time + dt)
        return signal
        
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
        # Extract the parameters we need
        hplus_data = inp["hplus"]
        hcross_data = inp["hcross"]
        delta_t = inp["params"]["sam_p"]
        ra, dec, pol = inp["params"]["angle"]
        
        # Create TimeSeries objects
        hplus = TimeSeries(hplus_data, delta_t=delta_t, epoch=tc)
        hcross = TimeSeries(hcross_data, delta_t=delta_t, epoch=tc)
        
        # Apply tapering
        hplus = taper_timeseries(hplus, self.taper_type)
        hcross = taper_timeseries(hcross, self.taper_type)
        
        # Choose fastest projection method based on settings
        if self.method == "constant":
            # Use optimized implementation for constant method
            signal = self._fast_project(hplus, hcross, ra, dec, pol)
        else:
            # Use standard implementation for other methods
            signal = self.detector.project_wave(
                hplus, hcross, ra, dec, pol, 
                method=self.method, reference_time=tc
            )
        
        # Store result and clean up
        inp["signal"] = signal.numpy()
        
        # Remove original data to save memory
        del inp["hplus"]
        del inp["hcross"]
        
        return inp