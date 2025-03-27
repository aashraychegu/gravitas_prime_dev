from ..._filepaths._filepaths import CoRe_DB_path
import h5py as h5
from watpy.coredb.coredb import *
import numpy as np
import pathlib as p
import math
from ...utilites._preprocessing import *
from ..._resources.eos_to_index_map import eos_to_index_map
import watpy.utils.units as units
from collections import OrderedDict
import re
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import threading
import weakref

# Precompile regex pattern for extraction radius
RADIUS_PATTERN = re.compile(r'r(\d+)')

class h5Source:
    def __init__(
        self,
        path=CoRe_DB_path,
        selection_attributes=[
            "id_eos",
            "id_mass_starA",
            "id_mass_starB",
        ],
        device="cpu",
        sync=False,
        eos_to_index_map=eos_to_index_map[0],
        max_cache_size=100,  # Maximum number of datasets to cache
        max_file_handles=20,  # Maximum number of open file handles
        prefetch=True,       # Enable prefetching for faster sequential access
        num_workers=2,       # Number of worker threads for prefetching
    ) -> None:
        self.eos_to_index_map = eos_to_index_map
        self.path = path
        self.selection_attributes = selection_attributes
        self.device = device
        self.prefetch = prefetch
        
        # Use OrderedDict to track LRU for both caches
        self._h5_handles = OrderedDict()  # File handles cache
        self._data_cache = OrderedDict()  # Data cache
        self._recently_used = deque()     # Track most recently used files
        self._cache_lock = threading.RLock()  # Thread-safe cache operations
        
        self.max_cache_size = max_cache_size
        self.max_file_handles = max_file_handles
        
        # Setup prefetching if enabled
        self._prefetch_queue = deque()
        if prefetch:
            self._prefetch_executor = ThreadPoolExecutor(max_workers=num_workers)
            self._prefetch_futures = {}  # Track ongoing prefetch tasks
        
        with HiddenPrints():
            if not self.path.exists():
                self.path.mkdir(exist_ok=False)
            cdb = CoRe_db(self.path)
            if sync:
                cdb.sync(verbose=False, lfs=True)
            self.sims = cdb.sim
        
        # Set up finalizer for proper resource cleanup
        weakref.finalize(self, self._cleanup)

    def __getitem__(self, index: int):
        result = self.preprocess(*self.retrieve(index))
        
        # After retrieving current item, prefetch next items
        if self.prefetch and hasattr(index, '__add__'):  # If index can be incremented
            try:
                next_index = index + 1
                self._schedule_prefetch(next_index)
            except:
                pass  # Silently ignore prefetch errors
                
        return result
    
    def _schedule_prefetch(self, index):
        """Schedule prefetching of the given index"""
        if self.prefetch and index not in self._prefetch_futures:
            future = self._prefetch_executor.submit(self._prefetch_item, index)
            self._prefetch_futures[index] = future
    
    def _prefetch_item(self, index):
        """Prefetch item at index into cache"""
        try:
            # Get the PSL dict from the index
            psl = index  # assuming index is actually a PSL dict, adjust if needed
            
            # Create cache keys
            data = self.sims[psl["sim_key"]].run[psl["run_key"]]
            h5path = p.Path(data.data.path) / "data.h5"
            file_key = str(h5path)
            data_key = f"{file_key}:{psl['selected_wf']}:{psl['extraction_radii']}"
            
            # If data already in cache, no need to prefetch
            with self._cache_lock:
                if data_key in self._data_cache:
                    return
                
                # Get file handle from cache or open new one
                if file_key not in self._h5_handles:
                    self._h5_handles[file_key] = h5.File(h5path, "r")
                self._update_file_usage(file_key)
                h5file = self._h5_handles[file_key]
                
                # Extract data (same as in retrieve method)
                metadata = {i: data.md.data[i] for i in self.selection_attributes}
                dataset = h5file[psl["selected_wf"]][psl["extraction_radii"]]
                data_array = dataset[:, [1, 2, -1]].copy()
                hplus, hcross = data_array[:, 0], data_array[:, 1]
                pm_time = units.conv_time(units.geom, units.metric, data_array[:, 2])
                hplus, hcross, pm_time = cut_at_lowest_envelope(hplus, hcross, pm_time=pm_time)
                sam_p = (pm_time[-1] - pm_time[0]) / len(pm_time)
                
                # Cache the data
                if len(self._data_cache) >= self.max_cache_size:
                    self._data_cache.popitem(last=False)
                    
                self._data_cache[data_key] = {
                    "hplus": hplus,
                    "hcross": hcross,
                    "pm_time": pm_time,
                    "metadata": metadata,
                    "sam_p": sam_p
                }
        except Exception as e:
            # Log exception but don't crash the prefetch thread
            print(f"Prefetch error: {e}")
        finally:
            # Clean up futures dict to avoid memory leaks
            if index in self._prefetch_futures:
                del self._prefetch_futures[index]
    
    def _update_file_usage(self, file_key):
        """Update the recently used file list (thread-safe)"""
        with self._cache_lock:
            # Move this file to the end (most recently used)
            if file_key in self._recently_used:
                self._recently_used.remove(file_key)
            self._recently_used.append(file_key)
            
            # Close least recently used files if we have too many open
            while len(self._h5_handles) > self.max_file_handles:
                # Find least recently used file
                for old_key in self._recently_used:
                    if old_key in self._h5_handles:
                        # Close and remove this file handle
                        self._h5_handles[old_key].close()
                        del self._h5_handles[old_key]
                        self._recently_used.remove(old_key)
                        break

    def retrieve(self, psl: dict):
        data = self.sims[psl["sim_key"]].run[psl["run_key"]]
        
        # Create cache keys
        h5path = p.Path(data.data.path) / "data.h5"
        file_key = str(h5path)
        data_key = f"{file_key}:{psl['selected_wf']}:{psl['extraction_radii']}"
        
        # Check if data is already in cache (thread-safe)
        with self._cache_lock:
            if data_key in self._data_cache:
                # Move this item to the end (most recently used)
                self._data_cache.move_to_end(data_key)
                
                cached_data = self._data_cache[data_key]
                hplus, hcross, pm_time = cached_data["hplus"], cached_data["hcross"], cached_data["pm_time"]
                metadata = cached_data["metadata"]
                sam_p = cached_data["sam_p"]
                
                # Data found in cache - return early
                return (
                    hplus, hcross, pm_time, metadata, sam_p, 
                    psl["shifts"], psl["distance"], psl["angles"], 
                    psl["extraction_radii"], psl,
                )
        
        # Data not in cache, load it (thread-safe file handle management)
        with self._cache_lock:
            # Get file handle from cache or open new one
            if file_key not in self._h5_handles:
                self._h5_handles[file_key] = h5.File(h5path, "r")
            
            # Update file usage tracking
            self._update_file_usage(file_key)
            h5file = self._h5_handles[file_key]
        
        # Get metadata
        metadata = {i: data.md.data[i] for i in self.selection_attributes}
        
        # Read dataset - direct slicing for better performance
        dataset = h5file[psl["selected_wf"]][psl["extraction_radii"]]
        
        # Extract data directly to memory with single operation
        # Using fixed column indices is faster than dynamic lookups
        data_array = dataset[:, [1, 2, -1]].copy()
        
        # Extract columns efficiently
        hplus, hcross = data_array[:, 0], data_array[:, 1]
        
        # Convert time in one operation
        pm_time = units.conv_time(units.geom, units.metric, data_array[:, 2])
        
        # Process waveform
        hplus, hcross, pm_time = cut_at_lowest_envelope(hplus, hcross, pm_time=pm_time)
        sam_p = (pm_time[-1] - pm_time[0]) / len(pm_time)
        
        # Cache the data (thread-safe)
        with self._cache_lock:
            if len(self._data_cache) >= self.max_cache_size:
                # Use FIFO for data cache - remove oldest item
                self._data_cache.popitem(last=False)
                
            self._data_cache[data_key] = {
                "hplus": hplus,
                "hcross": hcross,
                "pm_time": pm_time,
                "metadata": metadata,
                "sam_p": sam_p
            }
        
        return (
            hplus,
            hcross,
            pm_time,
            metadata,
            sam_p,
            psl["shifts"],
            psl["distance"],
            psl["angles"],
            psl["extraction_radii"],
            psl,
        )

    def preprocess(self, hplus, hcross, pm_time, params, sam_p, 
                  percent_shift, rescale_to_radii, angle, current_extraction_radii, spec):
        """Optimized preprocessing of waveform data"""
        lts = len(hplus)
        
        # Only pad if necessary
        if lts < pad_to_for_planck_window:
            # Faster padding implementation using np.pad
            pad_front = math.floor(pad_to_for_planck_window - lts)
            pad_back = math.ceil(pad_to_for_planck_window - lts)
            
            # Use np.pad instead of concatenate for better performance
            hplus = np.pad(hplus, (pad_front, pad_back), mode='constant')
            hcross = np.pad(hcross, (pad_front, pad_back), mode='constant')
        
        # Use precompiled regex pattern for extraction radius
        match = RADIUS_PATTERN.search(current_extraction_radii)
        if match:
            clean_xrad = float(match.group(1))
        else:
            # Simplified fallback with fewer string operations
            try:
                parts = current_extraction_radii.split("_")[-1].split(".")
                clean_xrad = float(parts[0][1:])
            except:
                clean_xrad = float(current_extraction_radii.replace("r", "").split("_")[-1].split(".")[0])
        
        # Create params dict at once - use direct assignment for speed
        params_dict = OrderedDict([
            ("eos", self.eos_to_index_map[params["id_eos"]]),
            ("mass_starA", float(params["id_mass_starA"])),
            ("mass_starB", float(params["id_mass_starB"])),
            ("percent_shift", float(percent_shift)),
            ("rescale_to_radii", float(rescale_to_radii)),
            ("angle", angle),
            ("sam_p", float(sam_p)),
            ("current_extraction_radius", clean_xrad),
            ("spec", spec)
        ])
        
        # If device is not CPU, move arrays to the specified device
        if self.device != "cpu" and hasattr(hplus, "to"):  # For PyTorch tensors
            hplus = hplus.to(self.device)
            hcross = hcross.to(self.device)
            pm_time = pm_time.to(self.device)
        
        return OrderedDict([
            ("hplus", hplus),
            ("hcross", hcross),
            ("pm_time", pm_time),
            ("params", params_dict)
        ])

    def close(self):
        """Close all file handles and clear caches"""
        for f in self._h5_handles.values():
            f.close()
        self._h5_handles.clear()
        self._data_cache.clear()
        self._recently_used.clear()
        
        # Clean up prefetching resources

    def _cleanup(self):
        """Cleanup resources when object is garbage collected"""
        try:
            # Close all file handles
            for f in list(self._h5_handles.values()):
                try:
                    f.close()
                except:
                    pass  # Ignore errors during cleanup
            
            # Clear caches
            self._h5_handles.clear()
            self._data_cache.clear()
            self._recently_used.clear()
            
            # Clean up executor if exists
            if self.prefetch and hasattr(self, '_prefetch_executor'):
                self._prefetch_executor.shutdown(wait=False)
        except:
            pass  # Ensure cleanup doesn't raise exceptions
