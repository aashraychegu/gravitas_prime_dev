import numpy as np
from ...utilites._preprocessing import planck_window, window
class time_shift:
    def __init__(self, length=40*500):
        self.length = length
        
    def __call__(self, data):
        base = np.zeros(self.length)
        percent_shift = data["params"]["percent_shift"]+100
        max_shift = (self.length - len(data["signal"]))/2
        shift = int(percent_shift/100 * max_shift)
        base[shift:shift+len(data["signal"])] = data["signal"] #*window(data["signal"])
        data["signal"] = base
        return data
