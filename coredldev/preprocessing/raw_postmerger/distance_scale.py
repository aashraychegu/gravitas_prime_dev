import numpy as np
import watpy.utils.units as units
from pycbc.waveform import get_td_waveform

from watpy.utils.units import *
from watpy.utils.ioutils import *




PC_SI = 3.085677581491367e16  # m
MPC_SI = 1e6 * PC_SI


# def distance_scale():
#     def _distance_scale(data):
#         distance = data["params"]["rescale_to_radii"] * MPC_SI
#         amplitude_prefactor = (
#             (data["params"]["mass_starA"] + data["params"]["mass_starB"])
#             * MSun_meter()
#             / distance
#         )
#         data["signal"] = data["signal"] * amplitude_prefactor
#         return data

#     return _distance_scale

class distance_scale():
    def __init__(self):
        pass
    def __call__(self, data):
        distance = data["params"]["rescale_to_radii"] * MPC_SI
        amplitude_prefactor = (
            (data["params"]["mass_starA"] + data["params"]["mass_starB"])
            * MSun_meter()
            / distance
        )
        data["signal"] = data["signal"] * amplitude_prefactor
        return data