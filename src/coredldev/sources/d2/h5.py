from ..._filepaths._filepaths import CoRe_DB_path
import h5py as h5
from watpy.coredb.coredb import *
import numpy as np
import pathlib as p
import math
from ...utilites._preprocessing import *
from ..._resources.eos_to_index_map import eos_to_index_map


class h5Source:
    def __init__(
        self,
        path=CoRe_DB_path,
        selection_attributes=["id_eos", "id_mass_starA", "id_mass_starB"],
        device="cpu",
        sync=False,
        eos_to_index_map=eos_to_index_map,
    ) -> None:
        self.eos_to_index_map = eos_to_index_map
        self.path = path
        self.selection_attributes = selection_attributes
        self.device = device
        with HiddenPrints():
            if not self.path.exists():
                self.path.mkdir(exist_ok=False)
            cdb = CoRe_db(self.path)
            if sync:
                cdb.sync(verbose=False, lfs=True)
            self.sims = cdb.sim
        with HiddenPrints():
            if not self.path.exists():
                self.path.mkdir(exist_ok=False)
            cdb = CoRe_db(self.path)
            if sync:
                cdb.sync(verbose=False, lfs=True)
            self.sims = cdb.sim

    def __getitem__(self, index: int):
        return self.preprocess(*self.retrieve(index))

    def retrieve(self, psl):
        data = self.sims[psl[0]].run[psl[1]]
        h5path = p.Path(data.data.path) / "data.h5"
        metadata = {i: data.md.data[i] for i in self.selection_attributes}
        data = h5.File(h5path, "r")[psl[2]][psl[3]]  # type: ignore
        pm_time = data[:, -1]  # type: ignore
        data = cut_at_lowest_envelope(data[:, 1], data[:, 2])  # type: ignore
        sam_p = (pm_time[-1] - pm_time[0]) / len(pm_time)  # type: ignore
        return data, metadata, sam_p, psl[4], psl[5]

    def preprocess(self, ts, params, sam_p, percent_shift, snr):
        lts = len(ts)
        print(lts)
        if lts == 0:
            rts = np.zeros(90)
            rts[45] = ts[0]
            ts = rts
        elif lts < pad_to_for_planck_window:
            ts = np.concatenate(
                (
                    np.zeros(math.floor(pad_to_for_planck_window - lts)),
                    ts,
                    np.zeros(math.ceil(pad_to_for_planck_window - lts)),
                ),
                axis=0,
            )
            print("contingency")
        outlist = [
            self.eos_to_index_map[params["id_eos"]],
            float(params["id_mass_starA"]),
            float(params["id_mass_starB"]),
            float(power(ts)),
            float(percent_shift),
            float(snr),
        ]
        win = planck_window(math.floor(math.log(len(ts) / 2) * 6), len(ts))

        return (
            pad_width(wt(win, sam_p), percent_shift=percent_shift),
            outlist,
        )
