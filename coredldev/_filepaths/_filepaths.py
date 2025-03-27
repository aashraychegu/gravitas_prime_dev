import pathlib as p

resources = p.Path(__file__).parent.parent / "_resources"
CoRe_DB_path = resources / "CoRe_DB"
padded_spectrograms_path = resources / "padded_spectrograms"
pthfiles_path = resources / "pthfiles"
noduplication_path = pthfiles_path / "noduplication.pt"
allextrationradii_path = pthfiles_path / "allextrationradii.pt"
ligopsd_path = resources / "aLIGO_O4_high_asd.txt"
freqs_path = resources / "freqs.npy"
noisecache_path = resources / "noisecache"
all_paths = {
    "resources": resources,
    "CoRe_DB_path": CoRe_DB_path,
    "padded_spectrograms_path": padded_spectrograms_path,
    "pthfiles_path": pthfiles_path,
    "noduplication_path": noduplication_path,
    "allextrationradii_path": allextrationradii_path,
}
