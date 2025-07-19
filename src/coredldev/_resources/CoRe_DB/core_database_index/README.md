# CoRe_DB

### CoRe GW Database Main Index repository

This repository contains the metadata for all simulations available within the CoRe Gravitational Wave database.

To obtain the single simulation data please syncronize the corresponding repository.

Note that each repository contains all runs performed with the same model (but at different resolutions). This might result in sizeable downloads for models with several different resolutions available.

### Database structure

    + core_database/		        Gitlab Group
    |--- + BAM_0001/                Database entry for model BAM:0001 
    |    | metadata_main.txt        Metadata file for the database entry, common between all runs
    |    |--- + R01/                Run with specific resolution R01
    |    |    | metadata.txt        Metadata for this specific run
    |    |    | data.h5             HDF5 archive containing the waveform data
    |    |--- + R02/                Run with specific resolution R02 
    |    |    | metadata.txt        Metadata for this specific run
    |    |    | data.h5             HDF5 archive containing the waveform data
    |    | ...
    |--- + BAM_0002/                Database entry for model BAM:0002 
    |    | ...     
    | ...
    | ...
    |--- + THC_0001/                Database entry for model THC:0001
    |    | metadata_main.txt        Metadata file for the database entry, common between all runs
    |    |--- + R01/                Run with specific resolution R01
    |    |    | metadata.txt        Metadata for this specific run
    |    |    | data.h5             HDF5 archive containing the waveform data
    |    |--- + R02/                Run with specific resolution R02 
    |    |    | metadata.txt        Metadata for this specific run
    |    |    | data.h5             HDF5 archive containing the waveform data
    |    | ...
    | ...
    | ...
    |--- + core_database_index/     Contains the main metadata for each database entry (this project)
    |    | README.md            
    |    |--- + json/               Contains .json files for the index
    |    |    | NR.json             Metadata for the Numerical Relativity models
    |    |    | Hybrid.json         Metadata for the Hybrid waveforms between Numerical Relativity and Effective One Body    