import numpy as np
import pickle as pkl
import time
import gc
import h5py
import itertools
import json
from multiprocessing import Pool,Lock,current_process
from pprint import pprint
from pathlib import Path
from datetime import datetime,timedelta
from netCDF4 import Dataset
from numpy.lib.stride_tricks import as_strided

from goes_land_spectra.helpers import merge_welford,load_welford_grids
from goes_land_spectra.helpers import QueryResults
from goes_land_spectra.geos_geom import GeosGeom

proj_root = Path("/rhome/mdodson/goes-land-spectra")
## directory where downloaded files will be buffered.
download_dir = proj_root.joinpath("data/downloads")
## directory where domain arrays will be stored.
geom_dir = proj_root.joinpath("data/domains")
## directory where pkls of listings will be stored.
listing_dir = proj_root.joinpath("data/listings")
result_dir = proj_root.joinpath("data/results")
out_dir = proj_root.joinpath("data/cfds")
perm_path = proj_root.joinpath(
        "data/permutations/perm_geom-goes-conus-0_final.pkl")

## valid multiples of domain axis sizes for higher resolutions
valid_res_facs = [1,2,4]

## underscore-separated field labels for results pkls
all_fields = ["satellite","listing", "stime", "ftime",
        "domain", "month", "tod", "band"]
file_fields = ["satellite","listing","stime","ftime","domain"]
axis_fields = ["month", "tod", "band"]
stored_fields = ["px", "subpx", "metric"]

## fields that uniquely correspond to a file (ie not stored in the file)
include_only = {
    "satellite":["goes16"],
    "listing":["clearland-l1b-c0"],
    "stime":["20170701"],
    "ftime":["20240630"],
    "domain":["geom-goes-conus-0"],
    }

## dict providing
all_months = ["01","02","03","04","05","06","07","08","09","10","11","12"]
file_datasets = {
    "day-500m":[
        ("month", all_months),
        ("tod", ["43200", "54000", "64800", "75600", "0"]),
        ("band", ["C02"]),
        ],
    "day-1km":[
        ("month", all_months),
        ("tod", ["43200", "54000", "64800", "75600", "0"]),
        ("band", ["C01", "C03", "C05"]),
        ],
    "day-2km":[
        ("month", all_months),
        ("tod", ["43200", "54000", "64800", "75600", "0"]),
        ("band", ["C06", "C07", "C13", "C15"]),
        ],
    "night-2km":[
        ("month", all_months),
        ("tod", ["10800", "21600", "32400"]),
        ("band", ["C13", "C15"]),
        ],
    }

## metadata for h5 file creation, coordinate labeling
dataset_params = {
        "day-500m":{
            "coords":["px","subpx","month","tod","band","metric"],
            "res_fac":4,
            "lock_axes":["subpx", "month"],
            "chunk_config":[2048, 16, 12, 3, 3, 3],
            },
        "day-1km":{
            "coords":["px","subpx","month","tod","band","metric"],
            "res_fac":2,
            "lock_axes":["subpx", "month"],
            "chunk_config":[2048, 4, 12, 3, 3, 3],
            },
        "day-2km":{
            "coords":["px","subpx","month","tod","band","metric"],
            "res_fac":1,
            "lock_axes":["subpx", "month"],
            "chunk_config":[2048, 1, 12, 3, 3, 3],
            },
        "night-2km":{
            "coords":["px", "subpx", "month", "tod", "band", "metric"],
            "res_fac":1,
            "lock_axes":["subpx", "month"],
            "chunk_config":[2048, 1, 12, 3, 3, 3],
            }
        }


""" -- ( end normal configuration ) -- """


## ensure that the include_only request matches the format needed to only
## retrieve result pkls that go into a single file.
assert all(k in include_only.keys() and len(include_only[k])==1
        for k in file_fields)

out_fields = [include_only[k][0] for k in file_fields]

## determine the file domain and load the domain pkl for it
domain = out_fields[file_fields.index("domain")]
geom_path = geom_dir.joinpath(f"{domain}.pkl")
assert geom_path.exists()
ggdict = pkl.load(geom_path.open("rb"))

## domain is always defined in terms of the spatially smallest mask
geoms = {
    dk:(GeosGeom(**ggdict[dk][0]),ggdict[dk][1])
    for dk in ggdict.keys()
    }

domain_key = sorted(ggdict.keys(), key=lambda t:t[0]*t[1])[0]
ggargs_domain,m_dom_tmp = ggdict[domain_key]
gg_domain = GeosGeom(**ggargs_domain)
out_path = out_dir.joinpath(f"cfd_{'_'.join(out_fields)}.h5")

m_dom = {
    rf:np.repeat(np.repeat(m_dom_tmp, rf, axis=0), rf, axis=1)
    for rf in valid_res_facs
    }
ny,nx = m_dom[1].shape

if out_path.exists():
    print(f"Already exists: {out_path.as_posix()}")
    if input("overwrite? (Y): ") != "Y":
        exit(0)
    out_path.unlink()

## some domains have NaN coord values, so their masks must be corrected
m_nans = np.any(np.isnan(gg_domain.latlon), axis=-1)
m_valid_tmp = m_dom[1] & ~m_nans

## make dicts of valid masks at each resolution for easy use
m_valid = {
    rf:np.repeat(np.repeat(m_valid_tmp, rf, axis=0), rf, axis=1)
    for rf in valid_res_facs
    }
m_valid_1d = {rf:m_valid[rf][m_dom[rf]] for rf in valid_res_facs}
N = np.count_nonzero(m_valid_1d[1])

## load the permutation chosen for this hdf5
_,perm,_ = pkl.load(perm_path.open("rb"))
assert perm.shape == (N,2), f"{N=} ; {perm.shape=}"

## (p,2) all valid indeces at domain resolution
all_ixs = np.stack(np.where(m_valid[1]), axis=-1)
all_ixs_1d = np.where(m_valid[1].reshape(-1))

## construct pixel mappings for efficiently loading permuted arrays
ix_maps = {}
ix_map_labels = ["ix-fr", "ix-fry", "ix-frx", "ix-dom", "ix-subpx"]
for rf in valid_res_facs:
    ## (res_fac**2,2) array of index pertubations for mapping subpixel dim
    ptb_ixs = np.stack(np.meshgrid(
        np.arange(rf),
        np.arange(rf),
        indexing="ij"
        ), axis=-1).reshape(-1,2)
    ## (px,subpx,2) array of 2d hi-res indeces to 2d low-res pixels
    allix_2d = np.array([[
        (ixy * rf + ixpy, ixx * rf + ixpx)
        for ixpy,ixpx in ptb_ixs
        ] for ixy,ixx in all_ixs
        ])
    ## allix_1d and allix_2d mutually map m_valid pixel indeces in arrays
    ## of each resolution to (px,subpx) groups wrt the domain array
    allix_1d = allix_2d[...,0] * allix_2d.shape[1] + allix_2d[...,1]
    ixs_stored = allix_1d.reshape(-1)

    ## (domain_pixel,subpixel) indices
    allix_psp = np.stack(np.meshgrid(
        all_ixs_1d,
        np.arange(allix_1d.shape[1]),
        indexing="ij",
        ), axis=-1)

    assert allix_1d.size == N * rf**2
    ## (pixel, subpixel, maptype) shaped array, where the map type is
    ## (full-res 1d idx, full-res y idx, fulll-res x idx, domain-res 1d idx,
    ## domain subpixel idx]
    ix_maps[rf] = np.concatenate([
        allix_1d[...,None], allix_2d, allix_psp
        ], axis=-1)

    print()
    #ix_maps[rf][:,:,3] = perm[:,0,np.newaxis]
    print(ix_maps[rf][1202300,:,0])
    print(ix_maps[rf][1202300,:,1:3])
    print(ix_maps[rf][1202300,:,3:5])
    print(ix_maps[rf].shape)

"""
given mappings for rf and a data
"""

## make a mapping from all unique file axis combinations back to the dataset
## that contains them. This enforces that each axis field combo maps to only
## a single dataset.
combo_to_dataset = {}
for dsk,dsd in file_datasets.items():
    combos = itertools.product(*[dict(dsd)[k] for k in axis_fields])
    for c in combos:
        if c in combo_to_dataset.keys():
            raise ValueError(f"{c} must map to only 1 valid dataset")
        combo_to_dataset[c] = dsk

## group existing results pkls by their axis fields
qr = QueryResults([
    p for p in result_dir.iterdir()
    if "acquired" not in p.stem
    ], all_fields)
qr = qr.subset(**include_only)
gdict = qr.group(axis_fields)
assert all(len(v)==1 for k,v in gdict.items()), \
        "each band/tod/month combination must be unique!"

"""
- collapse higher-resolution band to new axis (length 4 or 16)
- keep different datasets "day-500m", "day-1km", "day-2km", "night-2km"


1.
"""
h5_datasets = {}
def _add_dataset(out_h5):
    pass

print(f"Opening {out_path.as_posix()}")
with h5py.File(out_path, "w") as out_h5:
    grp_valid = out_h5.create_group(f"mask") ## 2d boolean valid masks
    grp_ixmap = out_h5.create_group(f"ixmap") ## index mappings
    grp_sas = out_h5.create_group(f"sas") ## scan angles
    for rf in valid_res_facs:
        grp_valid.create_dataset(
            rf, shape=m_valid[rf].shape, dtype=bool, chunks=None)
        grp_valid[rf][...] = m_valid[rf]
        grp_ixmap.create_dataset(
            rf, shape=ix_maps[rf].shape, dtype=np.uint32, chunks=None)
        grp_ixmap[rf][...] = ix_maps[rf]
        sas = np.stack(geoms[rf].args[1], axis=-1)
        grp_sas.create_dataset(
            rf, shape=geoms[rf].shape, dtype=np.float32, chunks=None)
        grp_sas[rf][...] = geoms[rf]

    h5grp = out_h5.create_group("data")
    h5grp.create_dataset(
        "perm",
        shape=(N,2),
        dtype=np.uint32,
        chunks=None,
        )
    ## add geometry data
    gparams,(sa_ew,sa_ns) = gg_domain.args()

    ## axis keys and pkl file paths
    for ak,rpath in [(k,v[0]) for k,v in gdict.items()]:
        rpath.stem.split("_")
        dsk = combo_to_dataset[ak]
        if dsk not in h5_datasets.keys():
            continue

        print(f"{ak} => {dsk}")






