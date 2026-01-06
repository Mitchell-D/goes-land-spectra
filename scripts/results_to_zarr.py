import numpy as np
import pickle as pkl
import time
import gc
#import h5py
import itertools
import json
import zarr
from multiprocessing import Pool,Lock,current_process
from pprint import pprint
from pathlib import Path
from datetime import datetime,timedelta
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

## Specify, in order, the dimensional configuration and feature axis labels
## for each distinct dataset. Leave coordinate axis labels defined as None.
all_months = ["01","02","03","04","05","06","07","08","09","10","11","12"]
day_tods = ["43200", "54000", "64800", "75600", "0"]
night_tods = ["10800", "21600", "32400"]
metrics = ["count", "min", "max", "m1", "m2", "m3", "m4"]
datasets = {
    "day-500m":[
        ("px", None),
        ("subpx", None),
        ("month", all_months),
        ("tod", day_tods),
        ("band", ["C02"]),
        ("metric", metrics)
        ],
    "day-1km":[
        ("px", None),
        ("subpx", None),
        ("month", all_months),
        ("tod", day_tods),
        ("band", ["C01", "C03", "C05"]),
        ("metric", metrics)
        ],
    "day-2km":[
        ("px", None),
        ("subpx", None),
        ("month", all_months),
        ("tod", day_tods),
        ("band", ["C06", "C07", "C13", "C15"]),
        ("metric", metrics)
        ],
    "night-2km":[
        ("px", None),
        ("subpx", None),
        ("month", all_months),
        ("tod", night_tods),
        ("band", ["C13", "C15"]),
        ("metric", metrics)
        ],
    }

## metadata for zarr file creation, coordinate labeling
file_params = {
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

""" -- ( load the domain geometry for all resolutions ) -- """

## determine the file domain and load the domain pkl for it
domain = out_fields[file_fields.index("domain")]
geom_path = geom_dir.joinpath(f"{domain}.pkl")
assert geom_path.exists()
ggdict = pkl.load(geom_path.open("rb"))

## domain is always defined in terms of the spatially smallest mask
geoms = {dk:(GeosGeom(**ggdict[dk][0]), ggdict[dk][1]) for dk in ggdict.keys()}
domain_key = sorted(ggdict.keys(), key=lambda t:t[0]*t[1])[0]
ggargs_domain,m_dom_tmp = ggdict[domain_key]
gg_domain = GeosGeom(**ggargs_domain)

## determine the path to the zarr file being created.
out_path = out_dir.joinpath(f"cfd_{'_'.join(out_fields)}.zarr")
if out_path.exists():
    print(f"Already exists: {out_path.as_posix()}.")
    exit(0)

""" -- ( load the domain geometry for all resolutions ) -- """

m_dom = {rf:np.repeat(np.repeat(m_dom_tmp, rf, axis=0), rf, axis=1)
    for rf in valid_res_facs}

num_dom = {rf:np.count_nonzero(v) for rf,v in m_dom.items()}

## some domains have NaN coord values, so their masks must be corrected
m_nans = np.any(np.isnan(gg_domain.latlon), axis=-1)
m_valid_tmp = m_dom[1] & ~m_nans

## make dicts of valid masks at each resolution for easy use
m_valid = {
    rf:np.repeat(np.repeat(m_valid_tmp, rf, axis=0), rf, axis=1)
    for rf in valid_res_facs
    }
## 1d boolean arrays matching domain size, True where coords non NaN
m_valid_1d = {rf:m_valid[rf][m_dom[rf]] for rf in valid_res_facs}
N = np.count_nonzero(m_valid_1d[1])

""" -- ( load the spatial permutation the user selected ) -- """

## load the permutation chosen for this hdf5
_,perm,_ = pkl.load(perm_path.open("rb"))
assert perm.shape == (N,2), f"{N=} ; {perm.shape=}"

## (P,2) all valid indices at domain resolution
ixs_valid = np.stack(np.where(m_valid[1]), axis=-1)

""" -- ( calculate index mappings between resolutions ) -- """

## construct pixel mappings for efficiently loading permuted arrays
ix_maps = {}
ix_map_labels = ["ix-fry", "ix-frx", "ix-dom", "ix-subpx"]
for rf in valid_res_facs:
    ## (res_fac**2,2) array of index pertubations for mapping subpixel dim
    ptb_ixs = np.stack(np.meshgrid(
        np.arange(rf),
        np.arange(rf),
        indexing="ij"
        ), axis=-1).reshape(-1,2)

    ## calculate the 1d domain index mapping to each valid pixel
    ix_ref = np.full(m_valid[rf].shape, 4294967295)
    ix_ref[m_dom[rf]] = np.arange(num_dom[rf])
    #ix_stored = ix_ref[m_valid_1d[rf]]

    ## (px,subpx,3) array of 2d hi-res indeces to 2d low-res pixels that
    ## contain them. the final axis has features:
    ## (highres_ixy, highres_ixx, highres_ixmask)
    ## which provides the 2d coordinates of sub-pixels in the high-res
    ## valid domain for each of the domain-res pixels.
    allix_2d = np.array([[
        (
            ixy*rf+ixpy, ## high-res y indeces
            ixx*rf+ixpx, ## high-res x indices
            ix_ref[ixy*rf+ixpy, ixx*rf+ixpx], ## 1d ixs of high-res domain mask
            ) for j,(ixpy,ixpx) in enumerate(ptb_ixs) ## combos of subpx idxs
        ] for i,(ixy,ixx) in enumerate(ixs_valid) ## valid idxs at domain res
        ])

    ## append indeces for pixel/subpixel for convenience.
    ## subpixels iterate row-wise first then column-wise
    allix_psp = np.stack(np.meshgrid(
        np.arange(N),
        np.arange(rf**2),
        indexing="ij",
        ), axis=-1)

    ## (ixy, ixx, ixd, ixv, ixs)
    ## (hi-res y,x ixs, hi-res 1d domain ix, 1d valid ix, subpixel ix)
    ix_maps[rf] = np.concatenate([allix_2d, allix_psp], axis=-1)

""" -- ( make sure that selected pkls are valid wrt axis features ) -- """

## make a mapping from all unique file axis combinations back to the dataset
## that contains them. This enforces that each axis field combo maps to only
## a single dataset.
combo_to_dataset = {}
for dsk,dsd in datasets.items():
    assert next(zip(*dsd)) == tuple(file_params[dsk]["coords"]), \
        "All coords in file_params and datasets must be the same. " + \
        f" Currently for {dsk}:" + \
        f"\ndatasets: {next(zip(*dsd))}" + \
        f"\nfile_params:{file_params[dsk]['coords']}"
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
axdict = qr.group(axis_fields)
assert all(len(v)==1 for k,v in axdict.items()), \
        "each band/tod/month combination must be unique!"

""" -- ( create the zarr file ) -- """

## declare a zstd compressor for the zarr data
compressor = zarr.codecs.BloscCodec(
        cname="zstd", clevel=5, shuffle="bitshuffle")

print(f"Opening {out_path.as_posix()}")
zstor = zarr.storage.LocalStore(out_path)

## create zarr groups
grp_root = zarr.group(store=zstor, overwrite=True)
grp_valid = grp_root.create_group(f"mask") ## 2d boolean valid masks
grp_ixmaps = grp_root.create_group(f"ixmap") ## index mappings
grp_geom = grp_root.create_group(f"geom") ## scan angles
grp_data = grp_root.create_group(f"data") ## result data

## add all geometry metadata needed to construct a GeosGeom
grp_geom.attrs.update(geoms[domain_key][0].args()[0])
## add the feature data for all datasets
grp_data.attrs.update({"datasets":datasets})
## add the file parameters
grp_root.attrs.update({"file_params":file_params})

## add the permutation & inverse
print(perm.shape)
ds_perm = grp_root.create_array(
        "perm",
        shape=(int(N),2),
        dtype=np.uint32,
        #chunks=None,
        compressors=[compressor],
        )
ds_perm[...] = perm

## ixmaps feature shape guide:
## (ixy, ixx, ixd, ixv, ixs)
## (hi-res y, hi-res x ixs, hi-res 1d domain ix, 1d valid ix, subpixel ix)

## for each distinct spatial resolution...
for rf in valid_res_facs:
    ## add valid mask data ()
    print(f"{m_valid[rf].shape=}")
    rfk = str(rf)
    grp_valid.create_array(
        rfk,
        shape=m_valid[rf].shape,
        dtype=bool,
        #chunks=None,
        )
    grp_valid[rfk][...] = m_valid[rf]

    ## add index map data ()
    print(f"{ix_maps[rf].shape=}")
    grp_ixmaps.create_array(
        rfk,
        shape=ix_maps[rf].shape,
        dtype=np.uint32,
        #chunks=None,
        compressors=[compressor],
        )
    grp_ixmaps[rfk][...] = ix_maps[rf][perm[:,0]]

    ## convert the scan angles to the (pixel, subpixel) form
    #'''
    sas = np.stack([
        geoms[m_valid[rf].shape][0].args()[1]["sa_ns"],
        geoms[m_valid[rf].shape][0].args()[1]["sa_ew"],
        ], axis=-1)

    sas_psp = np.full((N,rf**2,2), np.nan, dtype=np.float32)
    ix_map_1d = ix_maps[rf].reshape(-1,ix_maps[rf].shape[-1])
    print(ix_map_1d.shape)
    print(ix_map_1d[:32])
    sas_psp[ix_map_1d[:,3],ix_map_1d[:,4]] = sas[ix_map_1d[:,0],ix_map_1d[:,1]]

    #'''
    '''
    sas = np.stack([
        geoms[m_valid[rf].shape][0].args()[1]["sa_ns"],
        geoms[m_valid[rf].shape][0].args()[1]["sa_ew"],
        ], axis=-1)
    sas[ix_maps[rf][...,0],ix_maps[rf][...,1]]
    '''


    print(f"{sas_psp.shape=}")
    grp_geom.create_array(
        rfk,
        shape=sas_psp.shape,
        dtype=np.float32,
        #chunks=None,
        compressors=compressor,
        )
    grp_geom[rfk][...] = sas_psp[perm[:,0]]

## axis keys and pkl file paths
for ak,rpath in [(k,v[0]) for k,v in axdict.items()]:
    rpath.stem.split("_")
    dsk = combo_to_dataset[ak]
    print(f"{ak} => {dsk}")

print(grp_root)
