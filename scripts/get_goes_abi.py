import numpy as np
import pickle as pkl
import time
import gc
from multiprocessing import Pool,Lock,current_process
from pprint import pprint
from pathlib import Path
from datetime import datetime,timedelta
from netCDF4 import Dataset
from numpy.lib.stride_tricks import as_strided

from goes_land_spectra.GOESProduct import GOESProduct as GP
from goes_land_spectra.helpers import merge_welford
from goes_land_spectra.acquire import init_mp_get_goes_l1b_and_masks
from goes_land_spectra.acquire import mp_list_goes_day,init_s3_session
from goes_land_spectra.acquire import mp_get_goes_l1b_and_masks

proj_root = Path("/rhome/mdodson/goes-land-spectra")
## directory where downloaded files will be buffered.
download_dir = proj_root.joinpath("data/downloads")
## directory where domain arrays will be stored.
geom_dir = proj_root.joinpath("data/domains")
## directory where pkls of listings will be stored.
listing_dir = proj_root.joinpath("data/listings")
out_dir = proj_root.joinpath("data/results")

"""
GOES-16 (East): 20170228 - 20250407
GOES-17 (West): 20180828 - 20230110
GOES-18 (West): 20220512 - present
GOES-19 (East): 20241010 - present
"""
## start and end day for data listing
#sday,eday,gver = datetime(2018,1,1),datetime(2022,12,31) ## test period

sday,eday,gver = datetime(2017,7,1),datetime(2024,6,30),16 ## 7y 16E
#sday,eday,gver = datetime(2019,1,1),datetime(2021,12,31),17 ## 3y 17W
#sday,eday,gver = datetime(2022,7,1),datetime(2025,6,30),18 ## 3y 18W
#sday,eday,gver = datetime(2024,10,15),datetime(2025,3,15),19 ## <1y 19C

## L1b radiance bands to acquire
#l1b_bands = [1,2,3,5,6,7,13,15]
l1b_bands = [13,15]

## UTC hours of the day to capture shortwave and longwave radiances
#swtimes = [timedelta(hours=t) for t in [12,15,18,21,0]]
#swtimes = [timedelta(hours=t) for t in [15,18,21]]
#lwtimes = [timedelta(hours=t) for t in [12,15,18,21,0,3,6,9]]
#lwtimes = swtimes ## for now, no night pixels. need night-specific masking
swtimes = []
swtimes,lwtimes = [],[timedelta(hours=t) for t in [3,6,9]]

## maximum error in closest file time before timestep is invalidated
dt_thresh_mins = 5
#nworkers,batch_size = 4,2

## CONCLUSION: at least on hpc, large batch sizes are optimal to prevent
## over-taxing the head node.
#nworkers,batch_size = 8,24 ##
nworkers,batch_size = 6,24 ##

## identifying name of this dataset for the listing pkl
listing_name = f"goes{gver}_clearland-l1b-c0" ## lmask&l1b combo 0

new_listing = True
debug = True
redownload = False
delete_after_use = False ## look into storing compressed in-domain arrays
overwrite_results = False

""" ---------------- ( end normal configuration ) ---------------- """

lkey = f"{listing_name}" + \
    f"_{sday.strftime('%Y%m%d')}" + \
    f"_{eday.strftime('%Y%m%d')}"
listing_path = listing_dir.joinpath(f"listing_{lkey}.pkl")

## build a listing if requested, or used a matching stored one
if new_listing or not listing_path.exists():
    if debug:
        print(f"Getting new listing {listing_path.name}")
    all_days = []
    cday = sday
    while cday <= eday:
        all_days.append(cday)
        cday = cday + timedelta(days=1)

    l1b_bands_hours = [
        (f"C{n:02}",t)
        for n in l1b_bands
        for t in [swtimes,lwtimes][n>7]
        ]
    acmc_bands_hours = [("", t) for t in lwtimes]
    lst_bands_hours = [("", t) for t in lwtimes]

    ## map product to list of 2-tuple combos of bands with times of the day
    ## to acquire that band (as a timedelta, may be fractional hours)
    products=[
        #(GP(str(gver), "ABI", "L2", "ACMC"), acmc_bands_hours),
        (GP(str(gver), "ABI", "L2", "LSTC"), lst_bands_hours),
        (GP(str(gver), "ABI", "L1b", "RadC"), l1b_bands_hours),
        ]
    args = [{
        "date":d,
        "products":products,
        "time_offset_threshold":timedelta(minutes=15),
        "debug":debug,
        } for d in all_days]

    listing = []
    ## multiprocess over all requested days
    with Pool(nworkers, initializer=init_s3_session) as pool:
        for a,dl in pool.imap_unordered(mp_list_goes_day, args):
            if not dl:
                print(f"No timestep found for {a['date']}")
                continue
            print(f"Valid timesteps on {a['date']}: {len(dl.keys())}")
            listing += [
                ((a["date"].strftime("%Y%m%d"),h,td.seconds),v["keys"])
                for (h,td),v in dl.items()
                ]
    pkl.dump((args,listing), listing_path.open("wb"))
else:
    if debug:
        print(f"Loading existing listing {listing_path.name}")
    _,listing = pkl.load(listing_path.open("rb"))

## if a list of files that have already been acquired is present and the
## user doesn't want to repeat the results, negate the associated timesteps
loaded = []
acq_path = out_dir.joinpath(f"acquired_{lkey}.pkl")
if acq_path.exists() and not overwrite_results:
    loaded = pkl.load(acq_path.open("rb"))
    print(f"ignoring {len(loaded)} already-loaded timesteps")
## subset to not loaded and sort so that month/tod combos appear together,
## which minimizes the number of merges that need to happen. This step
## is critical for the head/worker balance's efficiency.
listing = sorted(
        list(filter(lambda l:l[0] not in loaded, listing)),
        key=lambda l:(l[0][0][4:6],l[0][1]) ## time of day then month
        )

batches = len(listing) // batch_size + bool(len(listing) % batch_size)
args = [{
    "geom_dir":geom_dir,
    "download_dir":download_dir,
    "bucket":f"noaa-goes{gver}",
    "listing":listing[bix*batch_size:bix*batch_size+batch_size],
    "replace_files":redownload,
    "delete_files":delete_after_use,
    "debug":debug,
    "masks_to_apply":["m_land", "m_valid"],
    } for bix in range(batches)]

## use the pkl naming scheme to point to existing results if requested
results = {} ## rkey:(geom,month,tod,band)
if not overwrite_results:
    for p in out_dir.iterdir():
        if "acquire" in p.name:
            continue
        sat,lk,ts0,tsf,gstr,mstr,sstr,bstr = p.stem.split("_")
        rkey = (gstr,mstr,sstr,bstr)
        tmp_lkey = "_".join((sat,lk,ts0,tsf))
        ## skip listings that don't match this one
        if not tmp_lkey == lkey:
            continue
        results[rkey] = p

## download and extract
metadata = {} ## (geom,band)
with Pool(nworkers, initializer=init_mp_get_goes_l1b_and_masks,
        initargs=(Lock(),)) as pool:
    ## iterate over batches returned by each worker
    for arg,(tmp_res,meta) in pool.imap_unordered(
            mp_get_goes_l1b_and_masks, args):
        if debug:
            t0 = time.perf_counter()
            loadtimes = []
        for mkey in meta.keys():
            if mkey not in metadata.keys():
                metadata[mkey] = meta[mkey]

        ## for each of the results returned, merge the welford dict with
        ## that of previously loaded results. Unique results are identified
        ## by keys that specify the (geom, month, ToD, band) combo
        for rkey in tmp_res.keys():
            out_path = out_dir.joinpath("_".join([lkey,*rkey])+".pkl")
            cur = tmp_res[rkey]
            prv,new = None,None
            ## if rkey isn't present, then either an overwrite is
            ## requested, or no data has been loaded yet
            if rkey not in results.keys():
                results[rkey] = out_path
                new = cur
            ## otherwise load and update the pkl file
            else:
                if debug:
                    tl1 = time.perf_counter()
                prv,_ = pkl.load(out_path.open("rb"))
                if debug:
                    loadtimes.append(time.perf_counter()-tl1)
                    print(f"head load: {loadtimes[-1]} ({rkey})")
                ## shape of previous counts dict needs to match current.
                assert prv["count"].shape==cur["count"].shape, \
                    (prv["count"].shape, cur["count"].shape)

                new = merge_welford(prv, cur)

            if debug:
                dc0 = time.perf_counter()
                dt = dc0 - t0
                print(f"head calc: {dt-sum(loadtimes)}")
            pkl.dump((new,metadata), out_path.open("wb"))
            if debug:
                dt = time.perf_counter() - dc0
                print(f"head dump: {dt}")

            ## try to free some memory (please do it, garbage collector)
            prv,cur,new,mv_prv,mv_cur,mv_both,mv_only_cur,mv_only_prv = \
                    [None]*8
            gc.collect() ## please please
        loaded += arg["listing"]
        pkl.dump(loaded, acq_path.open("wb"))
