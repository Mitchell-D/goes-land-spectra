import numpy as np
import boto3
import pickle as pkl
import time
import gc
from botocore import UNSIGNED
from botocore.client import Config
from multiprocessing import Pool,Lock
from pprint import pprint
from pathlib import Path
from datetime import datetime,timedelta
from netCDF4 import Dataset
from numpy.lib.stride_tricks import as_strided

from GeosGeom import GeosGeom
from GOESProduct import GOESProduct as GP
from GOESProduct import valid_goes_products
from helpers import merge_welford

def init_mp_get_goes_l1b_and_masks():
    ## semaphor for the geom pkl index, which is captured on read and write
    global geom_index_lock
    geom_index_lock = Lock()
    ## need s3 session as well for downloading files
    init_s3_session()

def init_s3_session():
    global s3
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))

def load_geos_geom_parallel(geom_pkl_path):
    """ """
    assert geom_pkl_path.exists()
    with geom_index_lock:
        ggargs,m_domain = pkl.load(geom_pkl_path.open("rb"))
        gg = GeosGeom(**ggargs)
    return gg,m_domain

def dump_geos_geom_parallel(geom_dir, cur_geom, nc_path, domain_mask=None):
    """
    Multiprocess friendly function that maintains a directory of GeosGeom pkl
    files. The directory has an index file mapping basic projection values
    to a pkl of radian arrays and geometry data associated with that satellite
    position.

    :@param geom_dir: Directory where geometry pkls and index will be placed.
    :@param cur_geom: projection dictionary for the current netCDF
    :@param nc_path: path to the netCDF associated with cur_geom in case scan
        angle arrays need to be extracted for a new geom.
    :@param domain_mask: Optional mask setting True to valid points in the
        domain, which enables throwing away pixels that are always OOB.
    """
    match_fields = (
        "perspective_point_height",
        "longitude_of_projection_origin",
        "semi_major_axis",
        "semi_minor_axis",
        )
    assert geom_dir.exists()
    index_path = geom_dir.joinpath("index.pkl")

    geom_index_lock.acquire() ## acquire the semaphor
    ## load the index file of geometries
    if index_path.exists():
        geoms = pkl.load(index_path.open("rb"))
    else:
        geoms = {}
    ## construct the key associated with this projection
    cur = tuple(float(cur_geom[k]) for k in match_fields)
    geom_pkl_path = None
    ## see if there is a pkl indexed that matches this projection
    for k,v in list(geoms.items())[::-1]:
        if np.all(np.isclose(cur,k)):
            geom_pkl_path = Path(v)

    ## if there is a pkl for this geom configuration, return its path
    if not geom_pkl_path is None:
        if geom_pkl_path.exists():
            geom_index_lock.release()
            return geom_pkl_path
    ## make a new listing entry for the pkl otherwise
    else:
        geom_pkl_path = geom_dir.joinpath(
            f"geom-goes-conus-{len(geoms.keys())}.pkl")
        geoms = {cur:geom_pkl_path.as_posix(), **geoms}
        pkl.dump(geoms, index_path.open("wb"))

    ds = Dataset(nc_path, "r")
    sa_ns,sa_ew = np.meshgrid(ds["y"][...], ds["x"][...], indexing="ij")
    ds.close()
    pkl.dump(({
        "n_s_scan_angles":sa_ns,
        "e_w_scan_angles":sa_ew,
        "sweep_angle_axis":"x",
        **{f:cur_geom[f] for f in match_fields},
        }, domain_mask), geom_pkl_path.open("wb"))
    geom_index_lock.release()
    return geom_pkl_path

def parse_goes_stime(fname:str):
    return datetime.strptime(fname.split("_")[3], "s%Y%j%H%M%S%f")

def acquire_goes_files(bucket:str, keys:list, download_dir:Path,
        replace=False, debug=False):
    """ """
    out_paths = [download_dir.joinpath(k.split("/")[-1]) for k in keys]
    if replace:
        todl = zip(out_paths,keys)
    else:
        todl = list(filter(lambda pk:not pk[0].exists(), zip(out_paths,keys)))
    for p,k in todl:
        s3.download_file(bucket, k, p)
    return out_paths

def mp_get_goes_l1b_and_masks(args):
    return args,get_goes_l1b_and_masks(**args)

def get_goes_l1b_and_masks(geom_dir:Path, bucket:str, listing:list,
        download_dir:Path, replace_files=False, delete_files=False,
        masks_to_apply=[], debug=False):
    """
    TODO: add bucket to listing so multiple platforms' data can be acquired
        at the same time
    """
    if debug:
        t0 = time.perf_counter()
    rad_results = {}
    metadata = {}
    px_counter = 0
    ## listing has one set of files for each unique time step
    for (dstr,_,sstr),keys in listing:
        paths = acquire_goes_files(
            bucket=bucket,
            keys=keys,
            download_dir=download_dir,
            replace=replace_files,
            )
        lmask_path = None
        rad_paths = [] ## list of 2-tuples (channel, path)
        for p in paths:
            ptype = p.stem.split("_")[1]
            if "LSTC" in ptype:
                lmask_path = p
            else:
                assert ptype.split("-")[:3] == ["ABI","L1b","RadC"]
                rad_paths.append((ptype[-3:], p))
        assert not lmask_path is None
        assert len(rad_paths) > 0
        rad_paths = sorted(rad_paths) ## sort by channel string
        mlabels,masks,proj = load_valid_mask(lmask_path)
        if not masks_to_apply:
            m_valid = np.full(masks[0].shape, True)
        else:
            m_valid = np.all(np.stack([
                masks[i] if not l.split(" ")[0]=="not" else ~masks[i]
                for i,l in enumerate(mlabels)
                if l.split(" ")[-1] in masks_to_apply
                ], axis=0), axis=0)

        ## atomically ensure the existence of and load the current satellite
        ## geometry and previous domain mask
        geom_path = dump_geos_geom_parallel(
                geom_dir=geom_dir,
                cur_geom=proj,
                nc_path=lmask_path,
                domain_mask=masks[mlabels.index("m_land")],
                )
        gg,m_domain = load_geos_geom_parallel(geom_path)
        domain_size = np.count_nonzero(m_domain)

        gkey = geom_path.stem
        if gkey in rad_results.keys():
            assert sstr not in rad_results[gkey].keys()
            rad_results[gkey][sstr] = {}
        else:
            rad_results[gkey] = {sstr:{}}

        #m_domain_valid = m_domain[m_valid]
        bands,radiances,valid_masks,meta = zip(*[
            (b,*get_abi_l1b_radiance(p, get_mask=True))
            for b,p in rad_paths
            ])

        for j,b in enumerate(bands):
            mkey = (geom_path.stem,b)
            if mkey not in metadata.keys():
                metadata[mkey] = meta[j]

        ## on the first pass, make a mask where all radiance bands are valid
        m_rad = None
        domy,domx = m_domain.shape
        res_facs = []
        for j,(band,m_cur) in enumerate(zip(bands,valid_masks)):
            cury,curx = m_cur.shape
            if domy==cury and domx==curx:
                m_tmp = m_cur
                res_facs.append(1)
            else:
                yfac = cury // domy
                xfac = curx // domx
                assert cury % domy == 0
                assert curx % domx == 0
                assert yfac==xfac, "should always be true for GOES"
                res_facs.append(yfac)
                m_tmp = np.all(as_strided(
                    m_cur,
                    shape=(domy,domx,yfac,xfac),
                    strides=(
                        m_cur.strides[0]*yfac, m_cur.strides[1]*yfac,
                        m_cur.strides[0], m_cur.strides[1]
                        ),
                    ), axis=(2,3))
            m_rad = m_tmp&m_valid if m_rad is None else m_rad&m_tmp

        rad_results = {
            (geom_path.stem, dstr[4:6], str(sstr), bk):None
            for bk in bands
            }

        #m_dom_cur = m_rad[m_domain]
        for band,rad,fac in zip(bands,radiances,res_facs):
            rkey = (geom_path.stem, dstr[4:6], str(sstr), band)
            print(f"handling {dstr} {rkey}")
            if rad_results[rkey] is None:
                tmp_size = domain_size * fac**2
                rad_results[rkey] = {
                    ## float64 since it must be used as a denominator
                    "shape":rad.shape,
                    "count":np.full(tmp_size, 0, dtype=np.float64),
                    "min":np.full(tmp_size, np.nan, dtype=np.float32),
                    "max":np.full(tmp_size, np.nan, dtype=np.float32),
                    "m1":np.full(tmp_size, np.nan, dtype=np.float32),
                    "m2":np.full(tmp_size, np.nan, dtype=np.float32),
                    "m3":np.full(tmp_size, np.nan, dtype=np.float32),
                    "m4":np.full(tmp_size, np.nan, dtype=np.float32),
                    }

            ## apply the domain mask, scaling it up if this array is larger
            if fac==1:
                m_dom_tmp = m_domain
                m_rad_tmp = m_rad
            else:
                m_dom_tmp = np.repeat(m_domain, fac, axis=0)
                m_dom_tmp = np.repeat(m_dom_tmp, fac, axis=1)
                m_rad_tmp = np.repeat(m_rad, fac, axis=0)
                m_rad_tmp = np.repeat(m_rad_tmp, fac, axis=1)

            m_sub = m_rad_tmp[m_dom_tmp] ## valid pixels in the domain
            sub_rad = rad[m_dom_tmp][m_sub] ## radiance in the domain

            ## initialize first valid pixels
            m_first = m_sub & (rad_results[rkey]["count"] == 0)
            if np.any(m_first):
                rad_results[rkey]["min"][m_first] = sub_rad[m_first[m_sub]]
                rad_results[rkey]["max"][m_first] = sub_rad[m_first[m_sub]]
                rad_results[rkey]["m1"][m_first] = sub_rad[m_first[m_sub]]
                rad_results[rkey]["m2"][m_first] = 0.
                rad_results[rkey]["m3"][m_first] = 0.
                rad_results[rkey]["m4"][m_first] = 0.

            ## substitute minimum and maximum
            rad_results[rkey]["max"][m_sub] = np.nanmin(
                    [rad_results[rkey]["max"][m_sub], sub_rad], axis=0)
            rad_results[rkey]["min"][m_sub] = np.nanmin(
                    [rad_results[rkey]["min"][m_sub], sub_rad], axis=0)

            ## increment the count
            count_prev = rad_results[rkey]["count"][m_sub]
            count_now = count_prev + 1
            rad_results[rkey]["count"][m_sub] = count_now

            ## get references to current moments needed for calculation
            m1 = rad_results[rkey]["m1"][m_sub]
            m2 = rad_results[rkey]["m2"][m_sub]
            m3 = rad_results[rkey]["m3"][m_sub]

            ## calculate coefficients based on current mean and counts
            d = sub_rad - m1
            d_n = d / count_now
            d_n2 = d_n * d_n
            c1 = d * d_n * count_prev

            ## update the moments using the terriberry extension of the formula
            ## (original source discontinued, but i found it on wikipedia)
            rad_results[rkey]["m1"][m_sub] += d_n
            rad_results[rkey]["m4"][m_sub] += c1 * d_n2 * (
                    count_now**2 - 3*count_now + 3
                    ) + 6 * d_n2 * m2 - 4 * d_n * m3
            rad_results[rkey]["m3"][m_sub] += c1 * d_n \
                    * (count_now - 2) - 3 * d_n * m2
            rad_results[rkey]["m2"][m_sub] += c1

            if debug:
                px_counter += np.count_nonzero(m_sub)
        if delete_files:
            for p in paths:
                p.unlink()
    if debug:
        dt = time.perf_counter() - t0
        print(f"worker px/sec: {px_counter/dt}")
    return rad_results,metadata ## assumes metadata consistent

def get_abi_l1b_radiance(nc_file:Path, get_mask:bool=False,
        convert_tb_ref=True, _ds=None):
    """
    Extract the radiances from a L1b netCDf, optionally including a boolean
    mask for off-disc values.

    :@param nc_file: NOAA-style ABI L1b netCDF file
    :@param get_mask: if True, returns a (M,N) shaped boolean array along
        with the radiances.

    :@return: (M,N) shaped array of scalar radiances. If get_mask is True,
        returns 2-tuple like (radiances, mask) such that 'mask' is a (M,N)
        shaped boolean array which is True for off-limb values.
    """
    req_meta = ["kappa0","planck_fk1","planck_fk2","planck_bc1","planck_bc2"]
    ds = _ds if _ds else Dataset(nc_file.as_posix())
    rad = ds["Rad"][...].data
    if get_mask:
        m_valid = ds["DQF"][...]==0
    else:
        m_valid = np.all(rad.shape, True)
    mkeys = [k for k in req_meta if k in ds.variables.keys()]
    metadata = {k:tmpd for k in mkeys if not (tmpd:=ds[k][:].data)==0}
    ds.close()
    if convert_tb_ref:
        if "kappa0" in metadata.keys():
            rad *= metadata["kappa0"]
        elif "planck_fk1" in metadata.keys():
            planck = (
                metadata["planck_fk1"][:].data,
                metadata["planck_fk2"][:].data,
                metadata["planck_bc1"][:].data,
                metadata["planck_bc2"][:].data,
                )
            rad = rad_to_Tb(rad, *planck)
    return rad,m_valid,metadata

def rad_to_Tb(rads:np.array, fk1:np.array, fk2:np.array,
              bc1:np.array, bc2:np.array):
    """
    Use combined-constant coefficients of the planck function to convert
    radiance to brightness temperature.
    """
    return (fk2/np.log(fk1/rads+1) - bc1) / bc2


def load_clear_mask(cmask_path:Path):
    """
    Load GOES L2 ACMC cloud mask data and return a mask of clear pixels, a
    mask of valid pixels, and a dictionary of projection information.
    """
    ds = Dataset(cmask_path, "r")
    proj = ds.variables["goes_imager_projection"]
    proj = {
        "proj_name": proj.grid_mapping_name,
        "perspective_point_height": proj.perspective_point_height,
        "semi_major_axis": proj.semi_major_axis,
        "semi_minor_axis": proj.semi_minor_axis,
        "latitude_of_projection_origin": proj.longitude_of_projection_origin,
        "longitude_of_projection_origin": proj.longitude_of_projection_origin,
        "sweep_angle_axis": proj.sweep_angle_axis
        }
    cmask = ds["BCM"]
    m_valid = np.ma.getmask(cmask)
    m_valid &= ds["DQF"][...].data == 0
    cmask = ds["BCM"][...].data == 0
    ds.close()
    return cmask,m_valid,proj

def load_valid_mask(lst_nc_path:Path):
    """
    Load GOES L2 LSTC cloud and land mask data and return an array of masks
    where the quality flags indicate a

    DQF mask rules (from ncdump):

    | bit place 0 | mask key | desc |
    | -- | ------- | -------------------------- |
    |  2 | m_valid | valid input data provided  |
    |  4 | m_clear | clear sky conditions       |
    |  8 | m_vza   | valid local zenith angle   |
    | 16 | m_land  | valid land or inland water |
    | 32 | m_value | valid LST data range       |
    """
    ds = Dataset(lst_nc_path, "r")
    proj = ds.variables["goes_imager_projection"]
    proj = {
        "proj_name": proj.grid_mapping_name,
        "perspective_point_height": proj.perspective_point_height,
        "semi_major_axis": proj.semi_major_axis,
        "semi_minor_axis": proj.semi_minor_axis,
        "latitude_of_projection_origin": proj.longitude_of_projection_origin,
        "longitude_of_projection_origin": proj.longitude_of_projection_origin,
        "sweep_angle_axis": proj.sweep_angle_axis
        }
    dqf = ds["DQF"][...]
    mlabels = ["m_valid", "m_clear", "m_vza", "m_land", "m_value"]
    masks = [(dqf&b)==0 for b in [2,4,8,16,32]]
    ds.close()
    return mlabels,masks,proj

def mp_list_goes_day(args):
    return args,list_goes_day(**args)

def list_goes_day(date, products, time_offset_threshold:timedelta,
        debug=False):
    """
    Download GOES files for a single day from the AWS bucket. Specify the
    products to acquire as a 2-tuple (goes_product, bands_tods) such that
    goes_product is a valid GOESProduct object, and bands_tods is a list of
    2-tuples (substring, utc_tod). Each bands_tods member describes a single
    file to search for, defined by a combination of a file name substring and
    a UTC time of day (defined as a timedelta) associated with that day.
    """
    ## create a dict mapping distinct times to a subdict of required files
    ## for that hour, and empty lists for indicating availability.
    downloads = {}
    for prod,bands_tods in products:
        for band,t in bands_tods:
            ht = (t.seconds//3600, t)
            if ht not in downloads.keys():
                downloads[ht] = {
                    "required":[], "keys":[], "available":None,
                    }
            pb = (str(prod),band) ## unique product/bands for this hour/time
            downloads[ht]["required"].append(pb)

    ## make a bool array for each distinct timestep tracking whether there is a
    ## valid download available for that timestep.
    for ht,v in downloads.items():
        downloads[ht]["available"] = np.full(len(v["required"]), False)

    ## Query the aws bucket to identify keys for the products that are actually
    ## available closest to the requested times.
    for prod,bands_tods in products:
        assert prod in valid_goes_products, prod
        hours_bands_times = {}
        ## make a dict hour -> band -> times to fit the bucket structure
        for b,t in bands_tods:
            h = t.seconds//3600
            if h not in hours_bands_times.keys():
                hours_bands_times[h] = {b:[]}
            if b not in hours_bands_times[h].keys():
                hours_bands_times[h][b] = [t]
            else:
                hours_bands_times[h][b].append(t)

        ## NOAA S3 buckets are organized like:
        ## noaa-goes{N}/{prod}/{YYYY}/{jjj}/{hh}/{file_key}
        ## where the file_key identifies available bands and capture times as
        ## substrings. Set up the nesting structure accordingly.
        bucket = f"noaa-goes{prod.satellite}"
        dstr = date.strftime("%Y/%j")
        for h in hours_bands_times.keys():
            ## list objects for this product/date/hour combination
            prefix = f"{prod.sensor}-{prod.level}-{prod.scan}/{dstr}/{h:02}"
            response = s3.list_objects_v2(
                    Bucket=bucket,
                    Prefix=prefix
                    ).get("Contents", [])
            ## only keep responses that match one of the substrings
            bands_results = [
                (b,r)
                for r in response
                for b in hours_bands_times[h].keys()
                if b in r["Key"].replace(prefix, "")
                ]
            if not bands_results:
                continue
            ## bands within a product are identified by a matching substring.
            avail_bands = set(next(zip(*bands_results)))
            for band in avail_bands:
                times,keys = zip(*[
                    (parse_goes_stime(r["Key"]).timestamp(),r["Key"])
                    for b,r in filter(lambda br:br[0]==band, bands_results)
                    ])
                times = np.array(times)
                ## add files to the download list that are closest to requested
                ## times, as long as they are within the configured threshold.
                for t in hours_bands_times[h][band]:
                    dt = np.abs(times-(date+t).timestamp())
                    fix = np.argmin(dt) ## index of closest file in time
                    ht = (h,t)
                    if dt[fix] < time_offset_threshold.seconds:
                        ## index of required file in tracking array
                        pb = (str(prod),band)
                        rix = downloads[ht]["required"].index(pb)
                        downloads[ht]["keys"].append(keys[fix])
                        downloads[ht]["available"][rix] = True

    ## identify and remove any timesteps that don't have all required files.
    unav = []
    for ht,fd in downloads.items():
        if not np.all(fd["available"]):
            uprod = [
                "/".join(fd["required"][ix])
                for ix in np.where(~fd["available"])[0]
                ]
            if debug:
                print(f"skipping: {(date,ht[1].seconds//3600)}. " + \
                        f"unav: {', '.join(uprod)}")
            unav.append(ht)
    for ht in unav:
        del downloads[ht]
    return downloads

if __name__=="__main__":
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
    l1b_bands = [1,2,3,5,6,7,13,15]

    ## UTC hours of the day to capture shortwave and longwave radiances
    swtimes = [timedelta(hours=t) for t in [12,15,18,21,0]]
    #lwtimes = [timedelta(hours=t) for t in [12,15,18,21,0,3,6,9]]
    lwtimes = swtimes ## for now, no night pixels. need night-specific masking

    ## maximum error in closest file time before timestep is invalidated
    dt_thresh_mins = 5
    #nworkers,batch_size = 4,2

    ## CONCLUSION: at least on hpc, large batch sizes are optimal to prevent
    ## over-taxing the head node.

    #nworkers,batch_size = 24,2 ## ~25min on meteor, head node very taxed

    ## on meteor, head update:.4, head dump:.2, worker ~2e6 px/sec inc. DL
    #nworkers,batch_size = 6,8 ## ~10min

    ## on meteor, head calc:~5.5sec sometimes, worker ~2e6 px/sec inc. DL
    ## very backed up head queue at the end
    #nworkers,batch_size = 12,4 ## ~11.5min

    ## on meteor, head calc:~5.5s sometimes, worker ~3e6 px/sec NOT inc. DL
    #nworkers,batch_size = 12,16 ## ~4min

    ## on meteor, head calc:~6.5s sometimes, worker ~5e6 px/sec NOT inc. DL
    nworkers,batch_size = 8,24 ##

    ## identifying name of this dataset for the listing pkl
    listing_name = f"goes{gver}_clearland-l1b-c0" ## lmask&l1b combo 0

    new_listing = False
    debug = True
    redownload = False
    delete_after_use = True ## look into storing compressed in-domain arrays
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
    listing = list(filter(lambda l:l[0] not in loaded, listing))

    batches = len(listing) // batch_size + bool(len(listing) % batch_size)
    args = [{
        "geom_dir":geom_dir,
        "download_dir":download_dir,
        "bucket":f"noaa-goes{gver}",
        "listing":listing[bix*batch_size:bix*batch_size+batch_size],
        "replace_files":redownload,
        "delete_files":delete_after_use,
        "debug":debug,
        "masks_to_apply":["m_valid", "m_clear", "m_vza", "m_land"],
        } for bix in range(batches)]

    ## use the pkl naming scheme to point to existing results if requested
    results = {} ## rkey:(geom,month,tod,band)
    if not overwrite_results:
        for p in out_dir.iterdir():
            sat,lk,ts0,tsf,gstr,mstr,sstr,bstr = p.stem.split("_")
            rkey = (gstr,mstr,sstr,bstr)
            tmp_lkey = "_".join((sat,lk,ts0,tsf))
            ## skip listings that don't match this one
            if not tmp_lkey == lkey:
                continue
            results[rkey] = p

    ## download and extract
    metadata = {} ## (geom,band)
    with Pool(nworkers, initializer=init_mp_get_goes_l1b_and_masks) as pool:
        for arg,(tmp_res,meta) in pool.imap_unordered(
                mp_get_goes_l1b_and_masks, args):
            if debug:
                t0 = time.perf_counter()
                loadtimes = []
            for mkey in meta.keys():
                if mkey not in metadata.keys():
                    metadata[mkey] = meta[mkey]
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
                        print(f"head load: {loadtimes[-1]}")
                    cur = tmp_res[rkey]
                    assert prv["count"].shape==cur["count"].shape

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
