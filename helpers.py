import numpy as np
import pickle as pkl
import time
from pprint import pprint
from pathlib import Path
from numpy.lib.stride_tricks import as_strided
from datetime import datetime,timedelta

from GeosGeom import GeosGeom

TEMP = {
    "C01":(3000,5000),
    "C02":(6000,10000),
    "C03":(3000,5000),
    "C05":(3000,5000),
    "C06":(1500,2500),
    "C07":(1500,2500),
    "C13":(1500,2500),
    "C15":(1500,2500),
    }

def load_welford_grids(pkl_paths:list, geom_dir:Path,
        lat_bounds=None, lon_bounds=None, subgrid_rule="complete",
        reduce_func=np.nanmean, metrics=None, merge=False, res_factor=1):
    """
    Load and re-grid GOES welford pkls

    :@param pkl_paths: list of pkl paths to open and optionally regrid
    :@param geom_dir: dir where geometry pkls matching the paths are found
    :@param metrics: list of metrics to extract. default to all of them.
    :@param merge: If True, statistics from all pkls will be merged together
    :@param regrid: If True,

    :@return: (out_array, m_domain, geos_geom, features) where out_array has
        shape (P,F,M) if regrid is False, (Y,X,F,M) otherwise. P are in-domain
        pixels, Y/X are lat/lon dimensions, F are the unique combinations
    """
    all_metrics = ["count", "min", "max", "m1", "m2", "m3", "m4"]
    if not isinstance(pkl_paths, (list,tuple)):
        pkl_paths = [Path(pkl_paths)]
    if metrics is None:
        metrics = all_metrics

    ## load the geographic domain for this
    tups = [p.stem.split("_") for p in pkl_paths]
    domains = list(zip(*tups))[4]
    assert all(d==domains[0] for d in domains[1:]),domains
    ggargs,m_domain = pkl.load(geom_dir.joinpath(
        f"{domains[0]}.pkl").open("rb"))
    gg = GeosGeom(**ggargs)

    ## find slices that describe the requested subgrid
    slc_lat,slc_lon = get_latlon_slice_bounds(
        lats=gg.lats,
        lons=gg.lons,
        lat_bounds=lat_bounds,
        lon_bounds=lon_bounds,
        subgrid_rule=subgrid_rule,
        oob_value=np.nan,
        )

    ## calculate the shape of the domain subgrid
    #domy = slc_lat.stop - slc_lat.start
    #domx = slc_lon.stop - slc_lon.start #m_domain.shape
    domy,domx = m_domain.shape
    m_sub = m_domain[slc_lat,slc_lon]

    out_shape = (
        m_sub.shape[0]*res_factor,
        m_sub.shape[1]*res_factor,
        [len(pkl_paths),1][merge]
        )
    dom_yixs,dom_xixs = np.where(m_domain)


    ## TODO: want to only regrid and operate on the subset defined by the
    ## slices. To do so, need to mask pixels

    '''
    new = {
        "count":prv["count"] + cur["count"],
        "min":np.full(out_shape, np.nan, dtype=np.float32),
        "max":np.full(out_shape, np.nan, dtype=np.float32),
        "m1":np.full(out_shape, np.nan, dtype=np.float32),
        "m2":np.full(out_shape, np.nan, dtype=np.float32),
        "m3":np.full(out_shape, np.nan, dtype=np.float32),
        "m4":np.full(out_shape, np.nan, dtype=np.float32),
        }
    '''
    new = None

    m_dom = {} ## domain masks per supported original factor
    res_merged = None
    tgt_fac_dom = res_factor ## target px factor wrt domain (2km)
    for j,p in enumerate(pkl_paths):
        ## determine how many times larger along both axes the data array is
        ## than the domain array
        res,meta = pkl.load(p.open("rb"))
        ## TODO: implement shape from results dict
        #cury,curx = res["count"].shape
        cury,curx = TEMP[p.stem.split("_")[-1]]

        ## convert the domain valid mask to the current array size
        cur_fac_dom = None
        if domy==cury and domx==curx:
            cur_fac_dom = 1
            #m_sub_tmp = np.copy(m_sub)
        else:
            yfac = cury // domy
            xfac = curx // domx
            print(cury, domy, curx, domx)
            assert cury % domy == 0
            assert curx % domx == 0
            assert yfac==xfac, "should always be true for GOES"
            cur_fac_dom = yfac
            ## adapt the subgrid mask to the target size
            #m_sub_tmp = np.repeat(m_sub, cur_fac_dom, axis=0)
            #m_sub_tmp = np.repeat(m_sub_tmp, cur_fac_dom, axis=1)

        if cur_fac_dom in dom_masks_per_fac.keys():
            m_dom[cur_fac_dom] = None

        ## determine the size factor of the current array wrt the target
        if cur_fac_dom == tgt_fac_dom:
            pass
        elif cur_fac_dom < tgt_fac_dom: ## expand current to target domain
            ## current resolution must strictly divide target resolution
            cur_fac_tgt = tgt_fac_dom // cur_fac_dom
            for k in all_metrics:
                res[k] = np.where(m_sub_tmp, res[k], np.nan)
                res[k] = np.repeat(res[k], cur_fac_tgt, axis=0)
                res[k] = np.repeat(res[k], cur_fac_tgt, axis=1)
            assert tgt_fac_dom % cur_fac_tgt == 0
        else: ## contract if the current is larger than the target domain
            ## target resolution must strictly divide current resolution
            cur_fac_tgt = cur_fac_dom // tgt_fac_dom
            assert cur_fac_tgt % tgt_fac_dom == 0
            for k in all_metrics:
                res[k] = np.where(m_sub_tmp, res[k], np.nan)
                res[k] = reduce_func(as_strided(
                    res[k],
                    shape=(out_shape[0],out_shape[1],cur_fac_tgt,cur_fac_tgt),
                    strides=(res[k].strides[0]*yfac, res[k].strides[1]*yfac,
                        res[k].strides[0], res[k].strides[1]),
                        ), axis=(2, 3))
        for k in all_metrics:
            print(res[k].shape)
        if merge:
            if res_merged is None:
                res_merged = res
            else:
                res_merged = merge_welford(res_merged, res)
        else:
            for k in all_metrics:
                new[k][...,j] = res[k]

def get_latlon_slice_bounds(lats, lons, lat_bounds=None, lon_bounds=None,
        subgrid_rule="complete", oob_value=np.nan):
    """ """
    ## set any oob values to nan for simplicity. This assumes there aren't
    ## sparse nans in the coord arrays while the oob value is different, but
    ## that assumption should hold in any reasonable circumstances.
    m_valid = (lats != oob_value) & (lons != oob_value)
    lats[~m_valid] = np.nan
    lons[~m_valid] = np.nan
    if not (lat_bounds is None and lon_bounds is None):
        if lats.shape != lons.shape:
            raise ValueError("lats and lons must have the same shape")
        if subgrid_rule not in ["strict", "complete"]:
            raise ValueError("subgrid_rule must be 'strict' or 'complete'")
        if lat_bounds is None and not lon_bounds is None:
            lon_bounds = (np.nanmin(lons), np.nanmax(lons))
        if lon_bounds is None and not lat_bounds is None:
            lat_bounds = (np.nanmin(lats), np.nanmax(lats))

        ## get a boolean mask of in-bounds
        lat_min, lat_max = lat_bounds
        lon_min, lon_max = lon_bounds
        lat_mask = (lats >= lat_min) & (lats <= lat_max) & m_valid
        ## if bounds cross dateline, use or condition
        if lon_max < lon_min:
            lon_mask = (lons >= lon_min) | (lons <= lon_max)
        else:
            lon_mask = (lons >= lon_min) & (lons <= lon_max)

        ## minimum bounding rectangle containing any in-bounds values
        m_sub = lat_mask & lon_mask & m_valid
        ixy_any = np.where(np.any(m_sub, axis=1))[0]
        ixx_any = np.where(np.any(m_sub, axis=0))[0]
        if len(ixy_any) == 0 or len(ixx_any) == 0:
            raise ValueError(
                f"No valid px in bounds: {lat_bounds} {lon_bounds}" + \
                f" coord lats: ({np.nanmin(lats)}, {np.nanmax(lats)})" + \
                f" coord lons: ({np.nanmin(lons)}, {np.nanmax(lons)})")
        if subgrid_rule=="complete":
            slc_lat = slice(ixy_any[0], ixy_any[-1] + 1)
            slc_lon = slice(ixx_any[0], ixx_any[-1] + 1)
        if subgrid_rule=="strict":
            raise ValueError(f"Not implemented. Have fun!!")
    else:
        slc_lat,slc_lon = slice(0,m_valid.shape[0]),slice(0,m_valid.shape[1])

    return slc_lat,slc_lon

def merge_welford(w1, w2):
    """
    given 2 dicts containing arbitrary but equally shaped arrays for keys
    "count", "m1", "m2", "m3", "m4", merges them using the Terribery
    adaptation of the welford algorithm for higher order moments
    """
    d1 = w1["m1"] - w2["m1"]
    d2 = d1 * d1
    d3 = d1 * d2
    d4 = d2 * d2
    cnew = w1["count"] + w2["count"]

    new = {}
    new["count"] = cnew
    new["m1"] = (w1["count"] * w1["m1"] + w2["count"] * w2["m1"]) / cnew

    new["m2"] = w1["m2"] + w2["m2"] + d2 * w1["count"] * w2["count"] / cnew

    new["m3"] = w1["m3"] + w2["m3"] \
        + d3*w1["count"]*w2["count"] * (w1["count"]-w2["count"]) / cnew**2 \
        + 3*d1 * (w1["count"] * w2["m2"] - w2["count"] * w1["m2"]) / cnew

    c1 = w1["count"]**2 - w1["count"] * w2["count"] + w2["count"]**2
    c2 = w1["m2"] * w2["count"]**2 + w2["m2"] * w1["count"]**2
    new["m4"] = w1["m4"] + w2["m4"] \
        + d4 * w1["count"] * w2["count"] * c1 / (cnew**3) \
        + 6 * d2 * c2 / cnew**2 \
        + 4 * d1 * (w1["count"] * w2["m3"] - w2["count"] * w1["m3"]) / cnew
    return new

def get_closest_latlon(self, lat, lon):
    """
    returns the index of the pixel closest to the provided lat/lon
    """
    masked_lats = np.nan_to_num(self._lats, 999999)
    masked_lons = np.nan_to_num(self._lons, 999999)

    # Get an array of angular distance to the desired lat/lon
    lat_diff = masked_lats-lat
    lon_diff = masked_lons-lon
    total_diff = np.sqrt(lat_diff**2+lon_diff**2)
    min_idx = tuple([ int(c[0]) for c in
        np.where(total_diff == np.amin(total_diff)) ])
    return min_idx
