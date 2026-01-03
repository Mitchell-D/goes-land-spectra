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
    #ggargs,m_domain = pkl.load(geom_dir.joinpath(
    #    f"{domains[0]}.pkl").open("rb"))
    ggdict = pkl.load(geom_dir.joinpath(f"{domains[0]}.pkl").open("rb"))
    ## domain is always defined in terms of the spatially smallest mask
    domain_key = sorted(ggdict.keys(), key=lambda t:t[0]*t[1])[0]
    ggargs_domain,m_domain = ggdict[domain_key]
    gg_domain = GeosGeom(**ggargs_domain)

    ## find slices that describe the requested subgrid
    slc_lat,slc_lon = get_latlon_slice_bounds(
        lats=gg_domain.lats,
        lons=gg_domain.lons,
        lat_bounds=lat_bounds,
        lon_bounds=lon_bounds,
        subgrid_rule=subgrid_rule,
        oob_value=np.nan,
        )

    ## calculate the shape of the domain subgrid
    domy,domx = m_domain.shape
    res_final = None
    tgt_fac_dom = res_factor ## target px factor wrt domain (2km)
    for j,p in enumerate(pkl_paths):
        ## determine how many times larger along both axes the data array is
        ## than the domain array
        res,meta = pkl.load(p.open("rb"))
        ## TODO: implement shape from results dict (done?)
        cury,curx = res["shape"]
        #cury,curx = TEMP[p.stem.split("_")[-1]]

        ## convert the domain valid mask to the current array size
        cur_fac_dom = None
        if domy==cury and domx==curx:
            cur_fac_dom = 1
            #m_sub_tmp = np.copy(m_sub)
        else:
            yfac = cury // domy
            xfac = curx // domx
            assert cury % domy == 0
            assert curx % domx == 0
            assert yfac==xfac, "should always be true for GOES"
            cur_fac_dom = yfac

        ## get 2d domain mask for this band size
        if cur_fac_dom != 1:
            tmpm = np.repeat(m_domain, cur_fac_dom, axis=0)
            tmpm = np.repeat(tmpm, cur_fac_dom, axis=1)
        else:
            tmpm = m_domain

        ## make a slice for the geographic bounds on the current band size
        tmp_slc = (slice(slc_lat.start*cur_fac_dom, slc_lat.stop*cur_fac_dom),
            slice(slc_lon.start*cur_fac_dom, slc_lon.stop*cur_fac_dom))

        ## make a 2d mask of the current size within the goegraphic bounds
        m_bounds = np.full(tmpm.shape, False)
        m_bounds[*tmp_slc] = True
        ## get in-bounds indices wrt the 1d stored array
        m_bounds_1d = m_bounds[tmpm]
        yixs,xixs = np.where(m_bounds & tmpm)
        ## convert full-array indeces to the size of the 2d subgrid
        yixs -= tmp_slc[0].start
        xixs -= tmp_slc[1].start
        ## determine the shape of the new 2d subgrid
        cur_shape_2d = (tmp_slc[0].stop-tmp_slc[0].start,
            tmp_slc[1].stop-tmp_slc[1].start)
        ## regrid included 1d data to the 2d subgrid
        cur = {}
        for k in all_metrics:
            cur[k] = np.full(cur_shape_2d, np.nan, dtype=np.float32)
            cur[k][yixs,xixs] = res[k][m_bounds_1d]

        ## determine the size factor of the current array wrt the target
        if cur_fac_dom == tgt_fac_dom:
            pass
        elif cur_fac_dom < tgt_fac_dom: ## expand current to target domain
            ## current resolution must strictly divide target resolution
            cur_fac_tgt = tgt_fac_dom // cur_fac_dom
            for k in all_metrics:
                cur[k] = np.repeat(cur[k], cur_fac_tgt, axis=0)
                cur[k] = np.repeat(cur[k], cur_fac_tgt, axis=1)
            assert tgt_fac_dom % cur_fac_tgt == 0
        else: ## contract if the current is larger than the target domain
            ## target resolution must strictly divide current resolution
            cur_fac_tgt = cur_fac_dom // tgt_fac_dom
            out_shape = (cur_shape_2d[0] // cur_fac_tgt,
                cur_shape_2d[1] // cur_fac_tgt)
            assert cur_fac_tgt % tgt_fac_dom == 0
            for k in all_metrics:
                cur[k] = reduce_func(as_strided(
                    cur[k],
                    shape=(out_shape[0],out_shape[1],cur_fac_tgt,cur_fac_tgt),
                    strides=(
                        cur[k].strides[0]*cur_fac_tgt,
                        cur[k].strides[1]*cur_fac_tgt,
                        cur[k].strides[0],
                        cur[k].strides[1]),
                        ), axis=(2, 3))
        if merge:
            if res_final is None:
                res_final = cur
            else:
                res_final = merge_welford(res_final, cur)
        else:
            if res_final is None:
                out_shape = (*cur["count"].shape, len(pkl_paths))
                res_final = {
                    "shape":out_shape,
                    "count":np.full(out_shape, np.nan, dtype=np.float32),
                    "min":np.full(out_shape, np.nan, dtype=np.float32),
                    "max":np.full(out_shape, np.nan, dtype=np.float32),
                    "m1":np.full(out_shape, np.nan, dtype=np.float32),
                    "m2":np.full(out_shape, np.nan, dtype=np.float32),
                    "m3":np.full(out_shape, np.nan, dtype=np.float32),
                    "m4":np.full(out_shape, np.nan, dtype=np.float32),
                    }
            for k in all_metrics:
                res_final[k][...,j] = res[k]

    tgt_full_res = (m_domain.shape[0]*tgt_fac_dom,
            m_domain.shape[1]*tgt_fac_dom)
    assert tgt_full_res in ggdict.keys(), (tgt_full_res, list(ggdict.keys()))
    gg_cur = GeosGeom(**ggdict[tgt_full_res][0])
    sub_slice = (slice(slc_lat.start*tgt_fac_dom,slc_lat.stop*tgt_fac_dom),
            slice(slc_lon.start*tgt_fac_dom,slc_lon.stop*tgt_fac_dom))
    latlon = (gg_cur.lats[sub_slice],gg_cur.lons[sub_slice])
    #print(latlon[0].shape, latlon[1].shape)
    return res_final, np.stack(latlon, axis=-1)

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
    valid_to_merge = ["count", "min", "max", "m1", "m2", "m3", "m4"]
    mv_w1 = w1["count"] > 0
    mv_w2 = w2["count"] > 0
    mv_both = mv_w1 & mv_w2
    mv_only_w1 = mv_w1 & ~mv_w2
    mv_only_w2 = mv_w2 & ~mv_w1

    #print(f"merging",mv_w1.shape,mv_w2.shape, (np.count_nonzero(mv_only_w1), np.count_nonzero(mv_only_w2), np.count_nonzero(mv_both)))

    new = {
        "count":np.full(w1["count"].shape, 0, dtype=np.float32),
        "min":np.full(w1["count"].shape, np.nan, dtype=np.float32),
        "min":np.full(w1["count"].shape, np.nan, dtype=np.float32),
        "max":np.full(w1["count"].shape, np.nan, dtype=np.float32),
        "m1":np.full(w1["count"].shape, np.nan, dtype=np.float32),
        "m2":np.full(w1["count"].shape, np.nan, dtype=np.float32),
        "m3":np.full(w1["count"].shape, np.nan, dtype=np.float32),
        "m4":np.full(w1["count"].shape, np.nan, dtype=np.float32),
        }

    if np.any(mv_only_w1):
        new["count"][mv_only_w1] = w1["count"][mv_only_w1]
        new["min"][mv_only_w1] = w1["min"][mv_only_w1]
        new["max"][mv_only_w1] = w1["max"][mv_only_w1]
        new["m1"][mv_only_w1] = w1["m1"][mv_only_w1]
        new["m2"][mv_only_w1] = w1["m2"][mv_only_w1]
        new["m3"][mv_only_w1] = w1["m3"][mv_only_w1]
        new["m4"][mv_only_w1] = w1["m4"][mv_only_w1]
    if np.any(mv_only_w2):
        new["count"][mv_only_w2] = w2["count"][mv_only_w2]
        new["min"][mv_only_w2] = w2["min"][mv_only_w2]
        new["max"][mv_only_w2] = w2["max"][mv_only_w2]
        new["m2"][mv_only_w2] = w2["m1"][mv_only_w2]
        new["m2"][mv_only_w2] = w2["m2"][mv_only_w2]
        new["m3"][mv_only_w2] = w2["m3"][mv_only_w2]
        new["m4"][mv_only_w2] = w2["m4"][mv_only_w2]

    if np.any(mv_both):
        b1,b2 = {},{} ## masked dict subsets for values in both
        to_merge = [
                k for k in set(w1.keys()).union(set(w2.keys()))
                if k in valid_to_merge
                ]
        no_merge = [
                k for k in set(w1.keys()).union(set(w2.keys()))
                if k not in valid_to_merge
                ]
        for k in to_merge:
            b1[k] = w1[k][mv_both]
            b2[k] = w2[k][mv_both]
        d1 = b1["m1"] - b2["m1"]
        d2 = d1 * d1
        d3 = d1 * d2
        d4 = d2 * d2
        cn = b1["count"] + b2["count"]

        bth = {}
        bth["count"] = cn
        bth["m1"] = (b1["count"] * b1["m1"] + b2["count"] * b2["m1"]) / cn

        bth["m2"] = b1["m2"] + b2["m2"] + d2 * b1["count"] * b2["count"] / cn

        bth["m3"] = b1["m3"] + b2["m3"] \
            + d3*b1["count"]*b2["count"]*(b1["count"]-b2["count"]) / cn**2 \
            + 3*d1 * (b1["count"] * b2["m2"] - b2["count"] * b1["m2"]) / cn

        c1 = b1["count"]**2 - b1["count"] * b2["count"] + b2["count"]**2
        c2 = b1["m2"] * b2["count"]**2 + b2["m2"] * b1["count"]**2
        bth["m4"] = b1["m4"] + b2["m4"] \
            + d4 * b1["count"] * b2["count"] * c1 / (cn**3) \
            + 6 * d2 * c2 / cn**2 \
            + 4 * d1 * (b1["count"] * b2["m3"] - b2["count"] * b1["m3"]) / cn

        if "max" in b1.keys() and "max" in b2.keys():
            bth["max"] = np.where(
                    b1["max"] > b2["max"],
                    b1["max"], b2["max"])
        if "min" in b1.keys() and "min" in b2.keys():
            bth["min"] = np.where(
                    b1["min"] < b2["min"],
                    b1["min"], b2["min"])
        for k in b1.keys():
            new[k][mv_both] = bth[k]
        for k in no_merge:
            assert w1[k]==w2[k], f"Mismatching {k}:({w1[k]}, {w2[k]})"
            new[k] = w1[k]
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
