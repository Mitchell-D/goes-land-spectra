import numpy as np
import pickle as pkl
import time
import imageio.v3 as iio
from pprint import pprint
from pathlib import Path
from numpy.lib.stride_tricks import as_strided
from datetime import datetime,timedelta

from goes_land_spectra.geos_geom import GeosGeom

def mp_gen_gif_from_group(args):
    return args,gen_gif_from_group(**args)
def gen_gif_from_group(group_key, group_paths, group_fields, name_fields,
        out_dir, duration=.1):
    """
    Make a gif Given a group of existing image files ordered like a single
    member of the dict returned by QueryResults.group.

    :@param group_key: tuple of strings identifying this file group
    :@param group_paths: List of Path objects for images in this group
    :@param name_fields: string names for underscore-separated fields in
        all file names
    :@param out_dir: directory where the subsequent gif will be generated
    """
    base_name = None
    for p in group_paths:
        ndict = dict(zip(name_fields, p.stem.split("_")))
        tmp_bn = "_".join([
            ndict[fk] for fk in name_fields if fk in group_fields
            ])
        if base_name is None:
            base_name = tmp_bn
        else:
            assert tmp_bn == base_name, (tmp_bn, base_name)
    out_path = out_dir.joinpath(f"{base_name}.gif")
    with iio.imopen(out_path, "w", plugin="pillow") as giff:
        for p in group_paths:
            giff.write(iio.imread(p), duration=.1)
    return out_path

def finalize_welford(welford:dict):
    """
    Given a set of welford arrays as stored in results pkls, calculate the
    stddev and higher moments by renormalizing them.
    """
    n = welford["count"]
    results = {}
    results["count"] = n
    if "m1" in welford.keys():
        results["mean"] = welford["m1"]
    if "m2" in welford.keys():
        variance = welford["m2"] / n
        results["stddev"] = np.sqrt(variance)
    if "m3" in welford.keys():
        results["skewness"] = (welford["m3"] / n) / (variance ** 1.5)
    if "m4" in welford.keys():
        results["kurtosis"] = (welford["m4"] / n) / (variance ** 2)
    if "min" in welford.keys():
        results["min"] = welford["min"]
    if "max" in welford.keys():
        results["max"] = welford["max"]
    return results

def load_welford_grids(pkl_paths:list, geom_dir:Path,
        lat_bounds=None, lon_bounds=None, subgrid_rule="complete",
        reduce_func="merge", metrics=None, merge=False, res_factor=1):
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

    '''
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
    '''

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
            if reduce_func!="merge":
                for k in all_metrics:
                    cur[k] = reduce_func(as_strided(
                        cur[k],
                        shape=(out_shape[0],out_shape[1],
                            cur_fac_tgt,cur_fac_tgt),
                        strides=(
                            cur[k].strides[0]*cur_fac_tgt,
                            cur[k].strides[1]*cur_fac_tgt,
                            cur[k].strides[0],
                            cur[k].strides[1]),
                            ), axis=(2, 3))
            else:
                merged = None
                for j,i in np.indices((cur_fac_tgt,cur_fac_tgt)).reshape(2,-1):
                    cur_subpx = {cur[k][j,i] for k in all_metrics}
                    merged = cur_subpx if merged is None \
                            else merge_welford(merged, cur_subpx)
                cur = merged

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
        new["m1"][mv_only_w2] = w2["m1"][mv_only_w2]
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
            if k not in w1.keys():
                new[k] = w2[k]
            elif k not in w2.keys():
                new[k] = w1[k]
            else:
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

class QueryResults:
    """
    search files given a common underscore-separated file structure.
    """
    def __init__(self, file_paths, name_fields):
        self._p = list(file_paths)
        self._f = list(name_fields)

    @property
    def paths(self):
        return self._p

    @property
    def tups(self):
        return list(map(lambda r:(r,r.stem.split("_")), self._p))

    def set_paths(self, file_paths):
        self._p = file_paths

    def add_paths(self, file_paths):
        self._p = list(set(*self._p,*file_paths))

    def subset(self, sub_dict=None, **kwargs):
        if not sub_dict is None:
            kwargs = {**sub_dict, **kwargs}
        for k,v in kwargs.items():
            assert k in self._f,f"{k} must be in {self._f}"
            if isinstance(v, str):
                kwargs[k] = [kwargs[k]]

        sub_paths = []
        ## for each file file...
        for p,pt in self.tups:
            pdct = dict(zip(self._f,pt))
            if all(pdct[k] in kwargs[k] for k in kwargs.keys()):
                sub_paths.append(p)

        if len(sub_paths)==0:
            return QueryResults([], self._f)

        return QueryResults(sub_paths, self._f)

    #def __repr__(self):
    #    return str(list(map(lambda p:p.as_posix(), self._p)))

    def group(self, group_fields:list, invert=False):
        """
        return pkls that share a combination of group_fields
        """
        groups = {}
        assert all(f in self._f for f in group_fields),group_fields
        if invert:
            group_fields = list(set(self._f)-set(group_fields))
        group_fields = sorted(group_fields, key=lambda k:self._f.index(k))
        gixs = [self._f.index(f) for f in group_fields]
        for p,t in self.tups:
            gkey = tuple(t[ix] for ix in gixs)
            if gkey not in groups.keys():
                groups[gkey] = []
            groups[gkey].append(p)
        return group_fields,groups

class HConfig:
    """
    Hierarchical configuration system mapping query dictionaries to a
    configuration dict that is the merger of all matching configurations.
    Priority is given to configurations with more specific query fields.
    """
    def __init__(self, config):
        """Initialize an empty configuration system."""
        self.configs = []
        for f,v in config:
            if isinstance(f, dict):
                f = f.items()
            self.add_config(f, v)

    def add_config(self, field, config_dict):
        """
        :@param field: List of 2-tuples representing key/value pairs
        :@param config_dict: Dict to return when this field matches
        """
        field = sorted(field)
        assert len(set(next(zip(*field))))==len(field)
        if len(self.configs) and field in next(zip(*self.configs)):
            raise ValueError(f"config field already loaded: {field}")
        self.configs.append((field, config_dict))

    def query(self, query_dict):
        """
        :@param query_dict: Dict of key/value pairs to match against
        :@return: dict of combined configuration from all matching fields
        """
        matches = []
        for field,config_dict in self.configs:
            if self._field_matches_query(field, query_dict):
                matches.append((field, config_dict))

        # Sort by number of tuples (ascending)
        # so larger fields override smaller ones
        matches = sorted(matches, key=lambda x:len(x[0]))

        pprint(query_dict)
        pprint(matches)
        print()
        # Combine configs with later (larger)fields
        # overriding earlier (smaller) ones
        result = {}
        for field,config_dict in matches:
            result.update(config_dict)

        return result

    def _field_matches_query(self, field, query_dict):
        """ """
        for key, value in field:
            if key not in query_dict:
                return False
            if isinstance(value, str):
                if query_dict[key] != value:
                    return False
            else:
                if query_dict[key] not in value:
                    return False
        return True
