import numpy as np
from pprint import pprint
from pathlib import Path

def get_latlon_slice_bounds(lat, lon, lat_bounds=None, lon_bounds=None,
        subgrid_rule="complete", oob_value=np.nan):
    """ """
    ## set any oob values to nan for simplicity. This assumes there aren't
    ## sparse nans in the coord arrays while the oob value is different, but
    ## that assumption should hold in any reasonable circumstances.
    m_valid = (lat != oob_value) & (lon != oob_value)
    lat[~m_valid] = np.nan
    lon[~m_valid] = np.nan
    lats,lons = gg.lats,gg.lons
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
        if inclusion=="complete":
            slc_lat = slice(ixy_any[0], ixy_any[-1] + 1)
            slc_lon = slice(ixx_any[0], ixx_any[-1] + 1)
        if inclusion=="strict":
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
