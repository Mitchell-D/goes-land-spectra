import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime,timedelta
from pprint import pprint

from GeosGeom import GeosGeom
from plotting import plot_geo_rgb,plot_geo_scalar
from helpers import load_welford_grids,finalize_welford,QueryResults

if __name__=="__main__":
    #proj_root = Path("/rhome/mdodson/goes-land-spectra")
    proj_root = Path("/Users/mtdodson/desktop/projects/goes-land-spectra")
    ## directory where domain arrays will be stored.
    geom_dir = proj_root.joinpath("data/domains")
    ## directory where pkls of listings will be stored.
    listing_dir = proj_root.joinpath("data/listings")
    out_dir = proj_root.joinpath("data/results")
    fig_dir = proj_root.joinpath("figures")

    name_fields = ["satellite","listing", "stime", "ftime",
            "domain", "month", "tod", "band"]

    lat_bounds = (30,40)
    lon_bounds = (-90,-75)

    ## result pkls are stratified by domain, listing, ToD and band
    include_only = {
        #"domain":["geom-goes-conus-1"], ## G19E
        "domain":["geom-goes-conus-0"], ## G19E
        "listing":["clearland-l1b-c0"],
        #"tod":["64800"], ## in seconds, only 18z
        #"tod":["54000","64800","75600"], ## in seconds, only 18z
        "tod":["0"], ## in seconds, only 18z
        "month":["01",],
        "band":["C13"]
        #"month":["11","12","01"],
        #"month":["07",],
        }

    plot_types = ["scalar", "rgb"]
    plot_metrics = ["min", "max", "m1","m2","m3","m4"]
    merge_over = ["tod", "month"]

    plot_reqs = {
        "rgb":{
            "force_subset":{
                "band":["C01", "C03", "C02"],
                },
            "px_scale":2, ## resolution factor wrt domain
            },
        "scalar":{
            "px_scale":1, ## resolution factor wrt domain
            },
        }

    qr = QueryResults(list(out_dir.iterdir()), name_fields)
    for ptype in plot_types:
        tmp_incl = {**include_only, **plot_reqs[ptype].get("force_subset",{})}
        qr = qr.subset(**tmp_incl)
        #mrg_keys,mrg_paths = zip(*qr.group(merge_over, invert=True).items())
        #pprint(dict(zip(mrg_keys,mrg_paths)))

        for mrg_keys,mrg_paths in qr.group(merge_over, invert=True).items():
            merged,latlon = load_welford_grids(
                    pkl_paths=mrg_paths,
                    geom_dir=geom_dir,
                    lat_bounds=lat_bounds,
                    lon_bounds=lon_bounds,
                    subgrid_rule="complete",
                    reduce_func=np.nanmean,
                    metrics=None, ## TODO: implement metric subset after merge
                    merge=True,
                    res_factor=4,
                    )
            merged = finalize_welford(merged)
            for k,v in merged.items():
                print(k, v.shape, latlon.shape)
                out_str = f"{ptype}_{'_'.join(mrg_keys)}_{k}_" + \
                        '-'.join(merge_over)
                out_path = fig_dir.joinpath(out_str)
                plot_geo_scalar(
                    data=merged[k],
                    latitude=latlon[...,0],
                    longitude=latlon[...,1],
                    plot_spec={
                        "cbar_orient":"horizontal",
                        },
                    fig_path=out_path,
                    show=False,
                    )

    ## sanity check valid counts in results dir
    '''
    for rp in out_dir.iterdir():
        sat,listing,dstr0,dstrf,geom,mstr,sstr,bstr = rp.stem.split("_")
        res,meta = pkl.load(rp.open("rb"))
        _,m_domain = pkl.load(geom_dir.joinpath(f"{geom}.pkl").open("rb"))
        print()
        print(rp.stem)
        #gg = GeosGeom(**gg)
        for rk in res.keys():
            ndom = np.count_nonzero(m_domain)
            assert res[rk].shape[0] % ndom == 0
            fac = res[rk].shape[0] // ndom
            print(f"{rk} invalid:",
                    np.count_nonzero(~np.isfinite(res[rk]))/(ndom*fac))
    '''
