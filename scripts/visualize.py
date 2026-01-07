import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime,timedelta
from pprint import pprint

from goes_land_spectra.geos_geom import GeosGeom
from goes_land_spectra.plotting import plot_geostationary
from goes_land_spectra.helpers import load_welford_grids,HConfig
from goes_land_spectra.helpers import finalize_welford,QueryResults

swbands = ["C01","C02","C03","C05"]
lwbands = ["C07","C13","C15"]
dcmap = "nipy_spectral" ## default color map
plot_spec_config = [
    [[("metric","count")],{"vmin":0,"vmax":200,"cmap":dcmap}],
    [[("metric",["min","max","mean","stddev","kurtosis"])],{"cmap":dcmap}],
    [[("metric","skewness")],{"cmap":"berlin"}],

    [[("band",swbands),("metric",["mean","min","max"])],{"vmin":0,"vmax":1}],
    [[("band",lwbands),("metric",["mean","min","max"])],
        {"vmin":230,"vmax":320}],

    [[("band", swbands), ("metric", "stddev")], {"vmin":0, "vmax":.1}],
    [[("band", lwbands), ("metric", "stddev")], {"vmin":0, "vmax":12}],

    [[("band", swbands), ("metric", "skewness")], {"vmin":-10, "vmax":10}],
    [[("band", lwbands), ("metric", "skewness")], {"vmin":-10, "vmax":10}],

    [[("band", swbands), ("metric", "kurtosis")], {"vmin":1, "vmax":120}],
    [[("band", lwbands), ("metric", "kurtosis")], {"vmin":1, "vmax":120}],
    ]

if __name__=="__main__":
    proj_root = Path("/rhome/mdodson/goes-land-spectra")
    #proj_root = Path("/Users/mtdodson/desktop/projects/goes-land-spectra")
    ## directory where domain arrays will be stored.
    geom_dir = proj_root.joinpath("data/domains")
    ## directory where pkls of listings will be stored.
    listing_dir = proj_root.joinpath("data/listings")
    out_dir = proj_root.joinpath("data/results")
    fig_dir = proj_root.joinpath("figures/scalar")

    name_fields = ["satellite","listing", "stime", "ftime",
            "domain", "month", "tod", "band"]

    #lat_bounds = (30,40)
    #lon_bounds = (-90,-75)
    lat_bounds,lon_bounds = None,None

    ## result pkls are stratified by domain, listing, ToD and band
    include_only = {
        #"domain":["geom-goes-conus-1"], ## G19E
        "domain":["geom-goes-conus-0"], ## G19E
        "listing":["clearland-l1b-c0"],
        #"tod":["64800"], ## in seconds, only 18z
        #"tod":["54000","64800","75600"], ## in seconds, only 18z
        #"tod":["0"], ## in seconds, only 18z
        #"month":["01",],
        #"band":["C13"]
        #"month":["11","12","01"],
        #"month":["07",],
        }

    plot_types = ["scalar", "rgb"]
    plot_metrics = ["min", "max", "m1","m2","m3","m4"]
    #merge_over = ["tod", "month"]
    merge_over = []

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

    for ptype in plot_types:
        qr = QueryResults(list(out_dir.iterdir()), name_fields)
        tmp_incl = {**include_only, **plot_reqs[ptype].get("force_subset",{})}
        qr = qr.subset(**tmp_incl)
        #mrg_keys,mrg_paths = zip(*qr.group(merge_over, invert=True).items())
        #pprint(dict(zip(mrg_keys,mrg_paths)))

        mfields,mdict = qr.group(merge_over, invert=True)
        for mkey,mrg_paths in mdict.items():
            ## make a dict describing this merged product so that the configuration
            ## can be efficiently queried
            product_fields = dict(zip(mfields,mkey))
            ## load the domain information for this group
            domain = list(map(
                lambda p:p.stem.split("_")[name_fields.index("domain")],
                mrg_paths
                ))
            assert all(d==domain[0] for d in domain[1:])
            domain = domain[0]
            ggdict = pkl.load(geom_dir.joinpath(f"{domain}.pkl").open("rb"))
            ## domain is always defined in terms of the spatially smallest mask
            dkey = sorted(ggdict.keys(), key=lambda t:t[0]*t[1])[0]
            ggargs_domain,m_domain = ggdict[dkey]
            gg_dom = GeosGeom(**ggargs_domain)
            extent = [
                np.amin(gg_dom.e_w_scan_angles) * \
                        gg_dom.perspective_point_height,
                np.amax(gg_dom.e_w_scan_angles) * \
                        gg_dom.perspective_point_height,
                np.amin(gg_dom.n_s_scan_angles) * \
                        gg_dom.perspective_point_height,
                np.amax(gg_dom.n_s_scan_angles) * \
                        gg_dom.perspective_point_height,
                ]

            merged,latlon = load_welford_grids(
                    pkl_paths=mrg_paths,
                    geom_dir=geom_dir,
                    lat_bounds=lat_bounds,
                    lon_bounds=lon_bounds,
                    subgrid_rule="complete",
                    reduce_func=np.nanmean,
                    metrics=None, ## TODO: implement metric subset after merge
                    merge=True,
                    res_factor=1,
                    )
            merged = finalize_welford(merged)

            for k,v in merged.items():
                product_fields["metric"] = k
                hc = HConfig(plot_spec_config)
                ## getting rid of nans from coordinates
                m_data_nans = np.isnan(merged[k])
                m_coord_nans = np.any(np.isnan(gg_dom.latlon), axis=-1)

                ## plot the data normally
                #'''
                out_str = f"{ptype}_{'_'.join(mkey)}_{k}"
                if len(merge_over):
                    out_str += f"_{'-'.join(merge_over)}"
                out_path = fig_dir.joinpath(out_str + ".png")
                print(out_path.stem, k,
                    f"{np.nanmin(merged[k]):.3f} {np.nanmax(merged[k]):.3f}")
                plot_geostationary(
                    data=np.where(m_coord_nans, np.nan, merged[k]),
                    sat_lon=gg_dom.longitude_of_projection_origin,
                    sat_height=gg_dom.perspective_point_height,
                    sat_sweep=gg_dom.sweep_angle_axis,
                    plot_spec={
                        "projection":{
                            "type":"geostationary",
                            },
                        "title":f"{' '.join(mkey)} {k}",
                        "extent":extent,
                        "cb_orient":"horizontal",
                        "gridlines_color":"black",
                        "dpi":160,
                        "interp":"none",
                        **hc.query(product_fields),
                        },
                    fig_path=out_path,
                    show=False,
                    debug=False,
                    )
                #'''

                ## plot nan values
                '''
                nanmap = np.where(m_coord_nans, 4, np.nan)
                nanmap = np.where(m_data_nans, 3, nanmap)
                nanmap = np.where(m_domain, nanmap, 0)

                out_str = f"{ptype}_NANS_{'_'.join(mkey)}_{k}" + \
                        '-'.join(merge_over)
                out_path = fig_dir.joinpath(out_str)
                plot_geostationary(
                    data=nanmap,
                    sat_lon=gg_dom.longitude_of_projection_origin,
                    sat_height=gg_dom.perspective_point_height,
                    sat_sweep=gg_dom.sweep_angle_axis,
                    plot_spec={
                        "projection":{
                            "type":"geostationary",
                            },
                        "title":f"{k} ({' '.join(mkey)})",
                        "extent":extent,
                        "cmap":"plasma",
                        "cb_orient":"horizontal",
                        "gridlines_color":"black",
                        "dpi":180,
                        "interp":"none",
                        },
                    fig_path=out_path,
                    show=False,
                    debug=False,
                    )
                '''

                ## plot the data using geo_scalar method
                '''
                plot_geo_scalar(
                    data=merged[k],
                    lat=latlon[...,0],
                    lon=latlon[...,1],
                    plot_spec={
                        "title":f""
                        "cbar_orient":"horizontal",
                        },
                    fig_path=out_path,
                    show=False,
                    )
                '''

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
