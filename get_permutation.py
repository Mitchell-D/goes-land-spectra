import numpy as np
import pickle as pkl
import time
from pprint import pprint
from pathlib import Path
from datetime import datetime,timedelta
from multiprocessing import Pool

from geos_geom import load_geos_geom
from permutation import permutations_from_configs
from plotting import plot_multiy_lines,plot_rgb_geos_on_plate
from plotting import plot_geostationary

'''
config_labels = ["target_avg_dist", "roll_threshold", "threshold_diminish",
        "recycle_count", "seed", "dynamic_roll_threshold"]
config_labels_conv = ["dist_threshold", "reperm_cap", "shuffle_frac", "seed"]
'''
perm_configs_global = [
        (2, .5, .01, 10, 20007221750, False),
        (2, .5, .01, 50, 20000221750, False),
        (2, .4, .01, 3, 20000721750, False),

        (3, .5, .01, 10, 20007221750, False),
        (3, .5, .01, 50, 20000721750, False),
        (3, .4, .01, 3, 20000721750, False),

        (2, .5, .01, 10, 2007221750, False),
        (2, .4, .01, 50, 2000072750, False),
        (2, .3, .01, 3, 2000721750, False),

        (4, .5, .01, 10, 200722150, False),
        (4, .4, .01, 50, 200721750, False),
        (4, .4, .01, 3, 7221750, False),

        (6, .5, .01, 10, 20072150, False),
        (6, .3, .01, 50, 20021750, False),
        (6, .3, .01, 3, 2000225, False),
        ]

perm_configs_conv = [
        (.5, 4, 1., 57927859614),
        (.5, 8, 1., 57927859614),
        (.5, 16, 1., 57927859614),
        (.5, 32, 1., 57927859614),
        (.5, 4, 1., 57927859614),
        (.5, 8, 1., 57927859614),
        (.5, 16, 1., 57927859614),
        (.5, 32, 1., 57927859614),
        (.5, 4, 1., 57927859614),
        (.5, 16, 1., 57927859614),
        (.5, 32, 1., 57927859614),
        (.5, 4, 1., 57927859614),
        (.5, 16, 1., 57927859614),
        (.5, 32, 1., 57927859614),

        (1, 4, 1., 57927859614),
        (1, 8, 1., 57927859614),
        (1, 16, 1., 57927859614),
        (1, 32, 1., 57927859614),
        (1, 4, 1., 57927859614),
        (1, 8, 1., 57927859614),
        (1, 16, 1., 57927859614),
        (1, 32, 1., 57927859614),
        (1, 4, 1., 57927859614),
        (1, 16, 1., 57927859614),
        (1, 32, 1., 57927859614),
        (1, 4, 1., 57927859614),
        (1, 16, 1., 57927859614),
        (1, 32, 1., 57927859614),

        (1.5, 4, 1., 57927859614),
        (1.5, 8, 1., 57927859614),
        (1.5, 16, 1., 57927859614),
        (1.5, 32, 1., 57927859614),
        (1.5, 4, 1., 57927859614),
        (1.5, 8, 1., 57927859614),
        (1.5, 16, 1., 57927859614),
        (1.5, 32, 1., 57927859614),
        (1.5, 4, 1., 57927859614),
        (1.5, 16, 1., 57927859614),
        (1.5, 32, 1., 57927859614),
        (1.5, 4, 1., 57927859614),
        (1.5, 16, 1., 57927859614),
        (1.5, 32, 1., 57927859614),

        (.5, 4, 1., 57927859614),
        (.5, 8, 1., 57927859614),
        (.5, 16, 1., 57927859614),
        (.5, 32, 1., 57927859614),
        (.5, 4, 1., 57927859614),
        (.5, 8, 1., 57927859614),
        (.5, 16, 1., 57927859614),
        (.5, 32, 1., 57927859614),
        (.5, 4, 1., 57927859614),
        (.5, 16, 1., 57927859614),
        (.5, 32, 1., 57927859614),
        (.5, 4, 1., 57927859614),
        (.5, 16, 1., 57927859614),
        (.5, 32, 1., 57927859614),

        (1, 4, .1, 57927859614),
        (1, 8, .1, 57927859614),
        (1, 16, .1, 57927859614),
        (1, 32, .1, 57927859614),
        (1, 4, .1, 57927859614),
        (1, 8, .1, 57927859614),
        (1, 16, .1, 57927859614),
        (1, 32, .1, 57927859614),
        (1, 4, .25, 57927859614),
        (1, 16, .25, 57927859614),
        (1, 32, .25, 57927859614),
        (1, 4, .5, 57927859614),
        (1, 16, .5, 57927859614),
        (1, 32, .5, 57927859614),

        (1.5, 4, .1, 57927859614),
        (1.5, 8, .1, 57927859614),
        (1.5, 16, .1, 57927859614),
        (1.5, 32, .1, 57927859614),
        (1.5, 4, .1, 57927859614),
        (1.5, 8, .1, 57927859614),
        (1.5, 16, .1, 57927859614),
        (1.5, 32, .1, 57927859614),
        (1.5, 4, .25, 57927859614),
        (1.5, 16, .25, 57927859614),
        (1.5, 32, .25, 57927859614),
        (1.5, 4, .5, 57927859614),
        (1.5, 16, .5, 57927859614),
        (1.5, 32, .5, 57927859614),
        ]

## "dist_threshold", "jump_cap", "shuffle_frac", "seed", "leaf_size"
perm_configs_fast = [
    #(.5, 3, .5, 202601041832, 32), ## basic config

    ## variations in both directions along each hyperparameter (except seed)
    (1., 3, .5, 202601041832, 32),
    (.25, 3, .5, 202601041832, 32),
    (.5, 6, .5, 202601041832, 32),
    (.5, 2, .5, 202601041832, 32),
    (.5, 3, .1, 202601041832, 32),
    (.5, 3, 1., 202601041832, 32),
    (.5, 3, .5, 202601041832, 64),
    (.5, 3, .5, 202601041832, 16),

    ## Much larger distance horizons
    (3, 3, .5, 202601041832, 32), ## basic config

    (5., 3, .5, 202601041832, 32),
    (1., 3, .5, 202601041832, 32),
    (3., 6, .5, 202601041832, 32),
    (3., 2, .5, 202601041832, 32),
    (3., 3, .1, 202601041832, 32),
    (3., 3, 1., 202601041832, 32),
    (3., 3, .5, 202601041832, 64),
    (3., 3, .5, 202601041832, 16),
    ]

def plot_perm_pkl(
        perm_pkl:Path, coords:np.array, subgrids:dict, geom,
        valid_mask:np.array, fig_dir:Path, show=False, chunk_size=64,
        plot_stats=False, plot_sparse_chunks=True, num_sparse_chunks=50,
        seed=None, debug=False):
    """
    :@param perm_pkl:
    :@param coords:
    :@param subgrids: dict mapping labels to ((y0,yf),(x0,xf)) coord bounds
    :@param fig_dir:
    :@param valid_mask: 2d boolean array with len(coords.shape[0]) True values
        corresponding to the locations of valid points
    :@param chunk_size: Hypothetical 1d chunk size used efficiency estimates
    :@param plot_stats:
    :@param seed: seed for choosing
    :@param debug:
    """
    args,perm,stats = pkl.load(perm_pkl.open("rb"))

    ## verify that the permutation includes all points
    assert perm[:,0].size==coords[:,0].size, \
            f"Perm size mismatch: {perm[:,0].size = }  {coords[:,0].size = }"
    check_valid = np.full(coords.shape[0],False)
    check_valid[perm[:,0]] = True
    assert np.all(check_valid), f"{perm_pkl.name} not a valid permutation!"

    print(f"\n{perm_pkl.stem}")
    print({k:v for k,v in args.items()
           if k not in ["coords", "seed", "batches_per_stat"]})
    print(f"dist avg: {stats[-1][0]:.2f} stddev: {stats[-1][1]:.2f}")


    ## (N,2) latlon array after permutation
    ll_perm = coords[perm[...,0]]
    ## (N,) latlon distances of permuted points from original location
    dists = np.sum((coords-ll_perm)**2, axis=1)**(1/2)
    ## (N,2) indeces of valid points in unpermuted space
    valid_idxs = np.argwhere(valid_mask)

    px_per_chunk = []
    for rlabel,((lat0,latf),(lon0,lonf)) in subgrids.items():
        ## 1d boolean mask of permuted points within the subgrid
        in_subset = (ll_perm[:,0] >= lat0) & (ll_perm[:,0] <= latf) & \
                (ll_perm[:,1] >= lon0) & (ll_perm[:,1] <= lonf)
        ## Number of unique contiguous chunks associated with the subset
        unq_chunks = np.unique(np.argwhere(in_subset)[:,0] // chunk_size)
        #if debug:
        #    print(f"{rlabel} {unq_chunks.size = }")
        px_per_chunk.append(np.count_nonzero(in_subset) / unq_chunks.size)

        ## (N,) indeces of subset points in permuted space
        subset_idxs = np.argwhere(in_subset).T

        ## Color unpermuted valid points white on a (lat,lon,3) rgb
        rgb = np.full(valid_mask.shape, 0)
        rgb[valid_idxs[:,0], valid_idxs[:,1]] = 255
        rgb = np.stack([rgb for i in range(3)], axis=-1)
        ## color unpermuted subset pixels red and permuted blue
        sub_ixs_noperm = valid_idxs[perm[:,0]][in_subset]
        sub_ixs_perm = valid_idxs[in_subset]
        rgb[sub_ixs_noperm[:,0],sub_ixs_noperm[:,1]] = np.array([0,0,255])
        rgb[sub_ixs_perm[:,0],sub_ixs_perm[:,1]] = np.array([255,0,0])

        ## must convert to meters wrt NADIR
        extent = [
            np.amin(geom.e_w_scan_angles)*gg.perspective_point_height,
            np.amax(geom.e_w_scan_angles)*gg.perspective_point_height,
            np.amin(geom.n_s_scan_angles)*gg.perspective_point_height,
            np.amax(geom.n_s_scan_angles)*gg.perspective_point_height,
            ]
        #'''
        plot_geostationary(
            data=rgb,
            sat_lon=gg.longitude_of_projection_origin,
            sat_height=gg.perspective_point_height,
            sat_sweep=gg.sweep_angle_axis,
            plot_spec={
                "projection":{
                    "type":"geostationary",
                    },
                "title":f"{perm_pkl.stem} ({rlabel})",
                "extent":extent,
                },
            fig_path=fig_dir.joinpath(f"{perm_pkl.stem}_{rlabel}.png"),
            show=show,
            debug=debug,
            )
        #'''

    if plot_sparse_chunks:
        rng = np.random.default_rng(seed)
        chunk_starts = np.arange(ll_perm.shape[0] // chunk_size) * chunk_size
        rng.shuffle(chunk_starts)
        ext_chunks = chunk_starts[:num_sparse_chunks]
        in_subset = np.full(ll_perm.shape[0], False)
        for cs in ext_chunks:
            in_subset[cs:cs+chunk_size] = True
        rgb = np.full(valid_mask.shape, 0)
        rgb[valid_idxs[:,0], valid_idxs[:,1]] = 255
        rgb = np.stack([rgb for i in range(3)], axis=-1)
        sub_ixs_noperm = valid_idxs[perm[:,0]][in_subset]
        sub_ixs_perm = valid_idxs[in_subset]
        rgb[sub_ixs_noperm[:,0],sub_ixs_noperm[:,1]] = np.array([255,0,0])
        rgb[sub_ixs_perm[:,0],sub_ixs_perm[:,1]] = np.array([0,0,255])

        ## must convert to meters wrt NADIR
        extent = [
            np.amin(geom.e_w_scan_angles)*gg.perspective_point_height,
            np.amax(geom.e_w_scan_angles)*gg.perspective_point_height,
            np.amin(geom.n_s_scan_angles)*gg.perspective_point_height,
            np.amax(geom.n_s_scan_angles)*gg.perspective_point_height,
            ]

        #'''
        plot_geostationary(
            data=rgb,
            sat_lon=gg.longitude_of_projection_origin,
            sat_height=gg.perspective_point_height,
            sat_sweep=gg.sweep_angle_axis,
            plot_spec={
                "projection":{
                    "type":"geostationary",
                    },
                "title":f"{perm_pkl.stem} ({rlabel})",
                "extent":extent,
                },
            fig_path=fig_dir.joinpath(f"{perm_pkl.stem}_chunked.png"),
            show=show,
            debug=debug,
            )

    mean_px_per_chunk = np.average(px_per_chunk)
    print(f"{perm_pkl.stem}, {mean_px_per_chunk = :.4f}")

    if plot_stats:
        plot_multiy_lines(
                data=list(zip(*stats)),
                xaxis=np.arange(len(stats)),
                plot_spec={
                    "x_label":"Iterations",
                    "y_labels":["Avg. Dist", "Stdev. Dist"],
                    "y_ranges":[(0,4),(0,4)],
                    "title":f"{perm_pkl.stem} {mean_px_per_chunk=:.3f}",
                    "spine_increment":.1,
                    },
                show=show,
                fig_path=fig_dir.joinpath(f"{perm_pkl.stem}_stats.png")
                )

if __name__=="__main__":
    #proj_root = Path("/rhome/mdodson/goes-land-spectra/")
    proj_root = Path("/Users/mtdodson/desktop/projects/goes-land-spectra")
    geom_dir = proj_root.joinpath("data/domains/")
    geom_index = pkl.load(geom_dir.joinpath("index.pkl").open("rb"))
    pkl_dir = proj_root.joinpath("data/permutations")
    fig_dir = proj_root.joinpath("figures/permutations")

    plot_geom_nans = False
    get_new_perms = False
    plot_perms = True
    show = False
    debug = False

    ## -- ( configuration for getting new permutations ) --
    workers = 10
    enum_start = 0
    #max_iterations = 4 ## not relevant when configured
    #pool_size = 1024*32 ## not relevant in fast mode
    kdt_workers = 4 ## only relevant for fast mode
    method = "fast"
    permute_geoms = [
        "geom-goes-conus-0",
        #"geom-goes-conus-1",
        #"geom-goes-conus-2",
        ]

    ## -- ( configuration for plotting permutations ) --
    substrs = [
        "geom-goes-conus-0",
        #"geom-goes-conus-1",
        #"geom-goes-conus-2",
        ]
    test_subgrids = {
        "seus":((33.88, 36.86), (-88.1, -83.6)),
        "michigan":((41.59, 45.94), (-88.28, -81.9)),
        "colorado":((36.99, 40.96), (-109.1, -102.05)),
        "etx":((28.97, 33.39), (-98.67, -93.28)),
        "ne":((43.0, 47.23), (-79.34, -67.99)),
        "nw":((42.00, 48.96), (-122.74, -118.24)),
        "cplains":((39.32, 41.69), (-96.88, -92.78)),
        "hplains":((43.73, 48.33), (-104.01, -96.08)),
        }
    chunk_size = 1024

    ## treat different geometric views independently regardless of task
    for view_key,vdict in geom_index.items():
        geom_path = geom_dir.joinpath(Path(vdict["path"]).name)
        if not geom_path.stem in permute_geoms:
            continue
        data_shapes = vdict["shapes"]
        domain_shape = sorted(data_shapes, key=lambda t:t[0]*t[1])[0]
        gg,m_domain = load_geos_geom(geom_path, shape=domain_shape)

        ## m_valid is True where
        m_nans = np.any(np.isnan(gg.latlon), axis=-1)
        m_valid = m_domain & ~m_nans
        m_valid_1d = m_valid[m_domain]

        latlon = np.stack([
            gg.lats[m_valid],
            gg.lons[m_valid],
            ], axis=1)


        ## sanity check plot nans in geometry matrix
        if plot_geom_nans:
            print(np.unique(m_domain[~m_nans]))
            from plotting import plot_geo_tiles,plot_geo_ints
            plot_geo_tiles(
                data=m_domain[~m_nans].astype(int),
                lat=gg.lats[~m_nans],
                lon=gg.lons[~m_nans],
                plot_spec={
                    #"proj_out":"plate_carree",
                    "proj_out":"geostationary",
                    "proj_out_args":{
                        "central_longitude":gg.longitude_of_projection_origin,
                        "satellite_height":gg.perspective_point_height,
                        },
                    "cmap":"magma",
                    "cbar_orient":"horizontal",
                    "cartopy_feats":["states", "borders"],
                    },
                show=show,
                )
            ## 0 if OOB, 1 if IB & valid coords, 2 if IB & invalid coords
            nan_map = np.where(
                m_domain,
                np.where(m_domain&np.any(np.isnan(gg.latlon),axis=-1),2,1),
                0
                )
            plot_geo_ints(
                #int_data=nan_map,
                int_data=m_valid_2d.astype(int),
                lat=gg.lats[m_domain],
                lon=gg.lons[m_domain],
                cbar_ticks=True,
                latlon_ticks=False,
                #colors=["white", "blue", "red"],
                colors=["white", "blue"],
                show=show,
                )

        ## extract permutations
        if get_new_perms:
            if method == "global":
                permutations_from_configs(
                    dataset_name=Path(vdict["path"]).stem,
                    latlon=latlon,
                    configs=perm_configs_global,
                    mode="global",
                    pkl_dir=pkl_dir,
                    seed=200007221750,
                    max_iterations=64,
                    enum_start=0,
                    return_stats=True,
                    debug=debug,
                    )

            if method == "conv":
                permutations_from_configs(
                    dataset_name=Path(vdict["path"]).stem,
                    coords=latlon,
                    configs=perm_configs_conv[1::2],
                    mode="conv",
                    pkl_dir=pkl_dir,
                    #7seed=200007221750,
                    seed=7221751,
                    max_iterations=max_iterations,
                    enum_start=enum_start,
                    pool_size=pool_size,
                    return_stats=True,
                    nworkers=workers,
                    debug=debug,
                    )

            if method == "fast":
                ## configs provide:
                ## ["dist_threshold","jump_cap","shuffle_frac","seed","leaf_size"]
                ## to permutation.get_permutation_fast()
                permutations_from_configs(
                    dataset_name=Path(vdict["path"]).stem,
                    coords=latlon,
                    configs=perm_configs_fast,
                    pkl_dir=pkl_dir,
                    mode="fast",
                    enum_start=enum_start,
                    #seed=14750185, ## handled by config
                    #max_iterations=4, ## handled by config (jump_cap)
                    #pool_size=pool_size, ## not relevant for fast mode
                    kdt_workers=kdt_workers,
                    return_stats=True,
                    nworkers=workers,
                    debug=debug,
                    )



        ## Check how many chunks it would take to extract certain subgrids

        if plot_perms:
            print_pkls = [p for p in pkl_dir.iterdir()
                    if any(s in p.name for s in substrs)]
            #print(np.amin(latlon, axis=0), np.amax(latlon, axis=0))
            for pf in print_pkls:
                pnum = int(pf.stem.split("_")[-1])
                #mask = m_valid if pnum>=75 else m_valid_base ## for era5
                mask = m_valid
                try:
                    plot_perm_pkl(
                            perm_pkl=pf,
                            coords=latlon,
                            subgrids=test_subgrids,
                            geom=gg,
                            valid_mask=mask,
                            fig_dir=fig_dir,
                            chunk_size=chunk_size,
                            plot_stats=True,
                            plot_sparse_chunks=True,
                            num_sparse_chunks=50,
                            seed=200007221750,
                            debug=debug,
                            show=show,
                            )
                except Exception as e:
                    print(f"Failed for {pf.name}")
                    print(e)
                    #raise e
            #'''
