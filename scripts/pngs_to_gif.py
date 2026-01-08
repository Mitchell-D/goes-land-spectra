import numpy as np
import imageio.v3 as iio
from pathlib import Path
from pprint import pprint
from multiprocessing import Pool

from goes_land_spectra.helpers import QueryResults,mp_gen_gif_from_group

if __name__=="__main__":
    proj_root = Path("/rhome/mdodson/goes-land-spectra")
    fig_dir = proj_root.joinpath("figures/scalar")
    gif_dir = proj_root.joinpath("figures/gifs-tod")
    name_fields = ["imtype", "sat", "listing", "t0", "tf", "domain",
            "month", "tod", "band", "metric"]
    nworkers = 20
    subset = {
        "imtype":"scalar",
        "sat":"goes16",
        "listing":"clearland-l1b-c0",
        "t0":"20170701",
        "tf":"20240630",
        "domain":"geom-goes-conus-0",
        }

    ## combination of fields that defines a distinct gif.
    group_fields = [
        ## include subset fields so that they appear in the file name
        "imtype", "sat", "listing", "t0", "tf", "domain",
        #"tod", "band", "metric", ## (tod,band,metric) combos to distinct gifs
        "month", "band", "metric",
        ]

    qr = QueryResults(fig_dir.iterdir(),name_fields)
    _,pgroups = qr.subset(subset).group(group_fields)

    print(f"found {len(pgroups)} distinct file groups")

    args = [{
        "group_key":k,
        "group_paths":sorted(v),
        "group_fields":group_fields,
        "name_fields":name_fields,
        "out_dir":gif_dir,
        "duration":.25
        } for k,v in pgroups.items()
        ]

    with Pool(nworkers) as pool:
        for args,rp in pool.imap_unordered(mp_gen_gif_from_group, args):
            print(f"Generated {rp.as_posix()}")
