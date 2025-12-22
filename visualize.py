import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime,timedelta
from pprint import pprint

from GeosGeom import GeosGeom

if __name__=="__main__":
    proj_root = Path("/rhome/mdodson/goes-land-spectra")
    ## directory where domain arrays will be stored.
    geom_dir = proj_root.joinpath("data/domains")
    ## directory where pkls of listings will be stored.
    listing_dir = proj_root.joinpath("data/listings")
    out_dir = proj_root.joinpath("data/results")

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
