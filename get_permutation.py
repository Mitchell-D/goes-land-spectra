import numpy as np
import pickle as pkl
import time
from pprint import pprint
from pathlib import Path
from datetime import datetime,timedelta

from geos_geom import load_geos_geom
from plotting import plot_geo_ints

if __name__=="__main__":
    proj_root = Path("/rhome/mdodson/goes-land-spectra/")
    geom_dir = proj_root.joinpath("data/domains/")
    geom_index = pkl.load(geom_dir.joinpath("index.pkl").open("rb"))

    permute_geoms = [
        "geom-goes-conus-0",
        ]

    for view_key,vdict in geom_index.items():
        geom_path = geom_dir.joinpath(Path(vdict["path"]).name)
        if not geom_path.stem in permute_geoms:
            continue
        data_shapes = vdict["shapes"]
        domain_shape = sorted(data_shapes, key=lambda t:t[0]*t[1])[0]
        gg,m_domain = load_geos_geom(geom_path, shape=domain_shape)
        print(m_domain.shape, np.count_nonzero(m_domain), m_domain.size)
        print(np.count_nonzero(np.isnan(gg.latlon[m_domain])))
        plot_geo_ints(
            int_data=np.where(np.any(m_domain & np.isnan(gg.latlon)), axis=-1),
            lat=gg.lats,
            lon=gg.lons,
            show=True,
            )
