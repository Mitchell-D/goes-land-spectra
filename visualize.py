import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime,timedelta
from pprint import pprint

from GeosGeom import GeosGeom
from plotting import plot_geo_rgb,plot_geo_scalar
from acquire import merge_welford

class QueryResults:
    """
    search files given a common underscore-separated file structure.
    """
    def __init__(self, file_paths, name_fields):
        self._p = file_paths
        self._f = name_fields

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
        sub_paths,_ = zip(*[
            (p,pt) for p,pt in self.tups
            if all(any((pt[i]==s) if isinstance(s,str) else (pt[i] in s)
                for s in kwargs.get(k,[pt[i]])) for i,k in enumerate(self._f))
            ])
        return QueryResults(sub_paths, self._f)

    def __repr__(self):
        return list(map(lambda p:p.as_posix(),self._p))

    def group(self, group_fields:list, invert=False):
        """
        return pkls that share a combination of group_fields
        """
        groups = {}
        assert all(f in self._f for f in group_fields),group_fields
        if invert:
            group_fields = list(set(self._f)-set(group_fields))
        gixs = [self._f.index(f) for f in group_fields]
        for p,t in self.tups:
            gkey = tuple(t[ix] for ix in gixs)
            if gkey not in groups.keys():
                groups[gkey] = []
            groups[gkey].append(p)
        return groups

if __name__=="__main__":
    proj_root = Path("/rhome/mdodson/goes-land-spectra")
    ## directory where domain arrays will be stored.
    geom_dir = proj_root.joinpath("data/domains")
    ## directory where pkls of listings will be stored.
    listing_dir = proj_root.joinpath("data/listings")
    out_dir = proj_root.joinpath("data/results")

    name_fields = ["satellite","listing", "stime", "ftime",
            "domain", "month", "tod", "band"]

    ## result pkls are stratified by domain, listing, ToD and band
    include_only = {
        "domain":["geom-goes-conus-1"], ## G19E
        "listing":["clearland-l1b-c0"],
        #"tod":["64800"], ## in seconds, only 18z
        "tod":["54000","64800","75600"], ## in seconds, only 18z
        "month":["11","12","01"],
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
        mrg_keys,mrg_paths = zip(*qr.group(merge_over, invert=True).items())

        pprint(dict(zip(mrg_keys,mrg_paths)))

        ## narrow down result pkls to the requested combination
        '''
        res_paths,res_tups = zip(*[
            (p,pt) for p,pt in map(
                lambda r:(r,r.stem.split("_")), out_dir.iterdir())
            if any(pt[0]==s for s in tmp_incl.get("satellite",[pt[0]]))
            and any(pt[1]==s for s in tmp_incl.get("listing", [pt[1]]))
            and any(pt[2]==s for s in tmp_incl.get("stime", [pt[2]]))
            and any(pt[3]==s for s in tmp_incl.get("ftime", [pt[3]]))
            and any(pt[4]==s for s in tmp_incl.get("domain", [pt[4]]))
            and any(pt[5]==s for s in tmp_incl.get("month", [pt[5]]))
            and any(pt[6]==s for s in tmp_incl.get("tod", [pt[6]]))
            and any(pt[7]==s for s in tmp_incl.get("band", [pt[7]]))
            ])
        pprint(res_paths)
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
