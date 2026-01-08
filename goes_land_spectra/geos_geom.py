"""
Module to provide a common library for calculating latitude, longitudes
and viewing zenith angles for geostationary satellites
"""
import numpy as np
import pickle as pkl
import warnings
from pvlib import solarposition
from pathlib import Path
from netCDF4 import Dataset

# Without ignoring, numpy throws lots of runtime warnings due to asymptotic
# trig values off the edge of the Earth.
warnings.filterwarnings('ignore', category=RuntimeWarning)

class GeosGeom:
    """
    Determine latitude, longitudes and viewing zenith angles from
    geostationary satellite viewing angles
    """
    def __init__(self, longitude_of_projection_origin:float,
            perspective_point_height:float, semi_major_axis:float,
            semi_minor_axis:float, sweep_angle_axis:str,
            e_w_scan_angles:np.array, n_s_scan_angles:np.array):
        """
        Calculate longitudes, latitudes and viewing zenith angles from
        satellite perspective and scan angles
        :param longitude_of_projection_origin: subpoint longitude (degrees)
        :param e_w_scan_angles: E/W 2d satellite viewing angles (radians)
        :param n_s_scan_angles: N/S 2d satellite viewing angles (radians)
        :param perspective_point_height: altitude of satellite over MSL (m)
        :param semi_major_axis: Radius of Earth at the equator (m)
        :param semi_minor_axis: Radius of Earth at the poles (m)
        :param sweep_angle_axis: Sweep angle axis of the satellite
        """
        self.R_e = 6371000
        self.longitude_of_projection_origin = longitude_of_projection_origin
        self.perspective_point_height = perspective_point_height
        self.r_eq = semi_major_axis
        self.r_pol = semi_minor_axis
        self.e_w_scan_angles = e_w_scan_angles
        self.n_s_scan_angles = n_s_scan_angles
        self.sweep_angle_axis = sweep_angle_axis
        self._lats,self._lons = self.get_latlon()
        self._vzas = None

    def args(self):
        return [
            {
                "longitude_of_projection_origin": \
                        self.longitude_of_projection_origin,
                "perspective_point_height":self.perspective_point_height,
                "semi_major_axis":self.r_eq,
                "semi_minor_axis":self.r_pol,
                "sweep_angle_axis":self.sweep_angle_axis,
                },
            {
                "sa_ew":self.e_w_scan_angles,
                "sa_ns":self.n_s_scan_angles
                }
            ]

    def __repr__(self):
        """ Returns a string reporting lat/lon sizes and ranges """
        return f"GeosGeom:\n" + \
                f"\tLats rng: ({np.amin(np.nan_to_num(self._lats, 99999))},"+\
                f" {np.amax(np.nan_to_num(self._lats, -99999))})  " + \
                f"SIZE: {self._lats.size}  " + \
                f"NaNs: {np.count_nonzero(np.isnan(self._lats))}\n" + \
                f"\tLons rng: ({np.amin(np.nan_to_num(self._lons, 99999))},"+\
                f" {np.amax(np.nan_to_num(self._lons, -99999))})  " + \
                f"SIZE: {self._lons.size}  " + \
                f"NaNs: {np.count_nonzero(np.isnan(self._lons))}"

    def get_nearest_indeces(self, lat, lon):
        """ returns the index of the pixel closest to the provided lat/lon """
        masked_lats = np.nan_to_num(self._lats, 999999)
        masked_lons = np.nan_to_num(self._lons, 999999)

        # Get an array of angular distance to the desired lat/lon
        lat_diff = masked_lats-lat
        lon_diff = masked_lons-lon
        total_diff = np.sqrt(lat_diff**2+lon_diff**2)
        min_idx = tuple([ int(c[0]) for c in
            np.where(total_diff == np.amin(total_diff)) ])
        return min_idx

    def get_subgrid_indeces(self, lat_range:tuple=None, lon_range:tuple=None,
                            _debug:bool=False):
        """
        Returns indeces of lat/lon values closest to the provided latitude
        or longitude range

        Both latitude and longitude ranges must be specified, or else the
        terminal indeces of the full lat/lon array will be returned.

        :param lat_range: (min, max) latitude in degrees.
                Defaults to full size.
        :param lon_range: (min, max) longitude in degrees.
                Defaults to full size.
        :param _debug: If True, prints information about the subgrid found.

        :return: latitude and longitude index ranges closest to the desired
                values, using the following tuple format. These indeces are
                reported like a 2d array, so lat_index_0 is actually the
                largest latitude value since 2d arrays count from the "top".
                ( (lat_index_0, lat_index_f),
                    (lon_index_0, lon_index_f) )
        """
        valid_lats = not lat_range is None and len(lat_range)==2 \
                and not next(l is None for l in lat_range)
        valid_lons = not lon_range is None and len(lon_range)==2 \
                and not next(l is None for l in lon_range)
        if not valid_lats or not valid_lons:
            if _debug:
                print("latitude and longitude ranges not specified;" + \
                        " defaulting to full array size.")
            minlat, maxlon = self._lats.shape
            return ((0,minlat), (0, maxlon))

        lr_latlon = [None, None]
        ul_latlon = [None, None]
        lr_latlon[0], ul_latlon[0] = sorted(lat_range)
        ul_latlon[1], lr_latlon[1] = sorted(lon_range)
        lr_latlon, ul_latlon = zip(lat_range, lon_range[::-1])
        ul_index = self.get_nearest_indeces(*ul_latlon)
        lr_index = self.get_nearest_indeces(*lr_latlon)

        if _debug:
            print("requested lat range: ",lat_range)
            print("requested lon range: ",lon_range)
            print("upper left pixel: ",ul_latlon, ul_index,
                  (self._lats[ul_index], self._lons[ul_index]))
            print("lower right pixel: ",lr_latlon, lr_index,
                  (self._lats[lr_index], self._lons[lr_index]))

        return tuple(zip(ul_index, lr_index))

    @property
    def latlon(self) -> np.array:
        """ :return: 1d numpy array of latitude values in degrees """
        return np.stack([self._lats,self._lons],axis=-1)

    @property
    def lats(self) -> np.array:
        """ :return: 1d numpy array of latitude values in degrees """
        return self._lats

    @property
    def lons(self) -> np.array:
        """ :return: 1d numpy array of longitude values in degeres """
        return self._lons

    @property
    def szas(self, time, altitude=None, pressure=None, method="nrel_numba",
           temperature=12.0,  **kwargs) -> np.array:
        """
        Calculate solar zenith angles using the NREL SPA algorithm

        :@param temperature: Surface temp in degrees celsius, which is only
            really needed for refraction at a very high SZA.
        """
        solarposition.get_solarposition(
            time,
            latitude=self.lats,
            longitude=self.lons,
            altitude=altitude,
            pressure=pressure,
            method=method,
            temperature=temperature,
            **kwargs,
            )

    @property
    def vzas(self) -> np.array:
        """
        Viewing zenith angle getter
        """
        if self._vzas is None:
            self._vzas = self.get_viewing_zenith_angles()
        return self._vzas

    def get_latlon(self, use_pyproj:bool=False)->(np.array, np.array):
        """
        Calculate latitude, longitude (degrees) values from GOES ABI fixed
        grid (radians).

        See GOES-R PUG Volume 4 Section 7.1.2.8 Navigation of Image Data
        https://www.goes-r.gov/users/docs/PUG-GRB-vol4.pdf,
        or just the wikipedia page on geodetic coordinates.

        :returns: longitudes, latitudes arrays
        """
        # GOES-17 values, GOES-16 values
        # lon_origin  # -137, -75
        # r_eq  # 6378137, 6378137.0
        # r_pol  # 6356752.31414, 6356752.31414
        # h  # 42164160, 42164160.0
        # lambda_0  # -2.3911010752322315, -1.3089969389957472

        lon_origin = self.longitude_of_projection_origin
        r_eq = self.r_eq
        r_pol = self.r_pol
        h = self.perspective_point_height + r_eq
        lambda_0 = (lon_origin * np.pi) / 180.0,
        sweep = self.sweep_angle_axis,

        # Geodedic coordinate transformation
        if use_pyproj:
            print("pyproj-based deprojection isn't supported yet.")
            #proj = Proj(proj='geos',h=str(), lon_0=str(lon_origin),
            #            sweep=sweep, R=self.R_e)

        sinlatr = np.sin(self.n_s_scan_angles)
        sinlonr = np.sin(self.e_w_scan_angles)
        coslatr = np.cos(self.n_s_scan_angles)
        coslonr = np.cos(self.e_w_scan_angles)

        # Both GOES sats
        # N/S sa: (.151844, -.151844)
        # E/W sa: (-.151844, .151844)

        r_eq2 = r_eq * r_eq
        r_pol2 = r_pol * r_pol

        v1 = np.square(coslatr) + r_eq2 * np.square(sinlatr) / r_pol2
        a_var = np.square(sinlonr) + np.square(coslonr) * v1

        b_var = -2.0 * h * coslonr * coslatr
        c_var = h ** 2.0 - r_eq2
        r_s = (-b_var - np.sqrt(
            (b_var ** 2) - (4.0 * a_var * c_var))) / (2.0 * a_var)
        s_x = r_s * coslonr * coslatr
        s_y = -r_s * sinlonr
        s_z = r_s * coslonr * sinlatr
        h_sx = h - s_x
        lats = np.degrees(np.arctan((r_eq2 / r_pol2) * s_z /
                                    np.sqrt((h_sx * h_sx) + (s_y * s_y))))
        lons = np.degrees(lambda_0 - np.arctan(s_y / h_sx))

        """
        print("lat min/max:",np.amin(np.nan_to_num(lats, 99999)),
              np.amax(np.nan_to_num(lats,-99999)))
        print("lon min/max:",np.amin(np.nan_to_num(lons, 99999)),
                np.amax(np.nan_to_num(lons,-99999)))
        print("lats size/nancount:",lats.size,
              np.count_nonzero(np.isnan(lats)))
        print("lons size/nancount:",lons.size,
              np.count_nonzero(np.isnan(lons)))
        """
        m_invalid = np.ma.getmask(lats) | np.ma.getmask(lons)
        return np.where(m_invalid,np.nan,lats),np.where(m_invalid,np.nan,lons)

    def get_viewing_zenith_angles(self) -> np.array:
        """
        Generate viewing zenith angles for each ABI fixed grid point

        Viewing zenith angle (vza) (simplified version - law of sines)
        https://www.ngs.noaa.gov/CORS/Articles/SolerEisemannJSE.pdf

        :returns: viewing zenith angle array
        """
        r_eq = self.r_eq
        h = self.perspective_point_height + r_eq
        theta_s = np.sqrt(self.e_w_scan_angles**2 + self.n_s_scan_angles**2)
        vzas = np.arcsin((h / r_eq) * np.sin(theta_s))
        return vzas

def load_geos_geom(geom_pkl_path, shape=None):
    """
    Loads geometry data from a pkl path. If given a valid stored domain shape,
    returns the associated data as a 2-tuple (GeosGeom, m_domain). If no
    shape is provided, all stored domains are returned as a dict mapping domain
    shapes to 2-tuples as above.
    """
    assert geom_pkl_path.exists()
    #with geom_index_lock:
    ggdict = pkl.load(geom_pkl_path.open("rb"))
    if not shape is None:
        return (GeosGeom(**ggdict[shape][0]), ggdict[shape][1])
    ## return the smallest resolution by default
    shape = next(sorted(ggdict.keys(), key=lambda t:t[0]*t[1]))
    return {s:(GeosGeom(**ggdict[s][0]),ggdict[s][1]) for s in ggdict.keys()}

def dump_geos_geom(geom_dir, cur_geom, nc_path,
        cur_shape=None, domain_mask=None, debug=False):
    """
    Multiprocess friendly function that maintains a directory of GeosGeom pkl
    files. The directory has an index file mapping basic projection values
    to a pkl of radian arrays and geometry data associated with that satellite
    position.

    1. check if the index file exists. If it does, then load it. Otherwise,
       create a new one.

    2. load the index file. Check if the match_fields of user-provided
       cur_geom dict are all equal to one of the index entries.
       If so, get the geom pkl's path, and check whether the current array
       resolution is supported.

       2a. If the pkl exists, is indexed, and supports the current resolution,
           just return its path.
       2b. If the pkl is not indexed or doesn't exist, create a new entry and
           pkl given the current parameters, resolution, and scan angles.
       2c. If the pkl is indexed and exists but doesn't support the current
           resolution, load it and update its dictionary with the scan angles
           from nc_path.

    :@param geom_dir: Directory where geometry pkls and index will be placed.
    :@param cur_geom: projection dictionary for the current netCDF
    :@param nc_path: path to the netCDF associated with cur_geom in case scan
        angle arrays need to be extracted for a new geom.
    :@param cur_shape: If provided, verifies that cur_shape matches the shapes
        of viewing angle arrays already stored in the geom, and stores it
        otherwise.
    :@param domain_mask: Optional mask setting True to valid points in the
        domain, which enables throwing away pixels that are always OOB.
    """
    match_fields = (
        "perspective_point_height",
        "longitude_of_projection_origin",
        "semi_major_axis",
        "semi_minor_axis",
        )
    assert geom_dir.exists()
    index_path = geom_dir.joinpath("index.pkl")

    #geom_index_lock.acquire() ## acquire the semaphor
    ## load the index file of geometries
    if index_path.exists():
        with index_path.open("rb") as index_file:
            geoms = pkl.load(index_file)
            #print(f"geom OPEN {index_path.name} {current_process()}",
            #        flush=True)
    else:
        geoms = {}
    #for k,v in geoms.items():
    #    print(f"geom CUR {k} {v['shapes']} {current_process()}", flush=True)
    ## construct the key associated with this projection
    cur = tuple(float(cur_geom[k]) for k in match_fields)
    geom_pkl_path = None
    store_new_shape = False
    ## see if there is a pkl indexed that matches this projection
    for k,v in list(geoms.items())[::-1]:
        #if np.all(np.isclose(cur,k)):
        if tuple(cur)==tuple(k):
            cur = k ## so keys match later
            geom_pkl_path = Path(v["path"])
            if not cur_shape is None and cur_shape not in v["shapes"]:
                store_new_shape = True

    ## if there is a pkl for this geom configuration, return the path if the
    ## current shape is indexed. If not, update the pkl and return the path.
    if (not geom_pkl_path is None) and geom_pkl_path.exists():
        if not store_new_shape:
            return geom_pkl_path
        ## configured and existing but needs shape
        with geom_pkl_path.open("rb") as geom_file:
            ggdict = pkl.load(geom_file)
        ds = Dataset(nc_path, "r")
        sa_ns,sa_ew = np.meshgrid(
                ds["y"][...],
                ds["x"][...],
                indexing="ij",
                )
        ds.close()
        assert cur_shape == sa_ns.shape, \
                f"reported shape {cur_shape} != {sa_ns.shape}"
        ggdict[sa_ns.shape] = ({
            "n_s_scan_angles":sa_ns.data,
            "e_w_scan_angles":sa_ew.data,
            "sweep_angle_axis":"x",
            **{f:cur_geom[f] for f in match_fields},
            }, domain_mask)
        with geom_pkl_path.open("wb") as geom_file:
            pkl.dump(ggdict, geom_file)
        ## update the index listing with the new shape
        assert store_new_shape
        geoms[cur]
        geoms[cur]["shapes"].append(cur_shape)
        with index_path.open("wb") as index_file:
            pkl.dump(geoms, index_file)
        #print(f"geom UPDATE {cur} {cur_shape} " \
        #    + f"{np.count_nonzero(domain_mask)} {current_process()}",
        #    flush=True)
        return geom_pkl_path

    ## if here, geom may exist or be configured, but not both.
    ## in both cases, just make a new one.
    geom_pkl_path = geom_dir.joinpath(
        f"geom-goes-conus-{len(geoms.keys())}.pkl")
    ds = Dataset(nc_path, "r")
    sa_ns,sa_ew = np.meshgrid(ds["y"][...], ds["x"][...], indexing="ij")
    ds.close()
    ggdict = {
        sa_ns.shape:({
            "n_s_scan_angles":sa_ns.data,
            "e_w_scan_angles":sa_ew.data,
            "sweep_angle_axis":"x",
            **{f:cur_geom[f] for f in match_fields},
            }, domain_mask)
        }
    with geom_pkl_path.open("wb") as geom_file:
        pkl.dump(ggdict, geom_file)
    geoms.update({cur:{"path":geom_pkl_path, "shapes":[cur_shape]}})
    #print(f"geom NEW {cur} {cur_shape} " \
    #    + f"{np.count_nonzero(domain_mask)} {current_process()}", flush=True)
    with index_path.open("wb") as index_file:
        pkl.dump(geoms, index_file)
    return geom_pkl_path
