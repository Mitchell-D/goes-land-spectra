import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pprint import pprint
from pathlib import Path
from datetime import datetime,timedelta
from cartopy.mpl.ticker import LatitudeFormatter,LongitudeFormatter
from matplotlib.colors import ListedColormap

def _process_rgb_data(rgb_array, gamma, vmin=None, vmax=None, pct_clip=None):
    """
    Process RGB data with normalization and gamma correction.
    If vmin and vmax aren't available, clip between data percentiles.
    """
    def normalize_channel(data, vmin, vmax):
        valid_data = data[~np.isnan(data)]
        if vmin is None or vmax is None:
            plow, phigh = pct_clip
            vmin, vmax = np.percentile(valid_data, [plow, phigh])

        norm_data = np.clip((data - vmin) / (vmax - vmin), 0, 1)
        gamma_data = np.power(norm_data, 1/gamma)
        return gamma_data

    r = normalize_channel(rgb_array[:, :, 0], vmin, vmax)
    g = normalize_channel(rgb_array[:, :, 1], vmin, vmax)
    b = normalize_channel(rgb_array[:, :, 2], vmin, vmax)

    return np.dstack([r, g, b])

def plot_geostationary(
        data, sat_lon, sat_height, sat_sweep, plot_spec={}, fig_path=None,
        show=False, debug=False):
    """
    """
    ## Default plot specifications
    ps = {
        "gamma": None, "vmin": None, "vmax": None,
        "pct_clip": (0.5, 99.5),
        "cmap":"plasma","colorbar":True,"cb_label":"","cb_orient":"vertical",
        "projection":{"type":"geographic"}, "extent":None,
        "add_states":True, "add_coastlines":True, "borders_color":"black",
        "lw_borders":1,"lw_states":.5,"lw_coastlines": 1,
        "gridlines":True,"gridlines_color":"white","gridlines_alpha": 0.5,
        "title": "","figsize":(12,9),"dpi":120,"interp":"nearest",
        "cartopy_feats":["land","borders","states"],
        }
    ps.update(plot_spec)

    ## parse input data
    if isinstance(data, dict):
        ## dict with "red", "green", "blue" keys
        rgb_mode = True
        data_array = np.dstack([data["red"], data["green"], data["blue"]])
    elif data.ndim == 3 and data.shape[2] == 3:
        ## RGB array (Y, X, 3)
        rgb_mode = True
        data_array = data
    elif data.ndim == 2:
        ## Single channel (Y, X)
        rgb_mode = False
        data_array = data
    else:
        raise ValueError(
                f"Invalid data shape: {data.shape}. Expected (Y,X,3) or (Y,X)")

    ## set default gamma based on mode
    if ps["gamma"] is None:
        ps["gamma"] = 2.0 if rgb_mode else 1.0

    if rgb_mode:
        ## process RGB composite
        processed_data = _process_rgb_data(
                rgb_array=data_array,
                gamma=ps.get("gamma"),
                vmin=ps.get("vmin"),
                vmax=ps.get("vmax"),
                pct_clip=ps.get("pct_clip"),
                )
    else:
        ## process single channel, optionally applying gamma correction,
        ## percentile clipping, and color mapping
        processed_data = data_array
        '''
        valid_data = data[~np.isnan(data)]
        vmin,vmax = ps.get("vmin"),ps.get("vmax")
        if vmin is None or vmax is None:
            plow, phigh = ps.get("pct_clip")
            vmin, vmax = np.percentile(valid_data, [plow, phigh])
        norm_data = np.clip((data_array - vmin) / (vmax - vmin), 0, 1)
        gamma_data = np.power(norm_data, 1/ps.get("data"))
        cmap = plt.get_cmap(ps.get("cmap"))
        processed_data = cmap(gamma_data)
        '''

    # Set up projection
    proj_spec = ps["projection"]
    if proj_spec["type"] == "geostationary":
        proj = ccrs.Geostationary(
            central_longitude=sat_lon,
            satellite_height=sat_height,
            sweep_axis=sat_sweep
            )
    else:
        proj = ccrs.PlateCarree()

    transform = proj
    fig = plt.figure(figsize=ps["figsize"], dpi=ps["dpi"])
    ax = fig.add_subplot(1, 1, 1, projection=proj)

    extent = ps["extent"]
    if proj_spec["type"] == "geostationary":
        if extent is not None:
            im = ax.imshow(
                    processed_data,
                    extent=extent,
                    origin="upper",
                    interpolation=ps["interp"],
                    cmap=ps.get("cmap"),
                    )
        else:
            im = ax.imshow(
                    processed_data,
                    origin="upper",
                    interpolation=ps["interp"],
                    cmap=ps.get("cmap"),
                    )
    else:
        if extent is not None:
            im = ax.imshow(
                    processed_data,
                    extent=extent,
                    origin="upper",
                    interpolation=ps["interp"],
                    transform=transform,
                    cmap=ps.get("cmap"),
                    )
        else:
            im = ax.imshow(
                    processed_data,
                    origin="upper",
                    interpolation=ps["interp"],
                    transform=transform,
                    cmap=ps.get("cmap"),
                    )

    #if proj_spec["type"] == "geostationary" and extent is not None:
    #    ax.set_extent(extent, crs=transform)

    # Add colorbar for single channel
    if not rgb_mode and ps["colorbar"]:
        cbar = plt.colorbar(
                im,
                ax=ax,
                orientation=ps.get("cb_orient","horizontal"),
                cmap=ps.get("cmap"),
                pad=ps.get("cb_pad", .02),
                shrink=ps.get("cb_shrink", .9),
                )
        cbar.set_label(ps["cb_label"], fontsize=11)

    # Add map features
    if "coastlines" in ps.get("cartopy_feats"):
        ax.coastlines(
                resolution="50m",
                color=ps["borders_color"],
                linewidth=ps["lw_coastlines"],
                zorder=3,
                )

    if "borders" in ps.get("cartopy_feats"):
        ax.add_feature(
                cfeature.BORDERS,
                linewidth=ps["lw_borders"],
                edgecolor=ps["borders_color"],
                zorder=3,
                )

    if "states" in ps.get("cartopy_feats"):
        ax.add_feature(
                cfeature.STATES.with_scale("50m"),
                linewidth=ps["lw_states"],
                edgecolor=ps["borders_color"],
                zorder=3,
                )

    if ps.get("gridlines"):
        gl = ax.gridlines(
            draw_labels=proj_spec["type"] != "geostationary",
            linewidth=0.5,
            color=ps["gridlines_color"],
            alpha=ps["gridlines_alpha"],
            linestyle="--",
            zorder=4
            )
        gl.top_labels = False
        gl.right_labels = False

    ax.set_title(ps["title"], fontsize=14, weight="bold")
    plt.tight_layout()

    # Save or show
    if show:
        plt.show()
    if fig_path:
        plt.savefig(fig_path, bbox_inches="tight", dpi=ps["dpi"])
        if debug:
            print(f"Generated {fig_path.as_posix()}")

    return fig, ax

def plot_rgb_geos_on_plate(
        rgb:np.ndarray, center_lon, sat_height, semi_major, semi_minor,
        sweep_axis, lat_range:tuple, lon_range:tuple, plot_spec:dict={},
        fig_path=None, show=False):
    """
    """
    ps = {
        "title":"", "figsize":(16,12), "border_linewidth":2, "title_size":12,
        "proj_in":"plate_carree", "proj_in_args":{},
        "proj_out":"plate_carree", "proj_out_args":{},
        }

    plt.clf()
    ps.update(plot_spec)
    plt.rcParams.update({"font.size":ps.get("text_size", 12)})
    fig = plt.figure(figsize=ps.get("figsize"))

    globe = ccrs.Globe(semimajor_axis=semi_major, semiminor_axis=semi_minor)
    crs_in = ccrs.Geostationary(
            central_longitude=center_lon,
            satellite_height=sat_height,
            sweep_axis=sweep_axis,
            globe=globe,
            )
    crs_out = ccrs.PlateCarree()

    ## Transform the lower-left and upper-right corners
    lower_left = crs_in.transform_point(lon_range[0], lat_range[0], crs_out)
    upper_right = crs_in.transform_point(lon_range[1], lat_range[1], crs_out)
    extent = (lower_left[0], upper_right[0], lower_left[1], upper_right[1])

    ax = plt.axes(projection=crs_in)

    ax.imshow(rgb, extent=extent, transform=crs_in, origin="upper",
        interpolation=ps.get("interp", "none"))

    ax.coastlines(
            color=ps.get("border_color", "black"),
            linewidth=ps.get("border_linewidth"))
    ax.add_feature(
            ccrs.cartopy.feature.STATES,
            #color=ps.get("border_color", "black"),
            linewidth=ps.get("border_linewidth")
            )

    plt.title(ps.get("title"), fontweight='bold',
            fontsize=ps.get("title_size"))

    ax.set_extent(
            [lon_range[0], lon_range[1], lat_range[0], lat_range[1]],
            crs=crs_out
            )

    if not fig_path is None:
        fig.savefig(fig_path.as_posix(), bbox_inches="tight", dpi=80)
    if show:
        plt.show()
    plt.close()
    return

def plot_multiy_lines(data, xaxis, plot_spec={},
        show=False, fig_path=None):
    """
    """
    ps = {"fig_size":(12,6), "dpi":80, "spine_increment":.01,
            "date_format":"%Y-%m-%d", "xtick_rotation":30}
    ps.update(plot_spec)
    if len(xaxis) != len(data[0]):
        raise ValueError(
                "Length of 'xaxis' must match length of each dataset.")

    fig,host = plt.subplots(figsize=ps.get("fig_size"))
    fig.subplots_adjust(left=0.2 + ps.get("spine_increment") \
            * (len(data) - 1))

    axes = [host]
    colors = ps.get("colors", ["C" + str(i) for i in range(len(data))])
    y_labels = ps.get("y_labels", [""] * len(data))
    y_ranges = ps.get("y_ranges", [None] * len(data))

    ## Create additional y-axes on the left, offset horizontally
    for i in range(1, len(data)):
        ax = host.twinx()
        #ax.spines["left"] = ax.spines["right"]
        ax.yaxis.set_label_position("left")
        ax.yaxis.set_ticks_position("left")
        ax.spines["left"].set_position(
                ("axes", -1*ps.get("spine_increment") * i))
        axes.append(ax)

    ## Plot each series
    for i, (ax, series) in enumerate(zip(axes, data)):
        ax.plot(xaxis, series, color=colors[i], label=y_labels[i])
        ax.set_ylabel(y_labels[i], color=colors[i],
                fontsize=ps.get("label_size"))
        ax.tick_params(axis="y", colors=colors[i])
        if y_ranges[i] is not None:
            ax.set_ylim(y_ranges[i])

    host.set_xlabel(ps.get("x_label", "Time"), fontsize=ps.get("label_size"))
    host.tick_params(axis="x", rotation=ps.get("xtick_rotation"))

    if plot_spec.get("xtick_align"):
        plt.setp(host.get_xticklabels(),
                horizontalalignment=plot_spec.get("xtick_align"))

    if ps.get("zero_axis"):
        host.axhline(0, color="black")

    plt.title(ps.get("title", ""), fontdict={"fontsize":ps.get("title_size")})
    plt.tight_layout()
    if show:
        plt.show()
    if not fig_path is None:
        fig.savefig(fig_path, bbox_inches="tight", dpi=plot_spec.get("dpi"))
    plt.close()

def plot_geo_rgb(rgb_data, lat, lon, shapes=None,
    geo_bounds=None, latlon_ticks=True,
    int_labels=None, fig_path=None, cbar_ticks=False,
    show=False, plot_spec={}):
    """
    Plots a map with pixels colored according to a 2D array of integer values.

    :@param rgb_data: 2D numpy array of integer values to be visualized
    :@param latitudes: 1D array of latitudes corresponding to rows in `data`
    :@param longitudes: 1D array of longitudes corresponding to cols in`data`
    """
    ps = {
        "xlabel":"", "ylabel":"", "title":"", "dpi":80, "norm":None,
        "figsize":(12,12), "legend_ncols":1, "line_opacity":1, "cmap":"hsv",
        "label_size":14, "title_size":20, "shape_params":{"edgecolor":"black"},
        "cartopy_feats":["land", "borders", "states"],
        }
    ps.update(plot_spec)
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})

    if shapes:
        ax.add_geometries(
                shapes, ccrs.PlateCarree(), **ps.get("shape_params"))

    if "land" in ps.get("cartopy_feats"):
        ax.add_feature(cfeature.LAND)
    if "borders" in ps.get("cartopy_feats"):
        ax.add_feature(
                cfeature.BORDERS,
                linestyle=ps.get("border_style", "-"),
                linewidth=ps.get("border_linewidth", 2),
                edgecolor=ps.get("border_color", "black"),
                )
    if "states" in ps.get("cartopy_feats"):
        ax.add_feature(
                cfeature.STATES,
                linestyle=ps.get("border_style", "-"),
                linewidth=ps.get("border_linewidth", 2),
                edgecolor=ps.get("border_color", "black"),
                )
    if geo_bounds is None:
        geo_bounds = [np.amin(lon), np.amax(lon), np.amin(lat), np.amax(lat)]
    ax.set_extent(geo_bounds, crs=ccrs.PlateCarree())

    m_invalid = ~np.isfinite(rgb_data)
    rgb_data[m_invalid] = rgb_data[~m_invalid][0]
    rgb_data = rgb_data.astype(int)

    im = ax.imshow(
            rgb_data,
            origin=ps.get("origin", "upper"),
            cmap=cmap,
            extent=geo_bounds,
            interpolation=ps.get("interpolation")
            )

    if latlon_ticks:
        lonmin,lonmax,latmin,latmax = geo_bounds
        frq = ps.get("tick_frequency", 1)
        ax.set_yticks(np.linspace(latmin,latmax,rgb_shape.shape[0])[::frq],
                crs=ccrs.PlateCarree())
        ax.set_xticks(np.linspace(lonmin,lonmax,rgb_shape.shape[1])[::frq],
                crs=ccrs.PlateCarree())
        lon_formatter = LongitudeFormatter(zero_direction_label=True)
        lat_formatter = LatitudeFormatter()
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.yaxis.set_major_formatter(lat_formatter)
        ax.tick_params(rotation=ps.get("tick_rotation", 0))

    cbar.set_label(ps.get("cbar_label"))
    ax.set_title(ps.get("title", ""), fontsize=ps.get("title_fontsize", 18))
    if not fig_path is None:
        fig.set_size_inches(*ps.get("figsize"))
        fig.savefig(fig_path.as_posix(), bbox_inches="tight",
                dpi=ps.get("dpi", 80))
    if show:
        plt.show()
    plt.close()
    return

def plot_geo_ints(int_data, lat, lon, shapes=None,
    geo_bounds=None, latlon_ticks=False,
    int_labels=None, fig_path=None, cbar_ticks=False, colors=None,
    show=False, plot_spec={}):
    """
    Plots a map with pixels colored according to a 2D array of integer values.

    :@param int_data: 2D numpy array of integer values to be visualized
    :@param latitudes: 1D array of latitudes corresponding to rows in `data`
    :@param longitudes: 1D array of longitudes corresponding to columns in`data`
    :@param colors: list or dict mapping indeces present in int_data to
        matplotlib-valid colors
    """
    ps = {
        "xlabel":"", "ylabel":"", "title":"", "dpi":80, "norm":None,
        "figsize":(12,12), "legend_ncols":1, "line_opacity":1, "cmap":"hsv",
        "label_size":14, "title_size":20, "shape_params":{"edgecolor":"black"},
        "cartopy_feats":["land", "borders", "states"],
        }
    ps.update(plot_spec)
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})

    if shapes:
        ax.add_geometries(
                shapes, ccrs.PlateCarree(), **ps.get("shape_params"))

    if "land" in ps.get("cartopy_feats"):
        ax.add_feature(
                cfeature.LAND,
                #linestyle=ps.get("border_style", "-"),
                #linewidth=ps.get("border_linewidth", 2),
                #edgecolor=ps.get("border_color", "black"),
                )
    if "borders" in ps.get("cartopy_feats"):
        ax.add_feature(
                cfeature.BORDERS,
                linestyle=ps.get("border_style", "-"),
                linewidth=ps.get("border_linewidth", 2),
                edgecolor=ps.get("border_color", "black"),
                )
    if "states" in ps.get("cartopy_feats"):
        ax.add_feature(
                cfeature.STATES,
                linestyle=ps.get("border_style", "-"),
                linewidth=ps.get("border_linewidth", 2),
                edgecolor=ps.get("border_color", "black"),
                )
    if geo_bounds is None:
        geo_bounds = [np.nanmin(lon), np.nanmax(lon), np.nanmin(lat), np.nanmax(lat)]
    ax.set_extent(geo_bounds, crs=ccrs.PlateCarree())

    m_invalid = ~np.isfinite(int_data)
    int_data[m_invalid] = int_data[~m_invalid][0]
    int_data = int_data.astype(int)

    ## assign each unique integer to an index
    unq_ints = np.unique(int_data)
    val_to_ix = {v:ix for ix,v in enumerate(unq_ints)}
    if colors is None:
        ref_cmap = plt.get_cmap(ps.get("cmap", "tab20"), unq_ints.size)
        cmap = ListedColormap([ref_cmap(i) for i in range(unq_ints.size)])
    else:
        cmap = ListedColormap([colors[v] for v in unq_ints])
    if int_labels is None:
        ix_labels = list(unq_ints)
    else:
        ix_labels = [int_labels[v] for v in unq_ints]
    ix_data = np.vectorize(val_to_ix.get)(int_data).astype(float)
    ix_data[m_invalid] = np.nan

    im = ax.imshow(
            ix_data,
            origin=ps.get("origin", "upper"),
            cmap=cmap,
            extent=geo_bounds,
            interpolation=ps.get("interpolation")
            )

    if latlon_ticks:
        lonmin,lonmax,latmin,latmax = geo_bounds
        frq = ps.get("tick_frequency", 1)
        ax.set_yticks(np.linspace(latmin,latmax,ix_data.shape[0])[::frq],
                crs=ccrs.PlateCarree())
        ax.set_xticks(np.linspace(lonmin,lonmax,ix_data.shape[1])[::frq],
                crs=ccrs.PlateCarree())
        lon_formatter = LongitudeFormatter(zero_direction_label=True)
        lat_formatter = LatitudeFormatter()
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.yaxis.set_major_formatter(lat_formatter)
        ax.tick_params(rotation=ps.get("tick_rotation", 0))
    cbar = plt.colorbar(
            im, ax=ax,
            orientation=ps.get("cbar_orient", "vertical"),
            pad=ps.get("cbar_pad", 0.05),
            shrink=ps.get("cbar_shrink", 1.)
            )

    ## make a scale that centers ticks on their color bar increments
    if cbar_ticks:
        nunq = unq_ints.size
        ticks = np.linspace(0, nunq-1, nunq*2+1)[1::2]
        #ticks = np.array(list(range(nunq))) * (nunq-1)/nunq + .5
        cbar.set_ticks(ticks)
        cbar.ax.tick_params(rotation=ps.get("cbar_tick_rotation", 0))
        cbar.set_ticklabels(ix_labels)
        cbar.ax.tick_params(labelsize=ps.get("cbar_fontsize", 14))

    cbar.set_label(ps.get("cbar_label"))
    ax.set_title(ps.get("title", ""), fontsize=ps.get("title_fontsize", 18))
    if not fig_path is None:
        fig.set_size_inches(*ps.get("figsize"))
        fig.savefig(fig_path.as_posix(),
                bbox_inches="tight",dpi=ps.get("dpi", 80))
    if show:
        plt.show()
    plt.close()
    return

def plot_geo_scalar(data, lat, lon, hatch_data=None, shapes=None,
        bounds=None, plot_spec={}, latlon_ticks=False, show=False,
        fig_path=None, use_contours=False):
    """
    Plot a gridded scalar value on a geodetic domain, using cartopy for borders
    """
    ps = {
        "xlabel":"", "ylabel":"", "marker_size":4, "cmap":"jet_r", "dpi":200,
        "text_size":12, "title":"", "norm":"linear","figsize":None,
        "marker":"o", "cbar_shrink":1., "map_linewidth":2,
        "hatch_shading":"auto", "hatch_edgecolor":"black", "hatch_style":"xxx",
        "hatch_linewidth":1, "hatch_facecolor":"none", "hatch_edgewidth":1,
        "shape_params":{"edgecolor":"black"},
        "cartopy_feats":["land", "borders", "states"],
        "cbar_extendfrac":0.05, "custom_cmap_params":None,
        "proj":"plate_carree", "proj_args":{},
        }
    plt.clf()
    ps.update(plot_spec)
    plt.rcParams.update({"font.size":ps["text_size"]})

    crs_type = {
        "plate_carree":ccrs.PlateCarree,
        "geostationary":ccrs.Geostationary,
        }
    ## for geostationary, need central_longitude and satellite_height args
    crs = crs_type[ps.get("proj")](**ps.get("proj_args"))
    ax = plt.axes(projection=crs)
    fig = plt.gcf()
    if bounds is None:
        bounds = [np.amin(lon), np.amax(lon),
                  np.amin(lat), np.amax(lat)]
    ax.set_extent(bounds, crs=crs)

    ax.add_feature(cfeature.LAND, linewidth=ps.get("map_linewidth"))
    #ax.add_feature(cfeature.LAKES, linewidth=ps.get("map_linewidth"))
    #ax.add_feature(cfeature.RIVERS, linewidth=ps.get("map_linewidth"))

    ax.set_title(ps.get("title"), fontsize=ps.get("fontsize_title", 12))
    ax.set_xlabel(ps.get("xlabel"), fontsize=ps.get("fontsize_labels", 10))
    ax.set_ylabel(ps.get("ylabel"), fontsize=ps.get("fontsize_labels", 10))

    cmap_params = {}
    if not ps.get("custom_cmap_params") is None:
        ccp = ps["custom_cmap_params"]
        cmap = matplotlib.colors.ListedColormap(ccp["colors"])
        if "extremes" in ccp.keys():
            assert isinstance(ccp["extremes"], (list,tuple))
            exlow,exhigh = ccp["extremes"]
            #cmap = cmap.with_extremes(under=exlow, over=exhigh)
            cmap.set_over(exhigh)
            cmap.set_under(exlow)
        norm = matplotlib.colors.BoundaryNorm(ccp["bounds"], cmap.N)
        cmap_params = {"cmap":cmap, "norm":norm}
    else:
        cmap_parms = {
            "cmap":ps.get("cmap"),
            "norm":ps.get("norm"),
            "vmin":ps.get("vmin"),
            "norm":ps.get("vmax"),
            }

    if use_contours:
        scat = ax.contourf(lon, lat, data, **cmap_params)
    else:
        scat = ax.pcolormesh(lon, lat, data, **cmap_params)

    if latlon_ticks:
        lonmin,lonmax,latmin,latmax = bounds
        frq = ps.get("tick_frequency", 1)
        ax.set_yticks(np.linspace(latmin,latmax,data.shape[0])[::frq],
                crs=crs)
        ax.set_xticks(np.linspace(lonmin,lonmax,data.shape[1])[::frq],
                crs=crs)
        lon_formatter = LongitudeFormatter(zero_direction_label=True)
        lat_formatter = LatitudeFormatter()
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.yaxis.set_major_formatter(lat_formatter)
        ax.tick_params(rotation=ps.get("tick_rotation", 0))

    if not hatch_data is None:
        hatch_plot = ax.contourf(
            lon,
            lat,
            hatch_data.astype(int),
            levels=[0.5,1.5],
            hatches=ps.get("hatch_style"),
            #linewidth=ps.get("hatch_linewidth"),
            colors="none",
            #edgecolor=ps.get("hatch_edgecolor"),
            )

    if not shapes is None:
        ax.add_geometries(shapes, crs, **ps.get("shape_params"))

    if "land" in ps.get("cartopy_feats"):
        ax.add_feature(
                cfeature.LAND,
                #linestyle=ps.get("border_style", "-"),
                #linewidth=ps.get("border_linewidth", 2),
                #edgecolor=ps.get("border_color", "black"),
                )
    if "borders" in ps.get("cartopy_feats"):
        ax.add_feature(
                cfeature.BORDERS,
                linestyle=ps.get("border_style", "-"),
                linewidth=ps.get("border_linewidth", 2),
                edgecolor=ps.get("border_color", "black"),
                )
    if "states" in ps.get("cartopy_feats"):
        ax.add_feature(
                cfeature.STATES,
                linestyle=ps.get("border_style", "-"),
                linewidth=ps.get("border_linewidth", 2),
                edgecolor=ps.get("border_color", "black"),
                )

    cbar = fig.colorbar(
            scat,
            ax=ax,
            shrink=ps.get("cbar_shrink"),
            label=ps.get("cbar_label"),
            orientation=ps.get("cbar_orient", "vertical"),
            pad=ps.get("cbar_pad", 0.0),
            norm=ps.get("norm"),
            extendfrac=ps.get("cbar_extendfrac"),
            spacing=ps.get("cbar_spacing", "uniform"),
            extend=ps.get("cbar_extend", "both"),
            )
    cbar.ax.tick_params(labelsize=ps.get("fontsize_labels", 10))
    scat.figure.axes[0].tick_params(
            axis="both", labelsize=ps.get("fontsize_labels",10))

    if not fig_path is None:
        if not ps.get("figsize") is None:
            fig.set_size_inches(*ps.get("figsize"))
        fig.savefig(fig_path.as_posix(), bbox_inches="tight",dpi=ps.get("dpi"))
    if show:
        plt.show()
    plt.close()

def plot_geo_tiles(data, lat, lon, shapes=None,
        bounds=None, plot_spec={}, show=False,
        fig_path=None):
    """
    given 1d data, lat, and lon arrays, use tricontourf delauny tessellation
    to create a smooth plot
    """
    ps = {
        "xlabel":"", "ylabel":"", "marker_size":4, "cmap":"jet_r", "dpi":200,
        "text_size":12, "title":"", "norm":"linear","figsize":None,
        "marker":"o", "cbar_shrink":1., "map_linewidth":2,
        "shape_params":{"edgecolor":"black"},
        "cartopy_feats":["land", "borders", "states"],
        "cbar_extendfrac":0.05, "custom_cmap_params":None,
        "proj_in":"plate_carree", "proj_in_args":{},
        "proj_out":"plate_carree", "proj_out_args":{},
        }
    plt.clf()
    ps.update(plot_spec)
    plt.rcParams.update({"font.size":ps["text_size"]})

    crs_type = {
        "plate_carree":ccrs.PlateCarree,
        "geostationary":ccrs.Geostationary,
        }
    ## for geostationary, need central_longitude and satellite_height args
    crs_in = crs_type[ps.get("proj_in")](**ps.get("proj_in_args"))
    crs_out = crs_type[ps.get("proj_out")](**ps.get("proj_out_args"))
    ax = plt.axes(projection=crs_out)
    fig = plt.gcf()
    #if bounds is None:
    #    bounds = [np.amin(lon), np.amax(lon),
    #              np.amin(lat), np.amax(lat)]
    #ax.set_extent(bounds, crs=crs)

    ax.add_feature(cfeature.LAND, linewidth=ps.get("map_linewidth"))

    ax.set_title(ps.get("title"), fontsize=ps.get("fontsize_title", 12))
    ax.set_xlabel(ps.get("xlabel"), fontsize=ps.get("fontsize_labels", 10))
    ax.set_ylabel(ps.get("ylabel"), fontsize=ps.get("fontsize_labels", 10))

    '''
    cmap_params = {}
    if not ps.get("custom_cmap_params") is None:
        ccp = ps["custom_cmap_params"]
        cmap = matplotlib.colors.ListedColormap(ccp["colors"])
        if "extremes" in ccp.keys():
            assert isinstance(ccp["extremes"], (list,tuple))
            exlow,exhigh = ccp["extremes"]
            #cmap = cmap.with_extremes(under=exlow, over=exhigh)
            cmap.set_over(exhigh)
            cmap.set_under(exlow)
        norm = matplotlib.colors.BoundaryNorm(ccp["bounds"], cmap.N)
        cmap_params = {"cmap":cmap, "norm":norm}
    else:
        cmap_params = {
            "cmap":ps.get("cmap"),
            "norm":ps.get("norm"),
            "vmin":ps.get("vmin"),
            "vmax":ps.get("vmax"),
            }
    '''

    proj_pts = crs_out.transform_points(crs_in, lon, lat)

    tiles = ax.tricontourf(
            proj_pts[...,0],
            proj_pts[...,1],
            data,
            cmap=ps.get("cmap"),
            norm=ps.get("norm", "linear"),
            vmin=ps.get("vmin"),
            vmax=ps.get("vmax"),
            levels=ps.get("levels", 32),
            zorder=1
            )

    if not shapes is None:
        ax.add_geometries(shapes, crs_out, **ps.get("shape_params"))

    if "land" in ps.get("cartopy_feats"):
        ax.add_feature(
                cfeature.LAND,
                zorder=3,
                )
    if "borders" in ps.get("cartopy_feats"):
        ax.add_feature(
                cfeature.BORDERS,
                linestyle=ps.get("border_style", "-"),
                linewidth=ps.get("border_linewidth", 2),
                edgecolor=ps.get("border_color", "black"),
                zorder=3,
                )
    if "states" in ps.get("cartopy_feats"):
        ax.add_feature(
                cfeature.STATES,
                linestyle=ps.get("border_style", "-"),
                linewidth=ps.get("border_linewidth", 2),
                edgecolor=ps.get("border_color", "black"),
                zorder=3,
                )

    cbar = fig.colorbar(
            tiles,
            ax=ax,
            shrink=ps.get("cbar_shrink"),
            label=ps.get("cbar_label"),
            orientation=ps.get("cbar_orient", "vertical"),
            pad=ps.get("cbar_pad", 0.0),
            norm=ps.get("norm"),
            extendfrac=ps.get("cbar_extendfrac"),
            spacing=ps.get("cbar_spacing", "uniform"),
            extend=ps.get("cbar_extend", "both"),
            )
    cbar.ax.tick_params(labelsize=ps.get("fontsize_labels", 10))
    tiles.figure.axes[0].tick_params(
            axis="both", labelsize=ps.get("fontsize_labels",10))

    if not fig_path is None:
        if not ps.get("figsize") is None:
            fig.set_size_inches(*ps.get("figsize"))
        fig.savefig(fig_path.as_posix(), bbox_inches="tight",dpi=ps.get("dpi"))
    if show:
        plt.show()
    plt.close()
