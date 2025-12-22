import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
from datetime import datetime,timedelta
from cartopy.mpl.ticker import LatitudeFormatter,LongitudeFormatter
from matplotlib.colors import ListedColormap

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
    geo_bounds=None, latlon_ticks=True,
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
        geo_bounds = [np.amin(lon), np.amax(lon), np.amin(lat), np.amax(lat)]
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

def plot_geo_scalar(data, latitude, longitude, hatch_data=None, shapes=None,
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
        }
    plt.clf()
    ps.update(plot_spec)
    plt.rcParams.update({"font.size":ps["text_size"]})

    ax = plt.axes(projection=ccrs.PlateCarree())
    fig = plt.gcf()
    if bounds is None:
        bounds = [np.amin(longitude), np.amax(longitude),
                  np.amin(latitude), np.amax(latitude)]
    ax.set_extent(bounds, crs=ccrs.PlateCarree())

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
        scat = ax.contourf(longitude, latitude, data, **cmap_params)
    else:
        scat = ax.pcolormesh(longitude, latitude, data, **cmap_params)

    if latlon_ticks:
        lonmin,lonmax,latmin,latmax = bounds
        frq = ps.get("tick_frequency", 1)
        ax.set_yticks(np.linspace(latmin,latmax,data.shape[0])[::frq],
                crs=ccrs.PlateCarree())
        ax.set_xticks(np.linspace(lonmin,lonmax,data.shape[1])[::frq],
                crs=ccrs.PlateCarree())
        lon_formatter = LongitudeFormatter(zero_direction_label=True)
        lat_formatter = LatitudeFormatter()
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.yaxis.set_major_formatter(lat_formatter)
        ax.tick_params(rotation=ps.get("tick_rotation", 0))

    if not hatch_data is None:
        hatch_plot = ax.contourf(
            longitude,
            latitude,
            hatch_data.astype(int),
            levels=[0.5,1.5],
            hatches=ps.get("hatch_style"),
            #linewidth=ps.get("hatch_linewidth"),
            colors="none",
            #edgecolor=ps.get("hatch_edgecolor"),
            )

        '''
        ax.pcolormesh(
            longitude,
            latitude,
            hatch_data,
            shading=ps.get("shading", "auto"),
            facecolor=ps.get("hatch_facecolor"),
            edgecolor=ps.get("hatch_edgecolor"),
            hatch=ps.get("hatch_style"),
            linewidth=ps.get("hatch_linewidth"),
            )
        '''

    if not shapes is None:
        ax.add_geometries(shapes, ccrs.PlateCarree(), **ps.get("shape_params"))

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

def mp_plot_binary_smvi(args):
    return plot_binary_smvi(**args)
def plot_binary_smvi(
        int_data, lat, lon, fstr, fig_path, polys, date, smvi_thresh):
    """
    """
    tstr = date.strftime("%Y%m%d")
    tstr2 = date.strftime("%Y-%m-%d")
    plot_geo_ints(
        int_data=int_data,
        lat=lat,
        lon=lon,
        int_labels=[
            "Out of Domain",
            f"SMVI Fraction <= {smvi_thresh}",
            f"SMVI Fraction > {smvi_thresh}",
            ],
        fig_path=fig_path,
        latlon_ticks=False,
        shapes=polys,
        cbar_ticks=True,
        plot_spec={
            "cbar_pad":0.02,
            "cbar_orient":"horizontal",
            "cbar_shrink":.8,
            "cbar_fontsize":14,
            "tick_frequency":12,
            "tick_rotation":45,
            "title":f"Counties with >{smvi_thresh*100}% SMVI" + \
                    f" {fstr} ({tstr2})",
            "tile_fontsize":18,
            "interpolation":"none",
            "shape_params":{
                "edgecolor":"silver",
                "facecolor":"none",
                "alpha":.85,
                },
            },
        colors=["#3D74B6", "#FBF5DE", "#DC3C22"],
        )
    return fig_path

def mp_plot_percentile_and_smvi(args):
    return plot_percentile_and_smvi(**args)
def plot_percentile_and_smvi(
        percentile_data, smvi_data, lat, lon, fstr, fig_path, polys, date,
        smvi_thresh):
    """
    """
    tstr = date.strftime("%Y%m%d")
    tstr2 = date.strftime("%Y-%m-%d")
    plot_geo_scalar(
        data=percentile_data,
        latitude=lat,
        longitude=lon,
        hatch_data=smvi_data,
        shapes=polys,
        latlon_ticks=False,
        show=False,
        fig_path=fig_path,
        plot_spec={
            "title":f"{fstr} percentile, hatched " + \
                    f"SMVI>{smvi_thresh*100}% ({tstr2})",
            "cbar_shrink":.9,
            "cbar_spacing":"proportional",
            "cbar_extend":"both",
            "cbar_orient":"horizontal",
            "cbar_pad":.05,
            "hatch_shading":"auto",
            "hatch_edgecolor":"none",
            "hatch_style":["xxx"],
            "hatch_facecolor":"none",
            "border_linewidth":1.2,
            "fontsize_labels":8,
            "custom_cmap_params":{
                "colors":[
                    [0.4196, 0.0, 0.0], ## 2-5
                    [0.9216, 0.0, 0.0], ## 5-10
                    [0.9216, 0.4588, 0.0], ## 10-20
                    [1.0, 0.702, 0.4], ## 20-30
                    [0.7843, 0.7843, 0.7843], ## 30-70
                    [0.5882, 0.8235, 0.9804], ## 70-80
                    [0.3137, 0.6471, 0.9608], ## 80-90
                    [0.1569, 0.5098, 0.9412], ## 90-95
                    [0.0784, 0.3922, 0.8235], ## 95-98
                    #"#C52104", ## 2-5
                    #"#FA5B0F", ## 5-10
                    #"#F28705", ## 10-20
                    #"#F2B807", ## 20-30
                    #"#FEF7CC", ## 30-50
                    #"#CCD3FE", ## 50-70
                    #"#E3E1E1", ## 30-70
                    #"#2998FF", ## 70-80
                    #"#0068C4", ## 80-90
                    #"#004B8D", ## 90-95
                    #"#00294D", ## 95-98
                    ],
                "bounds":[2,5,10,20,30,70,80,90,95,98],
                "extremes":("#710301", "#082136"),
                },
            "shape_params":{
                "edgecolor":"black",
                "linewidth":.5,
                "facecolor":"none",
                "alpha":.8,
                },
            "dpi":120,
            },
        )
    return fig_path
