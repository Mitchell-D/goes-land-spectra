# project design notes

Currently, dumping results into the pkls is convenient since a single
(sat, listing, t0, tf, geom, month, tod, band) combo is easy to pull
and update. Ultimately, though, most of the research analysis
should be done from a storage type that allows sampling across
some of those axes.

Stored per projection res in geom pkl: res:(pixel,2),m\_domain

Stored in result pkls: metric:[(pixel,), metadata]

Makes sense to sample across: (month, tod, band)

Should remain independent: (sat, listing, t0, tf, geom)

So each hdf5 will be identified by (sat, listing, t0, tf, geom),
and will contain the following arrays:

geom -> (Yr,Xr,2) scan angles at each res r, proj metadata

m\_domain -> (Yr,Xr) boolean mask of valid domain

results -> (pixel, month, tod, band, metric) array

The results array should be chunked to allow efficient access.
Currently a single (sat, listing, t0, tf, geom) combination,
including all combinations of:

(pixels, months, times, bands (per native res), metrics)

(that is, (pixels\_at\_res, 12, 5, bands\_at\_res, 7))

...requires about **97 GB**, which breaks down into **48.3 GB** for
band 2 at 0.5km, **36.2 GB** for bands 1, 3, and 5 at 1km, and
**12.1 GB** for bands 6, 7, 13, and 15 at 2km.

```
[nav] In [20]: 5 * 9663694923 / 1e9 ## 0.5km bands (only 2)
Out[20]: 48.318474615

[ins] In [21]: 5 * 7247823201 / 1e9 ## 1km bands (1, 3, 5)
Out[21]: 36.239116005

[nav] In [22]: 5 * 2416010412 / 1e9 ## 1km bands (6, 7, 13, 15)
Out[22]: 12.08005206
```

I got these estimates using commands like:

```bash
ll -d data/results/* | grep goes16 | grep geom-goes-conus-0 | grep _0_  | grep -e C01 -e C05 -e C03 | cut -c 46-  | xargs du -bc
```

One challenge is that good spatial chunking is hard when the pixels
are stored in 1d, however the localized permutations I developed for
my emulator work (ie [emulate-era5-land.helpers.get\_permutation][1])
can aid in solving this.

I'm already operating under a few assumptions about how the geom
system works...

1. the lowest-resolution domain mask array (by area) defines the
   valid array of all higher resolutions, and all higher resolutions
   have an integer res\_fac such that their shape equals
   (Y\*res\_fac, X\*res\_fac).
    - That is, if the low-res array has shape, (Y, X), and a high-res
      array has shape (Y*res\_fac, X*res\_fac), then a VALID pixel
      at (a,b52) of the low-res array implies all pixels in
      `[ a * res\_fac, a * (res\_fac + 1) )` and
      `[ b * res\_fac, b * (res\_fac + 1) )` are also VALID.

2. geometries can be uniquely defined according to the subsatellite
   point and the semi major/minor axes. Angular geometry remains
   consistent.

3. The land mask extracted for a geometry using the very first LST
   product is representative of all other arrays with matching geoms.



float32 chunks of (1024, 3, 3, 3)

Consider adding a vza threshold, conus-only mask, etc to save space.
