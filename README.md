# goes-land-spectra

Acquire goes data and calculate cloud-free bulk statistics and
histograms for each pixel in a subset of bands.

## planning

### goals

- histograms per pixel/month at several times per day
- min, max, mean, stddev per month
- detect and log anomalies, including coordinate misailgnment

### datasets

**L1b bands**

| band | λ(um) | res(km) | name     |
| ---- | ----- | ------- | -------- |
|   1  |  0.47 |   1.0   | blue     |
|   2  |  0.64 |   0.5   | red      |
|   3  |  0.86 |   1.0   | veggie   |
|   5  |  1.61 |   1.0   | NIR 1    |
|   6  |  2.24 |   2.0   | NIR 2    |
|   7  |  3.9  |   2.0   | SWIR     |
|  13  | 10.3  |   2.0   | Clean LW |
|  15  | 12.3  |   2.0   | Dirty LW |

**ancillary**



### platforms

GOES-16 (East): 20170228 - 20250407
GOES-18 (West): 20220512 - present

### sizes

**histograms**

For each band: (P,M,T,B)

P: pixels
M: months
T: times
B: bins

times ought to be fixed. For example, 12, 15, 18, 21, 0 UTC for
shortwave, with 3, 6, 9 added for longwave.

**bulk stats**

(P, M, T, F)

min, max, mean, median, count

## algorithm

should be appropriate to process 1 day at a time...

1. download full-resolution L1b CONUS files for each of the bands,
   as well as the clear sky mask.
2. check the coordinate grid against existing stored grids. If none
   of them are univerally close, make and store a new one.
3.
