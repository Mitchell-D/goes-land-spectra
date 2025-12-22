# goes-land-spectra

Acquire goes data and calculate cloud-free bulk statistics and
histograms for each pixel in a subset of bands.

## planning

### goals

**first pass**

- min, max, mean, stddev per month
- 3rd or 4th moment (skewness / kurtosis)
- general covariance structure?
- detect and log coordinate misalignment, unexpected nan values.
- basic histogram??

**second pass**

- distribution drift over time
- multivariate correlation structure of anomalies
  - Track instances of one or more bands having n-sigma anomaly;
    try to segment instances into groups and explain.
- histograms per pixel/month at several times per day (?)
  this is pretty expensive. Maybe focus on narrower problem.

**research topics**

-
- investigate the homogeneity of LST anomaly. Use it to build a regression
  model for clear-sky temperature fluctuation. (preferably without severely
  discretizing sfc classes).
  - calibrate to SCAN ground stations?
- classify out-of-distribution events over different regions
- sub-pixel cloud characterization from multi-satellite observations

**not gonna work (yet)**

- quantile estimation is difficult... seems like for single-pass
  estimates, using a reservoir method is recommended, which doesn't
  seem reasonable for data as stratified as mine.
- TDigest object (quantiles, cdf, trimmed mean)
  - NOTE: TDigest tracks a single random variable at a time. This
    may be useful during training or something, not for each p/t/m.

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

**auxiliary**

Cloud mask (L2 ACMC)



### platforms

**L1B data available on AWS** (at least intermittently)

- GOES-16 (East): 20170228 - 20250407
- GOES-17 (West): 20180828 - 20230110
- GOES-18 (West): 20220512 - present
- GOES-19 (East): 20241010 - present

**official orbital transition plans**

[GOES-East](https://www.goes-r.gov/downloads/users/transitionToOperations19/GOES-19%20T2O%20Schedule%2020240617.pdf)

- GOES-16 nudge 75.2 - 75.5, March 17 to April 1, 2025
- GOES-19 drift 89.5 - 75.2, March 17 to April 1, 2025
- GOES-16 drift 75.5 - 105,  April 4 to June 4, 2025

[GOES-West](https://www.ospo.noaa.gov/operations/goes/documents/GOES-18%20T2O%20Schedule%2020220927.pdf)

- GOES-18 nudge  89.5 - 136.8 May 16 to June 6, 2022
- GOES-17 nudge 137.2 - 137.3 July 5 to July 10, 2022
- GOES-18 nudge 136.8 - 137.0 July 5 to July 21, 2022

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
