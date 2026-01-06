# project design notes

## better file storage concept

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

## file sizes

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

## spatial chunking and geometric assumptions

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

## clustering of final feature datta.

Once the data is easy to access across space and dimensional combos,
it will be interesting to dimensionally reduce it and see what
clusters emerge.

I'm considering masked autoencoder approaches that can take the
(P,F) where num feats F == months\*tods\*bands\*metrics + static.

chatgpt suggests that modeling the full conditional of every feature
may waste capacity on hard-to-predict features. contrastive learning
models are less succeptible to this.

Option A (very strong, simple):

1. Standardize features
2. PCA -> 200 dims
3. Denoising autoencoder
4. Latent dim: 32–128

Option B (often best):

1. Standardize
2. PCA -> 200 dims
3. Contrastive encoder with feature masking
4. Latent dim: 64

Option C (if no PCA):

1. Deep residual MLP encoder
2. Masked / denoising objective
3. Large batches

the goal here is to get a latent representation that robustly
captures the correlation structure of similar pixels by preserving
adjacency with the high-dimensional input.

If you want to go further:

- Replace MSE with Huber loss
- Add contrastive loss on z
- Train with multiple masked views
- Enforce latent decorrelation (VICReg-style)

contrastive models are related to energy-based models

## why does contrastive learning work? (Phillip Isola)

want to infer the *common cause* from co-ocurring samples.

contrastive loss is a lower bound on mutual information of embeddings

embeddings are deterministic functions of samples

so contrastive loss minimization maximizes mutual information
between samples.

In practice, this consists of masking a random subset of both
arrays, and rewarding both the reconstruction loss of the sample
as well as the proximity of the learned latent embedding.

Authors mathematically show that there is a sweet spot in the mutual
information between two unmasked "views" of a sample that containing
the best representation.

Even if the downstream task requiring semantic embeddings is not
known, a task generally requires an unknown number of bits of
mutual information in order to be sufficiently represented.
Views that share approximately this amount of mutual information.

The representation should be **minimal and sufficient**

From *Understanding Contrastive Representation Learning through
Alighnment and Uniformity on the Hypersphere*, (Wang & Isola, 2020)
define metrics for alignment and uniformity.

**alignment**: expected true pair feature distance. Measures distance
between embeddings.

**uniformity**: log expected gaussian potential of data pair. In
other words, the expected alignment between those embeddings.
Minimized when embeddings map to a gaussian distribution.

Mutually minimizing uniformally and maximizing alignment encourages
a semantically rich and gaussian-approximate latent embedding space.

Paper shows examples where pareto front between the two metrics

## supervised contrastive learning (yannic kilcher)

Traditionally, supervised cross-entropy uses one-hot encodings
and thus only rewards correct predictions discretely. Due to softmax
normalization to a probability distribution, other feature
probabilities get proportionally "pushed down" as optimization
proceeds, but only as a consequence of only explicitly optimizing one
output parameter at a time.

Contrastive learning can be used to learn semantically rich data
representations first so that downstream tasks become easier.
This is because semantically similar classes are nearby given
contrastive-learned embeddings.

Then, a small classifier is trained on top of the embeddings
using normal cross-entropy loss.

## umap background info

UMAP may be the best way to dimensionally reduce the combinations.

Nerve theorem implies that the data shape can be approximated by
connecting 1 & 2 simpleces within a radius. However, high-dimensional
data tends to have some very dense and some very sparse areas.

Instead, UMAP uses a variable radius determined by the definition
of a reimannian metric on the data manifold.

Density is considered high when the k'th nearest neighbor is close,
and far when the k'th nearest neighbor is far away. The k you use
is a parameter that favors global structure if big, or local
structure is small. No good general heuristic.

Typically, clustering either involves factorizing the data matrix, or
building a weighted network graph of the data.

For UMAP, assume the data is uniformly distributed on the manifold,
and define a riemannian metric that makes that assumption true.

## categorical deep learning (ML Street Talk youtube)

LLMs cannot intuit spatial, algorithmic, or algebraic reasoning
using only language. Doing so in general is thought to require
more structured and unique algebraic structures learned through
category theory.

## couterarguments to UMAP / contrastive learning approach

Using a point cloud approach gets rid of meaningful structure that
is known a-priori, for example temporal adjacency on the daily and
monthly scale, and flattening dimensions of (month,tod,band,metric)
obscures cross-axis correlations.

PCA preprocessing can lose meaningful structure when inputs have
different noise levels, and signals that are locally meaningful may
be destroyed in this global noise reduction.

**challenges**

UMAP & CAEs will struggle with data size. Furthermore, sample
imbalance and over-aggregatin of samples will make training
challenging.

Suggests using 'hierarchical sampling' to find a representative
subset of ~100k-200k samples for initial UMAP fitting

Parametric UMAP enables training neural network, which facilitates
batching and more efficient transformation of new data.

For contrastive learning, carfully consider how positive or negative
samples are defined (ie with spatial proximity, feature similarity).

**reframing**

Perhaps reframe the problem as *temporal dynamics embedding*
rather than just dimensionality reduction.

**structured embedding**

- use convolution to encode sub-pixel textures
  - keep in mind that sun/px/sat geometry is important here.
- use 1d temporal convolution along month/tod axes
- use spectral attention/group convolution on band dimension

**architecture**

- **heierarchical encoder** (preserve temporal structure longer)
  - beta-VAE encourages disentanglement of axes
- **transformer**
  - treat (band, subpixel, metric)

**composite model**

- independent embeddings for each (band, resolution) combo regardless
  of time of day or season.
- Subsequent learning of temporal/seasonal trends based on embedding

**negative mining**

Need proximal negative samples for similar textures but different
spectra. Combined with contrastive learning, the manifold will be
more isotropic and uniform at smaller scales.

**validation of results**

effective embeddings...

- map similar land surface types to similar locations
- reconstruct sub-pixel patterns
- should be stable across time for stable surfaces
- should vary wrt time for variable surfaces (ie vegetation)

## spectral/textural embeddings

- 2-stage training (random masks, then fixed masks)
- may need to train one resolution at a time in hierarchical fashion
- high masking ratios require significantly longer pre-training.
- consider combining reconstruction loss with Structural Similarity
  Index (SSIM), or Huber loss.

I still need to decide how to handle different metrics. Most likely
I should embed them separately (or only pair mean and stddev) since
skewness and kurtosis are likely less strong predictors and less
useful to reconstruct.

Nonetheless, coarser pixel kurtosis may be a good predictor for
nested higher-resolution textures.

When reconstructing, also consider that the pixel order is less
important and less feasible to estimate than the overall textural
and spectral qualities. Consider a loss that reflects this.

One way of addressing this issue is using wasserstein distance.
That is, sort both by intensity before applying the loss metric.

This is appropriate for a single band, but for 1km -> 2km, spectral
information is also important. In this consideration, inputs are
shaped (B, P, F), and pixels are a point cloud in Fd space.

- **Chamfer** distance is fastest (using `torch.cdist`), and
  calculates distance to closest pixel. Some predictions may map to
  the same pixel, however.

- **Sinkhorn** divergence uses differentiable approximation of the
  **Earth Mover's Distance**, encouraging 1:1 pixel matching at the
  cost of iterative structure.  `geomloss.SamplesLoss` has a good
  implementation.

- **Sliced wasserstein** reprojects both target and prediction
  spectra to 1d via a random but mutually identical matrix, then
  sorts and scores intensities after that.


## tensor decomposition

Instead, consider tensor decomposition / factorization methods,
which produce low-rank tensors preserving low-dimensional structure.

M = A B^T

M has shape (M,N), A has shape (M,r), B has shape (N,r)

The large matrix M is represented by decompositions A and B with
rank r.

Typically, specialized algorithms must be used to calculate
decompositions, and they impose tight constraints on factor tensors
(linear contractiveness, positive definiteness, orthogonality).

FunFact library uses einstein notation based functional
representation of (nonlinear) tensor expression (via abstract syntax
tree), and factorizes the data for you.

## from [this article][1]

Consider a tensor decomposition into "fibers and slices", that
is, vectors and matrices along axis combinations.



[1]:https://medium.com/@anishhilary97/cp-decomposition-approximating-tensors-using-collection-of-vectors-8db6c25f29ab




