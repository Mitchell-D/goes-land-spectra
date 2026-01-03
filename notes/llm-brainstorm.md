# Brainstorming with LLMs

## prompt

Please help me brainstorm about ways I can use deep learning to gain
insights on a high-dimensional dataset that evolves temporally, and
has hidden parameters that govern the temporal dynamics.

In addition to time series that show the real evolution of the data
state, I have the first 4 statistical moments of each parameter
configuration to serve as supplementary data. I would like to explore
methods of...

1. clustering pixels that have similar hidden parameters
   (and thereby dynamics)
2. finding a numerical representation of the hidden parameters
3. forecasting the time series of a pixel given an initial state and
   parameter configuration.

If you have any ideas on other analysis approaches that could provide
useful information about the dataset, please mention them as well.


## chatgpt

Interprets the problem as "learning and interrogating a family of
hidden-parameter dynamical systems", and suggests *amortized
system identification*

Architecture options suggested:

1. **Encoder-decoder w/ static latent parameters**. That is, time
   series encoder is used to predict static parameters and an initial
   state.  Then, dynamics model uses generated static parameter,
   initial state, and covariates to generate results.
    - This does not guarantee parameter stability.

2. **NODE/NSDE w/ inferred parameters**. Advantage is that parameters
   have a clear mechanistic role, forecasting generalizes better,
   and stability/bifurcations/regimes are easier to analyze.
    - Neural SDEs & diffusion latent dynamics may be better for
      noisy systems.
    - Neural ODEs allow irregularly-spaced points in time. Learns a
      differential vector field such that numerically integrating
      along it produces the observations.

Recommends clustering w/ **k-means** after inferring parameters.

Otherwise, during training, **Deep Embedded Clustering** or
**Variational Deep Embedding** can be used to reward tighter
clusters and separation of qualitatively distinct dynamics.

Another option for clustering is **contrastive learning**, which
trains a classifier to identify positive pairs of time series when
they are from the same pixel.

For a numerical representation of the hidden parameters...

1. **Latent traversals/sensitivity analysis** entails varying one
   parameter at a time and simulating trajectories, observing
   qualitative changes in the results.
    - Look for control parameters, stability thresholds, etc.

2. **Koopman latent dynamics** Map to a latent space where a matrix
   determined by the parameters acts as the transition function when
   multiplied by the previous state.
    - the matrix's eigenvalues correspond to growth rates,
      frequencies, decay modes, etc.
    - can identify dynamical regimes, and useful for long-term fcst.
    - potentially physically interpretable.

3. **disentangled param discovery** use loss modifications to
   penalize high mutual information or correlation between params.
   Look into FactorVAE, beta-TCVAE

For forecasating...

1. Diffusion models or probabilistic RNNs learn the conditional
   distribution of the next state given the previous state + params

2. Consider encouraging pixels with similar moments to have similar
   parameters using **metric learning loss** w/ moment similarity.

Further analysis...

1. Look for correlation between moments and observed parameters

2. Look for parameter changes of a single trajectory over time

## Neural ODEs (Steven Brunton)

Even if training data is sampled discretely and regularly, the
learned flow map acts as if it is integrating over the state at
intermediate time points.

The structure of a NODE is such that the network can learn
information about the hidden state that is not in the input data.
The loss must be computed for timesteps intermediate to observations.

Dynamical systems optimization with lagrange multipliers deal with
this problem by introducing a "lagrange multiplier":

a(t) = - d Loss / d x(t)

that satisfies:

( d a(t) / d t )  =  - a(t) ( d F / d x(t) )

This is an adjoint equation, which can be calculated using autodiff
during training.

Ultimately, the continuous-time representation allows you to use
structured integrators, and to better fit data that is irregularly
spaced.

You can also train a NODE on irregularly spaced data, then use it
in order to generate synthetic data for training more explainable
model types.

Lagrangian neural networks (Cranmer et al., 2020) conserve energy
