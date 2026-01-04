# bulk stat vector embeddings

## how to make them...

contrastive autoencoder with alignment / uniformity loss as in the
paper [Understanding Contrastive Representation Learning through
Alignment and Uniformity on the Hypersphere][1] ([code here][2]).

## things to do with them...

- cluster latent representation to form surface classification.

- given embedding and current observation, predict the mean and
  standard deviation of NDVI,NDWI for this pixel in January.

[1]:https://proceedings.mlr.press/v119/wang20k/wang20k.pdf
[2]:https://github.com/ssnl/align_uniform/blob/master/examples/stl10/main.py
