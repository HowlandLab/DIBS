#!/usr/bin/env python3

from sklearn.manifold import TSNE as sk_tsne

import bhtsne

from openTSNE import TSNE
from openTSNE.affinity import PerplexityBasedNN
from openTSNE import initialization
from openTSNE import TSNEEmbedding
import pandas as pd
import numpy as np



def run_sk_tsne(train_features):
  x_train = np.array(train_features)
  return sk_tsne(n_components=2).fit_transform(x_train)


def run_bh_tsne(train_features):
  x_train = np.array(train_features)
  # params: perplexity=30.0, theta=0.5
  return bhtsne.tsne(x_train, dimensions=3)


def run_openTSNE(train_features):
  x_train = np.array(train_features)

  n_components = 3
  tsne = TSNE(
	n_components=n_components, # https://github.com/pavlin-policar/openTSNE/issues/121
	negative_gradient_method='bh',
	perplexity=30,
	metric='euclidean',
	verbose=True,
	n_jobs=10,
	random_state=42
      )

  embedding = tsne.fit(x_train)

  # np.savetxt(f"tsne{n_components}dims.csv", embedding, delimiter=',', header=",".join([f'X{i}' for i in range(embedding.shape[1])]))
  return embedding

def run_openTSNE_advanced_embedding(train_features):
  pass
  # ## Advanced embedding. https://opentsne.readthedocs.io/en/latest/examples/02_advanced_usage/02_advanced_usage.html

  # affinities_train = PerplexityBasedNN(
  #       x_train,
  #       perplexity=30,
  #       metric='euclidean',
  #       n_jobs=10,
  #       random_state=42
  #     )

  # affinity_init = initialization.pca(x_train, random_state=42)

  # affinity_embedding = TSNEEmbedding(
  #       affinity_init,
  #       affinities_train,
  #       n_components=3, # NOTE: DOESN'T DO ANYTHING!!
  #       negative_gradient_method='bh',
  #       n_jobs=10,
  #       verbose=True
  #     )


  # np.savetxt("affinity_tsne.csv", affinity_embedding, delimiter=',', header=",".join([f'X{i}' for i in range(affinity_embedding.shape[1])]))








