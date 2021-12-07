import os
from dibs import io
from dibs.base_pipeline import BasePipeline
from dibs.pipeline_pieces import TSNE, UMAP, SPECTRAL, HDBSCAN, BayesianGMM, GMM
from dibs import config

from matplotlib import pyplot as plt


# Instantiate a pipeline, load data, build it, create a clustering plot

algo_pairs = [
    (UMAP(),
        [
            HDBSCAN(),
            BayesianGMM(),
        ]
     ),
    # (TSNE(), SPECTRAL()),# Spectral is way to expensive.  Would have to implement sparse algo first
    # (TSNE(), HDBSCAN()),
]

data_name_splits = config.DEFAULT_TRAIN_DATA_DIR.split('/')
data_name = '_'.join(data_name_splits[2:])

p = BasePipeline(data_name)
## Add data based on config
## 1F, 15B, 1B, 15F, are good quality videos for air_control (nose data is good)
#good_animals = ['1F', '15B', '1B', '15F']
#good_csv_files = []
#
#for name in os.listdir(config.DEFAULT_TRAIN_DATA_DIR):
#    if any(animal in name for animal in good_animals):
#        good_csv_files.append(
#            os.path.join(config.DEFAULT_TRAIN_DATA_DIR, name)
#        )
#
#print(f'Good videos of good animals: {good_csv_files}')
#p = p.add_train_data_source(*good_csv_files)

p = p.add_train_data_source(config.DEFAULT_TRAIN_DATA_DIR)

EXPERIMENT_NAME = 'with_our_imputation_all_data_with_likelihood'

for embed, clusts in algo_pairs:
    embed_name = embed.__class__.__name__
    p._embedder = embed
    p._embedder_is_built = False

    for clust in clusts:
        clust_name = clust.__class__.__name__
        p._clusterer = clust
        p._clusterer_is_built = False

        p.build()
        fig, ax = p.plot_clusters_by_assignments(f'{p.name}-{embed_name}-{clust_name}')

        fig.savefig(f'{EXPERIMENT_NAME}_{data_name}_{embed_name}_{clust_name}.png')

    # TODO: Save to disk?

