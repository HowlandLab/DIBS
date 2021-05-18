#!/usr/bin/env python

import sys
import pandas as pd
from dibs.io import read_pipeline

# d = 'AARON_temp_output' # not needed when arg has dir

def main(infile, write_csv=False):
    if '.pipeline' not in infile:
        print(f'infile must be a *.pipeline file.  Got \'{infile}\' instead.')
        sys.exit(1)

    p = read_pipeline(infile)

    df = p._df_features_train_scaled_train_split_only

    t = df[['dim_1', 'dim_2', 'gmm_assignment']]
    t2 = t[~t.dim_1.isnull()]
    if write_csv:
        t2.to_csv(f'{infile}_tsne_embedding_with_gmm_assignment.csv')
    return t2


if __name__ == '__main__':
    infile = sys.argv[1]
    main(infile, write_csv=True)

