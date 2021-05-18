#!/usr/bin/env python

import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import math as m
import extract_tsne_embedding

infile = sys.argv[1]
if '.csv' in infile:
    df = pd.read_csv(infile)
elif '.pipeline' in infile:
    df = extract_tsne_embedding.main(infile)
else:
    print(f'expected *.pipeline or *.csv file.  Got \'{infile}\' instead.')

n = len(df.gmm_assignment.unique()) # number of clusters

mat = np.zeros((n, n)) # mat to be filled with assignment transition counts

gmm_a = df.gmm_assignment

for b1,b2 in zip(gmm_a[0:-2], gmm_a[1:-1]):
    mat[b1,b2] += 1


print(mat)
log_mat = np.log(mat + 1)
print(log_mat)
norm_mat = mat / np.max(mat)
print(norm_mat)

int_mat_zero_diag = np.array(mat, dtype='int') # diagonal to zero
np.fill_diagonal(int_mat_zero_diag, 0)

# set seaborn figure size in inches
sns.set(rc={'figure.figsize':(11.7,8.27)})
cmap=sns.color_palette("Blues", as_cmap=True)

def heatmap(m, name, **kwargs):
    plt.clf()
    hm = sns.heatmap(m, cmap=cmap, **kwargs)
    fig = hm.get_figure()
    fig.savefig(f'{name}.png')



## Commented these out because they were not terribly useful
# heatmap(log_mat, 'log_mat')
# heatmap(norm_mat, 'norm_mat') # looks weird

# norm_zero_diag_mat = int_mat_zero_diag / np.max(int_mat_zero_diag)
# print(norm_zero_diag_mat)

# heatmap(norm_zero_diag_mat, 'norm_zero_diag_mat',
#         annot=True, fmt='.2f')

# This was the most useful
heatmap(int_mat_zero_diag, 'mat',
        annot=True, fmt='d')

plt.show() # only last plot can be shown when running from command line (seems that way at least)
plt.clf()  # clear


