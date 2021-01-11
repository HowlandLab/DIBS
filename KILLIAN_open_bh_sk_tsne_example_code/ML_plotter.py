#!/usr/bin/env python3

import visuals
import tsnestuff
import pandas as pd
import numpy as np
import sys

t_data = pd.read_csv("./aaron_data__features_data_SCALED.csv")
t_data = t_data.iloc[:,0:6]

if len(sys.argv) > 1:
  func = sys.argv[1] # openTSNE sk_tsne bh_tsne
else:
  func = "openTSNE"

if func == "openTSNE":
  tsne_embedding = tsnestuff.run_openTSNE(t_data)
elif func == "bh_tsne":
  tsne_embedding = tsnestuff.run_bh_tsne(t_data)
elif func == "sk_tsne":
  tsne_embedding = tsnestuff.run_sk_tsne(t_data)
else:
  print("Do not recognice func:", func)
  sys.exit(1)



# visuals.plot_GM_assignments_in_3d_new(tsne_embedding, labs)
visuals.plot_tsne_in_3d(tsne_embedding)

