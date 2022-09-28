import sys
import os
import glob
import pandas as pd

indir=sys.argv[1]
outdir=sys.argv[2]

print(indir)

for infile in glob.glob(os.path.join(indir, '*.csv')):
    print(infile)
    df = pd.read_csv(infile, index_col=0)
    # print(df)
    print(df['Interaction'])
    print((df['Interaction'] == float('nan')).sum())

