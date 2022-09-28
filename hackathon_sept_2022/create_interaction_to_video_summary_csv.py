import sys
import os
import glob
import pandas as pd

# INSTRUCTIONS: 2 args.  <in directory with csvs to summarize> <out file name to place summary csv in>

indir=sys.argv[1]
outfile=sys.argv[2]

print(indir)

file_names_to_interaction_column = dict()

for infile in glob.glob(os.path.join(indir, '*.csv')):
    print(infile)
    df = pd.read_csv(infile, index_col=0)
    # print(df)
    interactions = df['Interaction']
    file_names_to_interaction_column[os.path.basename(infile)] = interactions

summary_df = pd.DataFrame(file_names_to_interaction_column)
summary_df.to_csv(outfile, index=False)

