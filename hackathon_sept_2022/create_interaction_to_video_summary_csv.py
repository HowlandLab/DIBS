import sys
import os
import glob
import pandas as pd

# INSTRUCTIONS: 2 args.  <in directory with csvs to summarize> <out file name to place summary csv in>

if len(sys.argv) != 3:
    raise RuntimeError('There is an error with the input arguments.  Expect 2 arguments, 1st: Input directory with csvs to summaryize.  2nd: Path to an output file where the results will be stored, must not exist.  Try adding quotes around the input arguments (each individually)')

indir=sys.argv[1]
outfile=sys.argv[2]


if os.path.exists(outfile):
    raise RuntimeError('The outfile already exists!  We will not overwrite any files!  Please provide a path that does not already exist!')


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
