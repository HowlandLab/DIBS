import sys
import os
import glob
import pandas as pd

# INSTRUCTIONS: 3 args.  <good interactions directory> <good everything else directory> <output directory>
# The output directory must not exist before running the script, we will create the new csvs in that directory to avoid data destruction.  You may move the new csvs whereever you need them afterwards


if len(sys.argv) != 4:
    raise RuntimeError('Something went wrong with input arguments, if you have read the instructions (above, in the source code), then try putting quotes around the input arguments (quotes around each separately)')

good_interactions_indir=sys.argv[1]
good_everything_else_indir=sys.argv[2]
outdir=sys.argv[3]

if os.path.exists(outdir):
    raise RuntimeError('The outdir already exists!! We will not overwrite any data!  You must give a path to a directory that does not exist that we can create!')

good_everything_else_files = {os.path.basename(f):f for f in glob.glob(os.path.join(good_everything_else_indir, '*.csv'))}

good_interaction_files = {os.path.basename(f):f for f in glob.glob(os.path.join(good_interactions_indir, '*.csv'))}

diff_of_files=set(good_interaction_files.keys()).symmetric_difference(set(good_everything_else_files.keys()))
if diff_of_files:
    print(f'WARNING::::::::: The following files can not be matched and output will not be produced for them!!!! {diff_of_files}')

os.mkdir(outdir) # If this gives you an error give a different output directory name!!

for basename, infile in good_interaction_files.items():
    print(basename, infile)
    df = pd.read_csv(infile, index_col=0)
    good_interactions = df['Interaction']
    other_in_file = good_everything_else_files.get(basename)
    if other_in_file:
        good_df = pd.read_csv(other_in_file, index_col=0)
        good_df['Interaction'] = good_interactions
        print('good_df:', good_df)
        good_df.to_csv(os.path.join(outdir, basename))
    else:
        print(f'WARNING!!!!!!!!!!!!!!!!!!! Not producing output file for good interactions in {infile}; See other WARNINGs above')
    
