import os
import sys
import dibs

path_to_pipe = sys.argv[1]
assert os.path.isfile(path_to_pipe), f'Path not found: {path_to_pipe}'

p: dibs.base_pipeline.BasePipeline = dibs.read_pipeline(path_to_pipe)
dir = os.path.dirname(path_to_pipe)
print(f"saving CSV to: {dir}")
out_path = os.path.join(dir, 'AARON_DATA.csv')
p.df_features_train.to_csv(out_path, index=None)


