import dask.dataframe as dd
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', help="Directory of the tmp folders")
parser.add_argument('-o', '--out', help="path of the output combined file")
args = parser.parse_args()

df = dd.read_csv(args.path).compute()
df.to_csv(args.out, index=False)
