import dask.dataframe as dd
import argparse
import glob
import os

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dir', help="Directory of the tmp folders")
parser.add_argument('-p', '--pattern', help="pattern of the tmp folders")
parser.add_argument('-f', '--file', help="pattern of the files")
parser.add_argument('-o', '--out', help="path of the output combined file")
args = parser.parse_args()

files = glob.glob(os.path.join(args.dir, f"{args.pattern}*", f"{args.file}*.csv"))

df = dd.read_csv(files).compute()
df.to_csv(args.out, index=None)
print(df.shapeawal)
