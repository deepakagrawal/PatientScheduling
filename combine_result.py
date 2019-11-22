import pandas as pd
import dask.dataframe as dd
import os, sys
import numpy as np
from glob import glob
from itertools import chain

# abspath to a folder as a string
folder = '/storage/work/dua143/PatSchedPang_v2/FairScheduling/gams_try/new3_folder_larger_results/'


files = []
start_dir = os.getcwd()
patterns = ["tmp*/"+i for i in ["CP*.csv", "SP*.csv", "DP*.csv", "DCPP*.csv"]]


for dir,_,_ in os.walk(folder):
    files.extend([glob(os.path.join(dir, i)) for i in patterns])

files = list(chain.from_iterable(files))
files = [i for i in files if os.stat(i).st_size>0]

dat = pd.concat([pd.read_csv(file) for file in files], ignore_index=True)


# dat = dat.append(pd.read_csv(folder+"CP_SPtestPaperSingleOpt_detLam_result_gen_loc_all_constant_Oct7_hammerzl4bu7mo.csv"), ignore_index=True)
dat = dat.groupby(['Choice','Doubly','Horizon','Joint','Loc_Dep','Policy','Run','Theta'], as_index=False).max()
# dat.drop_duplicates(inplace=True, subset=, keep='')
dat = dat[~((dat.Policy.isin(["CP", "SP"])) & (dat.Joint==1))]
dat.to_csv(folder+"ALL_DP_DCPP_result.csv", index=False)