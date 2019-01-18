import pandas as pd
import os, sys
import numpy as np
from glob import glob
from itertools import chain

# abspath to a folder as a string
folder = '/storage/work/dua143/PatSchedPang_v2/FairScheduling/gams_try/new_folder_larger_results/'


files = []
start_dir = os.getcwd()
patterns = ["tmp*/"+i for i in ["CP*.csv", "SP*.csv", "DP*.csv", "DCPP*.csv"]]


for dir,_,_ in os.walk(folder):
    files.extend([glob(os.path.join(dir, i)) for i in patterns])

files = list(chain.from_iterable(files))

dat = pd.DataFrame()
for file in files:
    if(os.stat(file).st_size >0):
        dat = dat.append(pd.read_csv(file), ignore_index=True)

# dat = dat.append(pd.read_csv(folder+"CP_SPtestPaperSingleOpt_detLam_result_gen_loc_all_constant_Oct7_hammerzl4bu7mo.csv"), ignore_index=True)
dat.to_csv(folder+"ALL_DP_DCPP_result.csv", index=False)