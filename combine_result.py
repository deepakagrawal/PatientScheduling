import pandas as pd
import os, sys
import numpy as np
from glob import glob

# abspath to a folder as a string
folder = '/storage/work/dua143/PatSchedPang_v2/FairScheduling/gams_try/joint_result_tmp_dir_v7_2hor/'


files = []
start_dir = os.getcwd()
pattern   = "tmp*/D*.csv"

for dir,_,_ in os.walk(folder):
    files.extend(glob(os.path.join(dir, pattern)))

dat = pd.DataFrame()
for file in files:
    if(os.stat(file).st_size >0):
        dat = dat.append(pd.read_csv(file), ignore_index=True)

dat = dat.append(pd.read_csv(folder+"CP_SPtestPaperSingleOpt_detLam_result_gen_loc_all_constant_Oct7_hammerzl4bu7mo.csv"), ignore_index=True)
dat.to_csv(folder+"DCPP_DP_joint_results.csv", index = False)