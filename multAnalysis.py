import os
os.chdir("H:\\schoolFiles\\bios7330AdvComputing\\gpuMultProj")
import multKernels as mk
import utilities as u
import pandas as pd
import cupy as cp
import numpy as np
from numba import cuda
import time
from numba import float32, int32, float64

###############################################################################
#comparing times for size
###############################################################################
# dims = (20, 100,200,400,800,1000,2000,4000,8000,10000)
dims = (20, 40)
timeHolder = pd.DataFrame({
    "dims": dims,
    "npTime": [np.nan]*len(dims),
    "ncTime": [np.nan]*len(dims),
    "ngTime": [np.nan]*len(dims),
    "tiTime": [np.nan]*len(dims)
    })

for i in range(timeHolder.shape[0]):
    dimi = timeHolder.loc[i,"dims"]
    Ai = np.random.normal(size = (dimi, dimi))
    Bi = np.random.normal(size = (dimi, dimi))
    resi = u.multTimes(Ai, Bi)
    timeHolder.loc[i,"npTime"] = resi["npTime"]
    timeHolder.loc[i,"ncTime"] = resi["ncTime"]
    timeHolder.loc[i,"ngTime"] = resi["ngTime"]
    timeHolder.loc[i,"tiTime"] = resi["tiTime"]
    print(timeHolder.iloc[i,:])
    print("dimension "+str(dimi)+" complete")


timeHolder
import joblib
import os
# os.chdir("H:\\schoolFiles\\bios7330AdvComputing\\gpuMultProj")
# joblib.dump(timeHolder, "2025_12_10_3 timeHolder.joblib")


from plotnine import *


valueCols = [col for col in timeHolder.columns if col.endswith("Time")]
timeHolderLong = pd.melt(timeHolder,
                         id_vars=["dims"], #columns to keep
                         value_vars=valueCols, #columns to pivot
                         var_name = "method", #name for the pivoted columns
                         value_name = "time" #name for the values
                         )

timeHolderLong["logTime"] = np.log(timeHolderLong["time"] + .0001)
ggplot(timeHolderLong, aes(x="dims", y="logTime", color = "method")) + geom_line()




###############################################################################
#comparing times for thread counts
###############################################################################



#start here tomorrow

threads = (2,5,10,20)
ttHolder = pd.DataFrame({
    "threads": threads,
    "ngTime": [np.nan]*len(threads),
    "tiTime": [np.nan]*len(threads)
    })

for i in range(ttHolder.shape[0]):
    threadi = ttHolder.loc[i,"threads"]
    resi = u.threadTimes(threadi)
    timeHolder.loc[i,"ngTime"] = resi["ngTime"]
    timeHolder.loc[i,"tiTime"] = resi["tiTime"]
    print(timeHolder.iloc[i,:])
    print("dimension "+str(dimi)+" complete")

ttHolder





