import os
os.chdir("H:\\schoolFiles\\bios7330AdvComputing\\gpuMultProj")
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import multKernels as mk
import utilities as u
import pandas as pd
import cupy as cp
import numpy as np
from numba import cuda
import time
from numba import float32, int32, float64
import random
random.seed(826)
import joblib


###############################################################################
#comparing times for size
###############################################################################
# dims = (20, 100,200,400,800,1000,2000,4000,8000,10000)
dims = (128,256, 512, 1024, 2048, 4096, 8192)
timeHolder = pd.DataFrame({
    "dims": dims,
    "npTime": [np.nan]*len(dims),
    "ncTime": [np.nan]*len(dims),
    "ngTime": [np.nan]*len(dims),
    "tiTime": [np.nan]*len(dims),
    "cuTime": [np.nan]*len(dims)
    })

for i in range(timeHolder.shape[0]):
    dimi = timeHolder.loc[i,"dims"]
    Ai = np.random.normal(size = (dimi, dimi))
    Bi = np.random.normal(size = (dimi, dimi))
    resi = u.multTimes(Ai, Bi, threads=16)
    timeHolder.loc[i,"npTime"] = resi["npTime"]
    timeHolder.loc[i,"ncTime"] = resi["ncTime"]
    timeHolder.loc[i,"ngTime"] = resi["ngTime"]
    timeHolder.loc[i,"tiTime"] = resi["tiTime"]
    timeHolder.loc[i,"cuTime"] = resi["cuTime"]
    print(timeHolder.iloc[i,:])
    print("dimension "+str(dimi)+" complete")


#timeHolder

#import os
# os.chdir("H:\\schoolFiles\\bios7330AdvComputing\\gpuMultProj")
# joblib.dump(timeHolder, "2025_12_10_3 timeHolder.joblib")
joblib.dump(timeHolder, "timeHolder.joblib")

# from plotnine import *


# valueCols = [col for col in timeHolder.columns if col.endswith("Time")]
# timeHolderLong = pd.melt(timeHolder,
#                          id_vars=["dims"], #columns to keep
#                          value_vars=valueCols, #columns to pivot
#                          var_name = "method", #name for the pivoted columns
#                          value_name = "time" #name for the values
#                          )

# timeHolderLong["logTime"] = np.log(timeHolderLong["time"] + .0001)
# ggplot(timeHolderLong, aes(x="dims", y="logTime", color = "method")) + geom_line()




###############################################################################
#comparing times for thread counts
###############################################################################



#start here tomorrow

threads = (2,4,8,16,32,64,128,256,1024)
ttHolder = pd.DataFrame({
    "threads": threads,
    "ngTime": [np.nan]*len(threads),
    "tiTime": [np.nan]*len(threads),
    "cuTime": [np.nan]*len(threads)
    })


a1 = u.threadTimes(threads = 2, dim=2048)
a2 = u.threadTimes(threads = 4, dim=2048)
a3 = u.threadTimes(threads = 8, dim=2048)
a4 = u.threadTimes(threads = 16, dim=2048)
a5 = u.threadTimes(threads = 32, dim=2048) #cant go higher than that

ttBit = pd.DataFrame({
    "threads": (2,4,8,16,32),
    "ngTime": (list(a1.values())[0], 
               list(a2.values())[0],
               list(a3.values())[0],
               list(a4.values())[0],
               list(a5.values())[0]),
    "tiTime": (list(a1.values())[1], 
               list(a2.values())[1],
               list(a3.values())[1],
               list(a4.values())[1],
               list(a5.values())[1]),
    "cuTime": (list(a1.values())[2], 
               list(a2.values())[2],
               list(a3.values())[2],
               list(a4.values())[2],
               list(a5.values())[2])
        })

b0 = u.threadTimes(threads = 2, dim=2000)
b1 = u.threadTimes(threads = 5, dim=2000)
b2 = u.threadTimes(threads = 10, dim=2000)
b3 = u.threadTimes(threads = 16, dim=2000)
b4 = u.threadTimes(threads = 20, dim=2000)

tt10 = pd.DataFrame({
    "threads": (2,5,10,16,20),
    "ngTime": (list(b0.values())[0],
               list(b1.values())[0], 
               list(b2.values())[0],
               list(b3.values())[0],
               list(b4.values())[0]),
    "tiTime": (list(b0.values())[1],
               list(b1.values())[1], 
               list(b2.values())[1],
               list(b3.values())[1],
               list(b4.values())[1]),
    "cuTime": (list(b0.values())[2],
               list(b1.values())[2], 
               list(b2.values())[2],
               list(b3.values())[2],
               list(b4.values())[2])
        })






#for some reason this doesnt work, same issue as before about predifing tile size
#and here this is changing, just gonna do it manually bc it works that way
# for i in range(ttHolder.shape[0]):
#     threadi = ttHolder.loc[i,"threads"]
#     resi = u.threadTimes(threads = threadi, dim = 2048)
#     timeHolder.loc[i,"ngTime"] = resi["ngTime"]
#     timeHolder.loc[i,"tiTime"] = resi["tiTime"]
#     print(timeHolder.iloc[i,:])
#     print("dimension "+str(dimi)+" complete")





#os.chdir("H:\\schoolFiles\\bios7330AdvComputing\\gpuMultProj")
joblib.dump(ttBit, "ttBit.joblib")
joblib.dump(tt10, "tt10.joblib")






