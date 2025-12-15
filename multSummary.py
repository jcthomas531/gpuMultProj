import os
os.chdir("H:\\schoolFiles\\bios7330AdvComputing\\gpuMultProj")
from plotnine import *
import pandas as pd
import numpy as np
import joblib
timeHolder = joblib.load("timeHolder.joblib")
ttBit = joblib.load("ttbit.joblib")
tt10 = joblib.load("tt10.joblib")

##################time with growing dimension

#pivot longer
valueCols = [col for col in timeHolder.columns if col.endswith("Time")]
timeHolderLong = pd.melt(timeHolder,
                         id_vars=["dims"], #columns to keep
                         value_vars=valueCols, #columns to pivot
                         var_name = "method", #name for the pivoted columns
                         value_name = "time" #name for the values
                         )
#create variable for the log of time
timeHolderLong["logTime"] = np.log(timeHolderLong["time"] + .0001)

#plots for time holder
(
 ggplot(timeHolderLong, aes(x="dims", y="time", color = "method")) 
+ geom_line() 
+ theme_bw() 
+ scale_color_manual(values = {"npTime": "black",
                               "ncTime":"red",
                               "ngTime":"black",
                               "tiTime":"black",
                               "cuTime":"black"}, 
                     labels = {"npTime": "numpy",
                                                    "ncTime":"Naive CPU",
                                                    "ngTime":"Naive GPU",
                                                    "tiTime":"Tiled GPU",
                                                    "cuTime":"cupy"})
+ labs(title = "Multiplication Time with Increasing Dimension",
       y = "Seconds",
       x = "Dimensions",
       color = "Method")
)




#remove niave cpu time
thl2 = timeHolderLong.loc[timeHolderLong["method"] != "ncTime"]

(
 ggplot(thl2, aes(x="dims", y="time", color = "method")) 
+ geom_line() 
+ theme_bw() 
+ scale_color_manual(values = {"npTime": "red",
                               "ngTime":"green",
                               "tiTime":"blue",
                               "cuTime":"black"}, 
                     labels = {"npTime": "numpy",
                               "ngTime":"Naive GPU",
                               "tiTime":"Tiled GPU",
                               "cuTime":"cupy"})
+ labs(title = "Multiplication Time with Increasing Dimension",
       y = "Seconds",
       x = "Dimensions",
       color = "Method")
)



#######################################varying thread count
#pivot longer
valueColsBit = [col for col in ttBit.columns if col.endswith("Time")]
ttBitLong = pd.melt(ttBit,
                         id_vars=["threads"], #columns to keep
                         value_vars=valueColsBit, #columns to pivot
                         var_name = "method", #name for the pivoted columns
                         value_name = "time" #name for the values
                         )
ttBitLong["sim"] = "bit"
valueCols10 = [col for col in tt10.columns if col.endswith("Time")]
tt10Long = pd.melt(tt10,
                         id_vars=["threads"], #columns to keep
                         value_vars=valueCols10, #columns to pivot
                         var_name = "method", #name for the pivoted columns
                         value_name = "time" #name for the values
                         )
tt10Long["sim"] = "10"

#bit plot
(
 ggplot(ttBitLong, aes(x="threads", y="time", color = "method")) 
+ geom_line() 
+ theme_bw() 
+ scale_color_manual(values = {"ngTime":"green",
                               "tiTime":"blue",
                               "cuTime":"black"}, 
                     labels = {"ngTime":"Naive GPU",
                               "tiTime":"Tiled GPU",
                               "cuTime":"cupy"})
+ labs(title = "Multiplication Time with Increasing Threads (2 powers)",
       y = "Seconds",
       x = "Threads",
       color = "Method")
)

#10 plot
(
 ggplot(tt10Long, aes(x="threads", y="time", color = "method")) 
+ geom_line() 
+ theme_bw() 
+ scale_color_manual(values = {"ngTime":"green",
                               "tiTime":"blue",
                               "cuTime":"black"}, 
                     labels = {"ngTime":"Naive GPU",
                               "tiTime":"Tiled GPU",
                               "cuTime":"cupy"})
+ labs(title = "Multiplication Time with Increasing Threads (non 2 powers)",
       y = "Seconds",
       x = "Threads",
       color = "Method")
)




#together
ttLong = pd.concat([ttBitLong, tt10Long], axis = 0)
(
 ggplot(ttLong, aes(x="threads", y="time", color = "method", linetype = "sim")) 
+ geom_line() 
+ theme_bw() 
+ scale_color_manual(values = {"ngTime":"green",
                               "tiTime":"blue",
                               "cuTime":"black"}, 
                     labels = {"ngTime":"Naive GPU",
                               "tiTime":"Tiled GPU",
                               "cuTime":"cupy"})
+ labs(title = "Multiplication Time with Increasing Threads",
       y = "Seconds",
       x = "Threads",
       color = "Method")
)


