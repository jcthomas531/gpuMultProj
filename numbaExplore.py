
import cupy as cp




#
import numpy as np
from numba import cuda
import time
cuda.detect()



###############################################################################
#moving things to and from the GPU
###############################################################################
#make array on local
arrayCpu = np.random.randint(0,12, size = (2000,2000))
#send array to gpu
arrayGpu = cuda.to_device(arrayCpu)
#aside
#can also use cupy objects that are on gpu with numba, and can interact with numba moved objects via cupy on the gpu
cp.asarray(arrayGpu)
#
#bring the data back to local
arrayGpu.copy_to_host()


###############################################################################
#niave cpu implimentation for comparison with gpu that is non-optimized like the numpy multiplication
###############################################################################


A = np.random.normal(size = (10, 20))
B = np.random.normal(size = (20, 30))

def naiveCpu(A, B):
    #define variables used more than once
    Adim = A.shape[0]
    Bdim = B.shape[1]
    #set up output matrix
    C = np.empty((Adim, Bdim))
    #loop thru
    for i in range(Adim):
        for j in range(Bdim):
            #mirroring set up for gpu
            sumHolder = 0
            for k in range(len(B[:,1])): #this could also be len(A[1,:]), they are necessarily equal
                sumHolder += A[i,k] * B[k,j]
            #output value to result matrix 
            C[i,j] = sumHolder
    
    
    return C

ncC = naiveCpu(A, B)
CTest = A @ B
np.allclose(CTest, ncC)
                

###############################################################################
#naive gpu implimentation with notes on how to use
###############################################################################
#making a kernel
#the kernel is executed on each thread
@cuda.jit
def naiveGpu(A, B, C):
    """Perform matrix multiplication of C = A * B
    """
    #find which thread we are on
    i, j = cuda.grid(2)  
    #ensure this is a valid thread bc the blocks/grid can extend beyond the bounds of the matrix
    if i < C.shape[0] and j < C.shape[1]:
        #set a holder valu
        tmp = 0.                    
        #perform the multiplication and add it in the holder value        
        for k in range(A.shape[1]):
            tmp += A[i, k] * B[k, j]
        #put the holder value in the proper spot of the matrix
        C[i, j] = tmp
        
        


#for the sake of staying within the numba package i am going to create all of the
#arrays on the cpu and then move them to the gpu like we would for data in the 
#real world. However, it is easier to use the cupy package as shown here
# cp.random.seed(42)
# A = cp.random.uniform(1, 10, size=(2000, 2000), dtype=np.float64)  # array 1
# B = cp.random.uniform(1, 10, size=(2000, 2000), dtype=np.float64)  # array 2
# C = cp.zeros((2000, 2000), dtype=np.float64)       # array where we store answer 



#create matrices
np.random.seed(826)
A = np.random.normal(size = (2000, 2000))
B = np.random.normal(size = (2000, 2000))
C = np.empty((2000, 2000))
# C[:] = np.nan


#move to gpu
A_gpu = cuda.to_device(A)
B_gpu = cuda.to_device(B)
C_gpu = cuda.to_device(C)

#setting up the threads, block, grid structure
# we must have enough threads for every entry in the matrix since that is our 
#computation unit in the scenario
#i am sure there are optimal choices for threads per block the tutorial says 
#somewhere in the neightborhood of a total of 128-512 threads per block
#i am setting this arbitrarily
#must these be square?
threadsPerBlock = (19,19)
#to get the blocks in the grid, we see how many blocks it would take to fill the grid and round up
#must be stored as integers
blocksPerGrid_x = int(np.ceil(C.shape[0]/threadsPerBlock[0]))
blocksPerGrid_y = int(np.ceil(C.shape[1]/threadsPerBlock[1]))
gridSize = (blocksPerGrid_x, blocksPerGrid_y)


#you cannot assign new objects on the gpu, thats why we first have to feed it C which
#is the array our result will be in
naiveGpu[gridSize, threadsPerBlock](A_gpu, B_gpu, C_gpu)
CRes = C_gpu.copy_to_host()
CRes

CTest = A @ B
np.allclose(CTest, CRes)


###############################################################################
#operationalizing timing and computation
###############################################################################

A = np.random.normal(size = (10, 20))
B = np.random.normal(size = (20, 30))

def multTimes(A,B, threads = 20):    
    if A.shape[1] != B.shape[0]:
        raise Exception("non-conformable arrays")
    
    #numpy multiplication
    npStart = time.time()
    npC = A @ B
    npEnd = time.time() 
    npTime = npEnd - npStart
    
    #naive cpu
    ncStart = time.time()
    ncC = naiveCpu(A, B)
    ncEnd = time.time()
    ncTime =  ncEnd - ncStart
    
    #naive gpu
    #set up response matrix
    C = np.empty((A.shape[0], B.shape[1]))
    # C[:] = np.nan
    #thread set up
    threadsPerBlock = (threads,threads)
    blocksPerGrid_x = int(np.ceil(C.shape[0]/threadsPerBlock[0]))
    blocksPerGrid_y = int(np.ceil(C.shape[1]/threadsPerBlock[1]))
    gridSize = (blocksPerGrid_x, blocksPerGrid_y)
    #send A,B,C to gpu
    #not timing this, just compute speed
    A_gpu = cuda.to_device(A)
    B_gpu = cuda.to_device(B)
    C_gpu = cuda.to_device(C)
    #warm up round, otherwise the kernel must compile
    naiveGpu[gridSize, threadsPerBlock](A_gpu, B_gpu, C_gpu)
    cuda.synchronize()
    #perform multiplication
    bgStart = time.time()
    #since this is executed on a different device, once this command is sent to the 
    #gpu, the rest of the code continues to operate on the cpu until this piece
    #is called upon. Then, if it is not finished, the cpu waits for the result to 
    #come back. in order to get proper timings, we must force it to finish before
    #moving on to the next piece, we can do that with cuda.synchronize()
    naiveGpu[gridSize, threadsPerBlock](A_gpu, B_gpu, C_gpu)
    cuda.synchronize()
    bgEnd = time.time()
    bgTime = bgEnd - bgStart
    #retrieve value from gpu
    #not timing this, just compute speed
    bgC = C_gpu.copy_to_host()
    
    
    
    #check all values against numpy implimentation
    if np.allclose(npC, ncC) == False:
        raise Exception("nc and bg multiplactions have different results")
    if np.allclose(npC, bgC) == False:
        raise Exception("np and bg multiplactions have different results")
    
    return {"npTime": npTime, "ncTime": ncTime, "bgTime": bgTime}
    

multTimes(A, B)


###############################################################################
#running for various dimensions, square only for now
###############################################################################

import pandas as pd
dims = (10,100,1000)
timeHolder = pd.DataFrame({
    "dims": dims,
    "npTime": [pd.NA]*len(dims),
    "ncTime": [pd.NA]*len(dims),
    "bgTime": [pd.NA]*len(dims)
    })

for i in range(timeHolder.shape[0]):
    dimi = timeHolder.loc[i,"dims"]
    Ai = np.random.normal(size = (dimi, dimi))
    Bi = np.random.normal(size = (dimi, dimi))
    resi = multTimes(Ai, Bi)
    timeHolder.loc[i,"npTime"] = resi["npTime"]
    timeHolder.loc[i,"ncTime"] = resi["ncTime"]
    timeHolder.loc[i,"bgTime"] = resi["bgTime"]
    print("dimension "+str(dimi)+" complete")


timeHolder

import matplotlib.pyplot as plt
plt.plot(timeHolder["dims"], timeHolder["ncTime"])












