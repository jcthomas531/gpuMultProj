import os
os.chdir("H:\\schoolFiles\\bios7330AdvComputing\\gpuMultProj")
import multKernels as mk
import cupy as cp
import numpy as np
from numba import cuda
import time
from numba import float32, int32, float64
from threadpoolctl import threadpool_limits


###############################################################################
#operationalizing timing and computation
###############################################################################

def multTimes(A,B, threads = 20, avoidMassiveCpu = True):    
    if A.shape[1] != B.shape[0]:
        raise Exception("non-conformable arrays")
    if A.shape[0] != A.shape[1]:
        raise Exception("A is not square")
    if B.shape[0] != B.shape[1]:
        raise Exception("B is not square")
    
    
    
    
    
    #dimension, will be the same bc we are only allowing square matrices
    N = A.shape[0]
    #using correct numeric type, less exact with float32 compared to float64 but it
    #will run quicker, we just have to remember to turn down the tolerance when checking
    A = A.astype(np.float32)
    B = B.astype(np.float32)
    
    #set up for gpu
    #response matrix
    C = np.empty((N, N)).astype(np.float32)
    #thread set up
    threadsPerBlock = (threads,threads)
    blocksPerGrid_x = int(np.ceil(C.shape[0]/threadsPerBlock[0]))
    blocksPerGrid_y = int(np.ceil(C.shape[1]/threadsPerBlock[1]))
    gridSize = (blocksPerGrid_x, blocksPerGrid_y)
    #send A,B to gpu
    #not timing this, just compute speed
    A_gpu = cuda.to_device(A)
    B_gpu = cuda.to_device(B)
    
    
    
    
    
    
    
    #numpy multiplication
    npStart = time.time()
    with threadpool_limits(limits=1):
        npC = A @ B
    npEnd = time.time() 
    npTime = npEnd - npStart
    
    #naive cpu
    #avoid large computations with niave method
    if avoidMassiveCpu == True and N>800:
        ncTime = np.nan
        ncC = npC
    else :
        ncStart = time.time()
        ncC = mk.naiveCpu(A, B)
        ncEnd = time.time()
        ncTime =  ncEnd - ncStart
    
    #naive gpu
    ngC_gpu = cuda.device_array((N, N), dtype=np.float32)
    #warm up round, otherwise the kernel must compile
    mk.naiveGpu[gridSize, threadsPerBlock](A_gpu, B_gpu, ngC_gpu)
    cuda.synchronize()
    #perform multiplication
    ngStart = time.time()
    #since this is executed on a different device, once this command is sent to the 
    #gpu, the rest of the code continues to operate on the cpu until this piece
    #is called upon. Then, if it is not finished, the cpu waits for the result to 
    #come back. in order to get proper timings, we must force it to finish before
    #moving on to the next piece, we can do that with cuda.synchronize()
    mk.naiveGpu[gridSize, threadsPerBlock](A_gpu, B_gpu, ngC_gpu)
    cuda.synchronize()
    ngEnd = time.time()
    ngTime = ngEnd - ngStart
    #retrieve value from gpu
    #not timing this, just compute speed
    ngC = ngC_gpu.copy_to_host()
    
    
    #tile gpu method
    tiC_gpu = cuda.device_array((N, N), dtype=np.float32)
    #use factory to create function so dimension can be dynamic
    tileMult = mk.tileMultFactory(threads)
    #warm up
    tileMult[gridSize, threadsPerBlock](A_gpu, B_gpu, tiC_gpu)
    cuda.synchronize()
    #perform multiplication
    tiStart = time.time()
    tileMult[gridSize, threadsPerBlock](A_gpu, B_gpu, tiC_gpu)
    cuda.synchronize()
    tiEnd = time.time()
    tiTime = tiEnd - tiStart
    #retrieve value
    tiC = tiC_gpu.copy_to_host()    
    
    
    
    #cupy method
    A_gpu2 = cp.asarray(A_gpu)
    B_gpu2 = cp.asarray(B_gpu)
    cuStart= time.time()
    cuC_gpu = cp.matmul(A_gpu2, B_gpu2)
    cuEnd = time.time()
    cuTime = cuEnd-cuStart
    cuC = cuC_gpu.get()
    
    
    tol_ = 1e-3
    #check all values against numpy implimentation
    if np.allclose(npC, ncC, atol=tol_, rtol=tol_) == False:
        raise Exception("np and nc multiplactions have different results")
    if np.allclose(npC, ngC, atol=tol_, rtol=tol_) == False:
        raise Exception("np and ng multiplactions have different results")
    if np.allclose(npC, tiC, atol=tol_, rtol=tol_) == False:
        raise Exception("np and ti multiplactions have different results")
    if np.allclose(npC, cuC, atol=tol_, rtol=tol_) == False:
        raise Exception("np and cu multiplactions have different results")
    
    return {"npTime": npTime, "ncTime": ncTime, "ngTime": ngTime, "tiTime": tiTime, "cuTime": cuTime}
    

#example
# A = np.random.normal(size = (20, 20))
# B = np.random.normal(size = (20, 20))
# multTimes(A, B, threads=10)






###############################################################################
#function to compare thread number times on a set matrix size
###############################################################################

def threadTimes(threads, dim = 2000):
    
    if dim % threads != 0:
        raise Exception("dimension must be divisible by threads")
        
    
    
    #get matrices of desired size
    A = np.random.normal(size = (dim, dim)).astype(np.float32)
    B = np.random.normal(size = (dim, dim)).astype(np.float32)
    #set up for gpu
    #response matrix
    C = np.empty((dim, dim)).astype(np.float32)
    #thread set up
    threadsPerBlock = (threads,threads)
    blocksPerGrid_x = int(np.ceil(C.shape[0]/threadsPerBlock[0]))
    blocksPerGrid_y = int(np.ceil(C.shape[1]/threadsPerBlock[1]))
    gridSize = (blocksPerGrid_x, blocksPerGrid_y)
    #send A,B to gpu
    #not timing this, just compute speed
    A_gpu = cuda.to_device(A)
    B_gpu = cuda.to_device(B)
    
    
    #naive gpu
    ngC_gpu = cuda.device_array((dim, dim), dtype=np.float32)
    #warm up round, otherwise the kernel must compile
    mk.naiveGpu[gridSize, threadsPerBlock](A_gpu, B_gpu, ngC_gpu)
    cuda.synchronize()
    #perform multiplication
    ngStart = time.time()
    #since this is executed on a different device, once this command is sent to the 
    #gpu, the rest of the code continues to operate on the cpu until this piece
    #is called upon. Then, if it is not finished, the cpu waits for the result to 
    #come back. in order to get proper timings, we must force it to finish before
    #moving on to the next piece, we can do that with cuda.synchronize()
    mk.naiveGpu[gridSize, threadsPerBlock](A_gpu, B_gpu, ngC_gpu)
    cuda.synchronize()
    ngEnd = time.time()
    ngTime = ngEnd - ngStart
    #not retrieving value here
    
    
    #tile gpu method
    tiC_gpu = cuda.device_array((dim, dim), dtype=np.float32)
    #use factory to create function so dimension can be dynamic
    tileMult = mk.tileMultFactory(threads)
    #warm up
    tileMult[gridSize, threadsPerBlock](A_gpu, B_gpu, tiC_gpu)
    cuda.synchronize()
    #perform multiplication
    tiStart = time.time()
    tileMult[gridSize, threadsPerBlock](A_gpu, B_gpu, tiC_gpu)
    cuda.synchronize()
    tiEnd = time.time()
    tiTime = tiEnd - tiStart
    #not retrieving value here
    
    #cupy method
    A_gpu2 = cp.asarray(A_gpu)
    B_gpu2 = cp.asarray(B_gpu)
    cuStart= time.time()
    cuC_gpu = cp.matmul(A_gpu2, B_gpu2)
    cuEnd = time.time()
    cuTime = cuEnd-cuStart
    
    
    
    
    return {"ngTime": ngTime, "tiTime": tiTime, "cuTime": cuTime}
    
    
    
#example:
#threadTimes(threads = 32, dim=2048)





