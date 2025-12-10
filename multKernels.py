

import cupy as cp
import numpy as np
from numba import cuda
import time
from numba import float32, int32, float64


###############################################################################
#niave cpu implimentation for comparison with gpu that is non-optimized like the numpy multiplication
###############################################################################




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

#example:
# A = np.random.normal(size = (10, 20))
# B = np.random.normal(size = (20, 30))
# ncC = naiveCpu(A, B)
# CTest = A @ B
# np.allclose(CTest, ncC)




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


#example
# #create matrices
# np.random.seed(826)
# A = np.random.normal(size = (2000, 2000))
# B = np.random.normal(size = (2000, 2000))
# C = np.empty((2000, 2000))
# # C[:] = np.nan


# #move to gpu
# A_gpu = cuda.to_device(A)
# B_gpu = cuda.to_device(B)
# C_gpu = cuda.to_device(C)

# #setting up the threads, block, grid structure
# # we must have enough threads for every entry in the matrix since that is our 
# #computation unit in the scenario
# #i am sure there are optimal choices for threads per block the tutorial says 
# #somewhere in the neightborhood of a total of 128-512 threads per block
# #i am setting this arbitrarily
# #must these be square?
# threadsPerBlock = (19,19)
# #to get the blocks in the grid, we see how many blocks it would take to fill the grid and round up
# #must be stored as integers
# blocksPerGrid_x = int(np.ceil(C.shape[0]/threadsPerBlock[0]))
# blocksPerGrid_y = int(np.ceil(C.shape[1]/threadsPerBlock[1]))
# gridSize = (blocksPerGrid_x, blocksPerGrid_y)


# #you cannot assign new objects on the gpu, thats why we first have to feed it C which
# #is the array our result will be in
# naiveGpu[gridSize, threadsPerBlock](A_gpu, B_gpu, C_gpu)
# CRes = C_gpu.copy_to_host()
# CRes

# CTest = A @ B
# np.allclose(CTest, CRes)



###############################################################################
#tiling gpu approach
#making above into a factory so i can dynamically set dimension size
###############################################################################
#!!!!!!!!this is only set up to work on square matrices!!!!!!!!!!!!!!!!!!!!!
#if you dont do this, you have set a single dimension size and once the code
#compiles (ie the function is run), you cannot change it
def tileMultFactory(tileSize):
    @cuda.jit
    def tileMult(A, B, C):
        
        #initialize an array in the shared memory for the parts of A and B to be loaded in
        #dimension and type must be known when this compiles
        sA = cuda.shared.array(shape=(tileSize, tileSize), dtype=float32)
        sB = cuda.shared.array(shape=(tileSize, tileSize), dtype=float32)
        
        #get the position of the thread relative to the entire grid
        absX, absY = cuda.grid(2)
        
        #get the position of the thread realtive to the block
        relX = cuda.threadIdx.x
        relY = cuda.threadIdx.y
        
        #tells us how many blocks are in the grid
        bpg = cuda.gridDim.x  
        
        
        
        #ensure that our absolute position is within the boundary of our output
        if absX >= C.shape[0] and absY >= C.shape[1]:
            # Quit if (absX, absY) is outside of valid C boundary
            return
        
        
        
        #now we can start to work within the shared memory
        #initialize value that will be added to across all the blocks corresponding
        #to this element
        tmp = 0.
        #iterating over all of the blocks in our grid
        for i in range(bpg):
            # Preload data into shared memory
            #load the block we want into shared memory one element at a time
            #this is done one element at a time bc threads work on one element 
            #of the matrix only
            #in words:
            #sA[realtiveX, relativeY] = A[absX, relativeY + blockNum * tileSize]
            #the addition here allows us to offset based on which block we are in
            #beause in order to calculate a single tile, we must load multiple blocks in
            sA[relX, relY] = A[absX, relY + i * tileSize]
            sB[relX, relY] = B[relX + i * tileSize, absY]
            
            
            #in order to do the multiplication correctly, we need to make sure that
            #one thread is not in one block while another thread is in a different one
            #to do this, we sync the threads
            cuda.syncthreads()

            #do the basic multiplication for the current thread using the current block
            #this is just as we did on the naive implimentation but now this value will
            #be added to as the outer loop moves through the blocks
            for j in range(tileSize):
                tmp += sA[relX, j] * sB[j, relY]

            #sync again so that we know we are done adding to this value before it
            #goes into the result matrix
            cuda.syncthreads()

        C[absX, absY] = tmp
    return tileMult


#example
# tileMult = tileMultFactory(10)

# N = 2000

# A = np.random.normal(size = (N, N)).astype(np.float32)
# B = np.random.normal(size = (N, N)).astype(np.float32)
# #C = np.empty((2000, 2000)).astype(np.float32)


# A_gpu = cuda.to_device(A)
# B_gpu = cuda.to_device(B)
# #always initialize the empty array on the device, not the host
# #only go from host to device when you have specific value that need to be transfered
# #C_gpu = cuda.to_device(C)
# C_gpu = cuda.device_array((N, N), dtype=np.float32)

# threadsPerBlock = (10,10)

# blocksPerGrid_x = int(np.ceil(N/threadsPerBlock[0]))
# blocksPerGrid_y = int(np.ceil(N/threadsPerBlock[1]))
# gridSize = (blocksPerGrid_x, blocksPerGrid_y)

# tileMult[gridSize, threadsPerBlock](A_gpu, B_gpu, C_gpu)
# CRes = C_gpu.copy_to_host()

# CTest = A @ B
# np.allclose(CTest, CRes, atol=1e-4, rtol=1e-4)
# #tolerance can be an issue when using float32, not as percise as 64 but will 
# #work faster, should be consistent across all methods used, float32 is probably 
# #better here and just lowering the tolerance threshold
