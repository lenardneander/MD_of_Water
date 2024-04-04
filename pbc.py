import numpy as np
from numba import njit, prange

@njit(parallel=True)
def pbc_shift (x,L):
    """Determine positions of all particles under pbcs
    x:          2d-array
                positions of all atoms

    L:          float
                cubic box length
         
    returns:    2d-array
                positions of all atoms under pbcs
    """


    for i in prange(0,len(x)):
        for k in range(0,3):
            if x[i,k]> L/2:
                x[i,k]=x[i,k]-L
            elif x[i,k]< -L/2:
                x[i,k]=x[i,k]+L
    return(x)

@njit(parallel=False)
def pbc_distance(x,L):
    """Determine distances of all particles under pbcs
    x:          2d-array
                positions of all atoms

    L:          float
                cubic box length
         
    returns:    2d-array
                distances of all atoms under pbcs
    """
   
    dist = x - np.expand_dims(x, axis=1)          #Calculate distance without pbs
    periodic_plus = (dist>L/2)                    #Check if distance is larger than half of boxsize
    periodic_minus = (dist<-L/2)
    dist = np.where(periodic_plus,dist,dist+L)    #If distance is larger than half of boxsize shift distance by boxsize
    dist = np.where(periodic_minus,dist,dist-L)

    return dist

