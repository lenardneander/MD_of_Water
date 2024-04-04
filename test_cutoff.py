import numpy as np
import integrator as ig
import pytest
#from DataStructures.core import NumpyData
#from DataStructures.generateBox import generate_grid_box
from DataStructures.core import NumpyData
from DataStructures.generateBox import generate_grid_box

def test_cutoff_lj():
    ### set 6 particles on different vertexes of a unit cube
    a = np.zeros((6,3))
    positions = a
    positions[1:5,1]=1
    positions[3:,0]=1
    positions[2:4,2]=1
    positions_2=positions
    # print(positions)
    # print(positions.shape)
    vectors = np.zeros((positions.shape[0], positions.shape[0], positions.shape[1]))
    # print(vectors)

    for i in range(positions.shape[0]):
        # get the vectors row wise
        tmp_vecs = positions_2[(i + 1):] - positions[i]
        vectors[i, (i + 1):] = tmp_vecs
        vectors[(i + 1):, i] = -tmp_vecs
    
    # print(vectors)
    dists = np.linalg.norm(vectors, axis=-1)
    vectors /= dists[:,:,None]
    

    ###  lj-force with cutoff-radius rc  ###
    # set rc =0.5 so there is not force between any pair of particles, they should all be cut off
    f_0 = ig.lj_forces(dists, vectors,1,1,rc=0.5)
    # set rc =2 so there is force for any pair of particles
    f_1 = ig.lj_forces(dists, vectors,1,1,rc=2)
    f_2 = ig.lj_forces(dists, vectors,1,1)

    assert f_0.all() == a.all() and f_1.all()==f_2.all()


def test_cutoff_coulomb():
    ### set 6 particles on different vertexes of a unit cube
    a = np.zeros((6, 3))
    positions = a
    positions[1:5, 1] = 1
    positions[3:, 0] = 1
    positions[2:4, 2] = 1
    positions_2 = positions
    vectors = np.zeros((positions.shape[0], positions.shape[0], positions.shape[1]))

    for i in range(positions.shape[0]):
        # get the vectors row wise
        tmp_vecs = positions_2[(i + 1):] - positions[i]
        vectors[i, (i + 1):] = tmp_vecs
        vectors[(i + 1):, i] = -tmp_vecs

    dists = np.linalg.norm(vectors, axis=-1)
    vectors /= dists[:, :, None]
    qs = np.ones(vectors.shape[0])

    ###  coulomb-force with cutoff-radius rc  ###
    # set rc =0.5 so there is not force between any pair of particles, they should all be cut off
    f_0 = ig.coulomb_forces(dists, vectors, 1, qs, rc=0.5)
    # set rc =2 so there is force for any pair of particles
    f_1 = ig.coulomb_forces(dists, vectors, 1, qs, rc=2)
    f_2 = ig.coulomb_forces(dists, vectors, 1, qs)

    assert f_0.all() == a.all() and f_1.all() == f_2.all()
   


