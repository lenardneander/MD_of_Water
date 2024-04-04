import pytest
import numpy as np
import integrator
from DataStructures import core
import pbc


def test_external_forces_shape():

    test_input_vecs = np.zeros([12,12,3])
    test_input2_vecs = np.zeros([6,6,3])

    test_input_dists = np.zeros([12, 12])
    test_input2_dists = np.zeros([6, 6])

    assert integrator.lj_forces(test_input_dists,test_input_vecs,1,1).shape == (12,3)
    assert integrator.lj_forces(test_input2_dists,test_input2_vecs,1,1).shape == (6,3)
    assert integrator.coulomb_forces(test_input_dists,test_input_vecs,1,np.zeros(12)).shape == (12,3)
    assert integrator.coulomb_forces(test_input2_dists,test_input2_vecs,1,np.zeros(6)).shape == (6,3)

    
def test_coulomb_force_direction():

    # we skip intramolecular forces, so this has to be padded with dummy values
    # only index 0 and 3 are relevant
    x = np.array([[1.,2.,3.],[0.,0.,1.],[0.,0.,2.],[2.,3.,4.],[0.,0.,3.],[0.,0.,4.]])

    vectors = pbc.pbc_distance(x,100)
    dists = np.linalg.norm(vectors, axis=-1)

    dir1 = integrator.coulomb_forces(dists, vectors, 1,np.array([1,0,0,1,0,0]))    #Repulsive

    dir2 = integrator.coulomb_forces(dists, vectors, 1,np.array([-1,0,0,1,0,0]))    #Attractive

    assert np.all(dir1[0] < 0) 
    assert np.all(dir1[3] > 0)

    assert np.all(dir2[0] > 0) 
    assert np.all(dir2[3] < 0)

def test_coulomb_magnitude():

    # we skip intramolecular forces, so this has to be padded with dummy values
    # only index 0 and 3 are relevant
    x = np.array([[1.,2.,3.],[0.,0.,1.],[0.,0.,2.],[2.,3.,4.],[0.,0.,3.],[0.,0.,4.]])

    vectors = pbc.pbc_distance(x, 100)
    dists = np.linalg.norm(vectors, axis=-1)
    vectors /= dists[:,:,None]

    coulomb1 = np.linalg.norm(integrator.coulomb_forces(dists, vectors,1,np.array([1,0,0,1,0,0]))[0])
    coulomb2 = np.linalg.norm(integrator.coulomb_forces(dists, vectors,1,np.array([1,0,0,1,0,0]))[3])

    assert np.round(coulomb1,5) == np.round(1/np.linalg.norm(np.array([1,1,1]))**2,5)
    assert np.round(coulomb2,5) == np.round(1/np.linalg.norm(np.array([1,1,1]))**2,5)  


def test_ljf_magnitude():

    x = np.array([[1.,2.,3.],[3.,4.,5.]])

    vectors = pbc.pbc_distance(x, 100)
    dists = np.linalg.norm(vectors, axis=-1)
    vectors /= dists[:, :, None]

    ljf1 = np.linalg.norm(integrator.lj_forces(dists, vectors,1,1)[0])

    ljf2 = np.linalg.norm(integrator.lj_forces(dists, vectors,1,1)[1])

    solution = 6/np.linalg.norm(np.array([2,2,2]))**7 - 12/np.linalg.norm(np.array([2,2,2]))**13

    assert np.round(ljf1,5) == np.round(solution,5)
    assert np.round(ljf2,5) == np.round(solution,5) 


def test_bond_force_shape():

    ho_vecs = np.ones([10,2,3])
    ho_dists = np.ones([10,2])

    assert integrator.bond_forces(ho_dists,ho_vecs,1,0).shape == (10,3,3)

def test_bond_force_dircetion():

    ho_vectors = np.array([[[1.,1.,1.],[-1.,-1.,-1.]]])
    ho_dists = np.linalg.norm(ho_vectors, axis=-1)
    ho_vectors /= ho_dists[:,:,None]

    assert np.all(integrator.bond_forces(ho_dists, ho_vectors,1,0) == np.array([[0,0,0],[1,1,1],[-1,-1,-1]]))

def test_bond_force_eq_distance():

    ho_vectors = np.array([[[1., 1., 1.], [-1., -1., -1.]]])
    ho_dists = np.linalg.norm(ho_vectors, axis=-1)
    ho_vectors /= ho_dists[:, :, None]

    assert np.all(integrator.bond_forces(ho_dists,ho_vectors,1,np.sqrt(3)) == np.zeros([1,3,3]))
    #oxy_position
    r1 = np.array([1, 2, 2])
    #hydro_positions
    r2 = np.array([1, 2, 3])
    r3 = np.array([1, 2, 1])
    r0 = 2
    k = 2
    expected_force1 = np.array([0, 0, 0])
    expected_force2 = np.array([0, 0, 2])
    expected_force3 = np.array([0, 0, -2])

    ho_vectors = np.array([[[0., 0., -1.], [0., 0., 1.]]])
    ho_dists = np.linalg.norm(ho_vectors, axis=-1)
    ho_vectors /= ho_dists[:, :, None]

    force_array = integrator.bond_forces(ho_dists,ho_vectors, k, r0)
    force1, force2, force3 = force_array[0,0], force_array[0,1], force_array[0,2]
    assert np.allclose(force1, expected_force1), f'Error: {force1}'
    assert np.allclose(force2, expected_force2), f'Error: {force2}'
    assert np.allclose(force3, expected_force3), f'Error: {force3}'

    
    





