from DataStructures.core import NumpyData
from DataStructures.generateBox import generate_grid_box
import pytest
import numpy as np
import os


def test_NumpyData():

    # setup params
    cell_dims = np.asarray([3,3,1])
    grid_spacings = np.asarray([1.4,1.4,1])
    bond_length = 0.94
    angle = np.pi
    charge_dict = {1:0.5,8:-1.}
    mass_dict = {1:1., 8:16.}
    # all atoms are inside the verlet radius of each other
    verlet_radius = 3

    # setup io params
    file_name = 'test.pdb'
    file_error = 0.005
    test_time = 42

    numeric_tolerance = 1e-6


    # make a test box
    test_box = generate_grid_box(cell_dims, grid_spacings, bond_length, angle)
    no_molecules = np.floor(cell_dims/grid_spacings).prod()
    molecule_positions = test_box.get_molecule_positions()


    # check number of molecules
    assert molecule_positions.shape[0] == no_molecules

    # check oxygen indices
    assert (test_box.elements[test_box.molecule_idxs[:,0].flatten()] == 8).all()

    # check distance calculations:
    testdistance_calc(test_box, bond_length, angle, verlet_radius, numeric_tolerance=numeric_tolerance)

    # generate another box with random orientation and check again.
    test_box = generate_grid_box(cell_dims, grid_spacings, bond_length, angle, random_orientation=True)
    testdistance_calc(test_box, bond_length, angle, verlet_radius, numeric_tolerance=numeric_tolerance)

    # test the io, assign charges, velocities etc.
    test_box.setup_charges(charge_dict)
    test_box.setup_masses(mass_dict)

    assert test_box.charges is not None
    assert test_box.masses is not None

    new_positions = np.random.rand(*test_box.positions.shape)
    test_box.add_timestamp_to_trajectory(new_positions, test_time)

    test_box.to_file(file_name)
    loaded_test_box = NumpyData(cell_dims)
    loaded_test_box.from_file(file_name, mass_dict, charge_dict, frame=0)

    # files round everything to 3 decimal places so we have to take care of this as well
    assert compare_position_vectors(test_box.positions, loaded_test_box.positions, tolerance=file_error), \
        print(f'old_box:\n {test_box.positions[:12]} \n new_box:\n {test_box.positions[:12]}  ')

    test_box.from_file(file_name, mass_dict, charge_dict, frame=1)
    assert compare_position_vectors(new_positions, loaded_test_box.positions, tolerance=file_error), \
        print(f'old_box:\n {test_box.positions[:12]} \n new_box:\n {test_box.positions[:12]}  ')
    os.remove(file_name)


@pytest.mark.skip('This is just a helper func')
def compare_position_vectors(vecs1: np.ndarray, vecs2: np.ndarray, tolerance: float =1e-12) -> bool:
    """
    This is just a helper to see whether at least one vector in vecs1 is also in vecs2 (up to tolerance).
    :param vecs1: The first n_atom,3 np.ndarray
    :param vecs2: The first n_atom,3 np.ndarray
    :param tolerance: Numerical tolerance, because files generally round to 3 decimal places
    :return: whether the above is true or not
    """

    result = 0

    diff_array = vecs1[:,None,:] - vecs2[None,:,:]
    diff_array = np.linalg.norm(diff_array, axis=-1)
    for i in range(diff_array.shape[0]-2):
        result += (np.abs(diff_array[i, (i+1):]).max() < tolerance)
    return bool(~result)


@pytest.mark.skip('This is just a helper func')
def testdistance_calc(box: NumpyData, bond_length: float, angle: float, verlet_radius: float,
                      numeric_tolerance: float=1e-8):
    """
    Tests the different distance calculations of test_box. Assumes all watermolecules have
    the same bond_length and angle
    :param test_box: NumpyData Object to test the distance calcs on
    :param angle: water molecule angle
    :param bond_length: water molecule bond length
    :param verlet_radius: radius for neighbor lists
    :param numerical_tolerance: tolerance for equality comparisons
    :returns: hopefully no AssertionErrors
    """
    all_vecs = box.get_vectors() #inter_vecs, inter_dists = box.get_vectors_distances()
    atom_positions = box.get_positions() # positions of every atom

    n_atoms = atom_positions.shape[0]  

    # check the inter distance calculation
    for i in range(n_atoms):
        for j in range(i+1, n_atoms):
            ij_vec = atom_positions[j] - atom_positions[i]
            assert (np.abs(ij_vec - all_vecs[i, j]) < numeric_tolerance).all(), \
                print(f'vector {i},{j} is misleading') #(np.abs(ij_vec - inter_vecs[i, j]) < numeric_tolerance).all()

    # test the oxygen vectors
    o_vecs, _ = box.vecs_dists_to_oxy_format(all_vecs, np.zeros((all_vecs.shape[0],all_vecs.shape[0])))
    oxygen_positions = atom_positions[box.get_elements() == 8]
    n_oxy_atoms = oxygen_positions.shape[0]
    for o_i in range(n_oxy_atoms):
        for o_j in range(o_i +1, n_oxy_atoms):
            o_ij_vec = oxygen_positions[o_j, :] - oxygen_positions[o_i,:]
            assert (np.abs(o_ij_vec - o_vecs[o_i, o_j]) < numeric_tolerance).all(), \
                print(f'o_vector {o_i},{o_j} is misleading')  # o_vecs[o_i,o_j]


    # test the molecule vectors
    ho_vecs, _ = box.vecs_dists_to_ho_format(all_vecs, np.zeros((all_vecs.shape[0],all_vecs.shape[0])))
    mol_positions = box.get_molecule_positions()
    n_mols = mol_positions.shape[0]
    for l in range(n_mols):
        h1_vec_check = ho_vecs[l,0] + mol_positions[l, 1] - mol_positions[l, 0]
        h2_vec_check = ho_vecs[l,1] + mol_positions[l, 2] - mol_positions[l, 0]
        assert (np.abs(h1_vec_check) < numeric_tolerance).all(), \
            print(f'h1 vector is misleading in molecule {l}')
        assert (np.abs(h2_vec_check) < numeric_tolerance).all(), \
            print(f'h2 vector is misleading in molecule {l}')
