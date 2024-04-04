from DataStructures.core import NumpyData
from DataStructures.generateBox import generate_grid_box
import numpy as np
from m_shake import shake_step

def test_mshake():
    # setup params
    cell_dims = np.asarray([7, 7, 3])
    grid_spacings = np.asarray([1.4, 1.4, 1])
    bond_length = 0.94
    angle = np.pi
    dt = 0.5
    bond_eq = 1.0

    # pairs of atomic number and expected mass/charge
    charge_dict = {1: 0.5, 8: -1.}
    mass_dict = {1: 1., 8: 16.}

    # make a example box of a grid of water-molecules with random orientations
    test_box = generate_grid_box(cell_dims, grid_spacings, bond_length, angle, random_orientation=True)
    test_box.setup_masses(mass_dict)
    test_box.setup_charges(charge_dict)

    # setup integration loop
    x = test_box.get_positions()
    p = test_box.get_momenta()
    m = test_box.get_masses()
    p[:, :] = 0

    # integrate for 1000 steps with a random medium force
    for i in range(1000):
        force = np.random.randn(*x.shape) / 4
        molecule_idxs = test_box.molecule_idxs
        o_mass = test_box.mass_dict[8]
        h_mass = test_box.mass_dict[1]
        x, p, failed = shake_step(x, p, m, force, dt, bond_eq, molecule_idxs, o_mass=o_mass, h_mass=h_mass)
        if failed:
            break

    # check that meeting the boundary conditions was successfull
    assert not failed, print('M shake not sucessfull')