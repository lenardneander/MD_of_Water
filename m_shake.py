import numpy as np
from DataStructures.core import NumpyData
from numba import njit

@njit()
def shake_step(x: np.ndarray, p: np.ndarray, m:np.ndarray, force: np.ndarray, dt: float, bond_eq: float,
               molecule_idxs: np.ndarray, pbcs: float = 1e6, o_mass: float = 16., h_mass: float = 1., max_it=10
               , tolerance=0.0001, gamma=0.1, T=300):
    """
    Completes a M shake step keeping the bond_lengths fixed at bond_eq,
    Integration is done via the leapfrog integrator
    :param x: np.ndarray of shape [n_atom, 3] with the atomic positions
    :param p: np.ndarray of shape [n_atom, 3] with the atomic momenta
    :param m: np.ndarray of shape [n_atom] with the atomic masses
    :param force: np.ndarray of shape [n_atom, 3] with the atom wise forces
    :param dt: float - timestep size
    :param bond_eq: bond_distance to be fixed
    :param molecule_idxs: np.ndarray of shape [n_mols, 3] with the atom indices of molecule i at i in the first axis
    :param o_mass: mass of the oxygen atoms
    :param h_mass: mass of the hydrogen atoms
    :param max_it: max number of M-shake iterations
    :param tolerance: the max divergence from the actual desired bondlength
    :param pbcs: box size (cubic boxes please)
    :param gamma: coupling constant for langevin thermostat
    :param T: temperature (in K)
    :return: x_t_1, p_t_1, failed_m_shake_step
     x_t_1, p_t_1: np.ndarrays new positions and momenta
     failed_m_shake_step: bool, whether the mshake step failed
    """
    # unconstrained step
    expanded_m = np.expand_dims(m, axis=-1)
    N = m.shape[0]

    # BAOA step first
    p = p + force * dt  # B
    x_t_1 = x + p / expanded_m * dt / 2  # A
    p = np.exp(-gamma*dt)*p + 0.0911837*np.sqrt((expanded_m*T*(1-np.exp(-2*gamma*dt))))*np.random.normal(0,1,(N,3)) #O
    x_t_1 = x_t_1 + p /expanded_m * dt / 2

    # Correction afterwards

    # get the original bond vectors
    bond_vecs = get_bond_vecs(x, molecule_idxs, pbcs)

    failed_m_shake_step = True
    # iterate matrix inversion
    for i in range(max_it):
        # get new bond vecs
        bond_vecs_t_1 = get_bond_vecs(x_t_1, molecule_idxs, pbcs)

        # abort if the constraint is already satisfied
        if (np.abs((bond_vecs_t_1**2).sum(axis=-1) - bond_eq**2) < tolerance).all():
            failed_m_shake_step = False
            break

        # solve the equations otherwise
        # get the lagrange multipliers
        ls = solve_mshake_mat(bond_vecs, bond_vecs_t_1, o_mass, h_mass, bond_eq, dt)
        # correct the positions
        h_atom_correction = 2 * (dt ** 2) * np.expand_dims(ls, axis=-1) * -bond_vecs
        all_atom_correction = m_shake_to_all_atom(h_atom_correction, molecule_idxs) / expanded_m
        x_t_1 += all_atom_correction

    p_t_1 = (x_t_1 - x)*expanded_m / dt

    return x_t_1, p_t_1, failed_m_shake_step

@njit()
def solve_mshake_mat(bond_vecs: np.ndarray, bond_vecs_t_1: np.ndarray, m_oxy: float, m_h: float, bond_eq: float,
                     dt: float):
    """
    Solves the M Shake matrix vector equation to identify the lagrange multipliers
    :param bond_vecs: np.ndarray of shape [n_hydro *2, 3] of the bond vectors pointing from O to H
    :param bond_vecs_t_1: np.ndarray of shape [n_hydro *2, 3] of the bond vectors in the next timestep
    :param m_oxy: float mass of the oxygen atoms
    :param m_h: float mass of the hydrogen atoms
    :param bond_eq: float bond length to be fixed
    :param dt: float timestep size
    :return: ls, a np.ndarray of shape [n_hydro*2] containing the (approximate) lagrange multipliers
    """

    no_rows = bond_vecs.shape[0]

    # calculate A matrix elements
    diag_elements = (1 / m_oxy + 1 / m_h) * (bond_vecs * bond_vecs_t_1).sum(axis=-1)

    # in the small blocks coupling both hydrogens in each molecule, the dot products are a bit offset
    # here the dot product between the bond vector pointing to one hydrogen from the last timestep
    # and the bond vector pointing to the other hydrogen from this timestep is needed
    upper_diag_elements = np.zeros((no_rows - 1))
    upper_diag_elements[::2] = (1 / m_oxy) * (bond_vecs[1::2] * bond_vecs_t_1[::2]).sum(axis=-1)
    lower_diag_elements = np.zeros((no_rows - 1))
    lower_diag_elements[::2] = (1 / m_oxy) * (bond_vecs[::2] * bond_vecs_t_1[1::2]).sum(axis=-1)

    # construct A matrix
    A = np.zeros((bond_vecs.shape[0], bond_vecs.shape[0]))

    # fill in the values
    np.fill_diagonal(A, diag_elements)

    # here we need to select every second entry (the matrix has 2x2 blocks along the diagonal)
    for i in range(no_rows-1):
        A[i, i+1] = upper_diag_elements[i]
        A[i+1, i] = lower_diag_elements[i]

    # calculate the c vector
    c = ((bond_vecs_t_1 ** 2).sum(axis=-1) - bond_eq ** 2) / (4 * dt ** 2)

    # solve the system
    ls = np.linalg.solve(A, c)
    return ls

@njit()
def get_bond_vecs(x: np.ndarray, mol_idxs: np.ndarray, pbcs: float):
    """
    Calculates the bond vecs again. This is the part that makes M shake messy and the part I dont like.
    :param x: np.ndarray of shape [n_atom, 3] with the atomic positions
    :param mol_idxs: np.ndarray of shape [n_mols, 3] where entry i contains the indices of the three atoms
    :param pbcs: the box size (only cubic boxes please)
    :return: bond_vecs, np.ndarray of shape [n_mols*2,3] with the vectors pointing from oxygen to the hydrogens in a mol
    """

    # get the indices first
    o_idxs = mol_idxs[:,0]
    h1_idxs = mol_idxs[:,1]
    h2_idxs = mol_idxs[:,2]

    # make a result array
    bond_vecs = np.zeros((mol_idxs.shape[0]*2, 3))
    # every second entry is the O->H1 vector
    bond_vecs[::2] = x[h1_idxs] - x[o_idxs]
    bond_vecs[1::2] = x[h2_idxs] - x[o_idxs]

    # apply pbcs
    bond_vecs = np.where(bond_vecs > (pbcs / 2), bond_vecs - pbcs, bond_vecs)
    bond_vecs = np.where(bond_vecs < (-pbcs / 2), bond_vecs + pbcs, bond_vecs)

    return bond_vecs

@njit()
def m_shake_to_all_atom(h_atom_correction, mol_idxs):
    """
    Calculates the bond vecs again. This is the part that makes M shake messy and the part I dont like.
    :param h_atom_correction: np.ndarray of shape [n_mols*2, 3] with the hydrogen atomic position corrections
    :param mol_idxs: np.ndarray of shape [n_mols, 3] where entry i contains the indices of the three atoms
    :return: all_atom_correction, np.ndarray of shape [n_atom,3] with the corrections for all atoms
    """

    # get the indices first
    o_idxs = mol_idxs[:,0]
    h1_idxs = mol_idxs[:,1]
    h2_idxs = mol_idxs[:,2]

    all_atom_correction = np.zeros((mol_idxs.flatten().shape[0], 3))
    all_atom_correction[h1_idxs] = h_atom_correction[::2]
    all_atom_correction[h2_idxs] = h_atom_correction[1::2]
    all_atom_correction[o_idxs] = -(all_atom_correction[h1_idxs] + all_atom_correction[h2_idxs])

    return all_atom_correction