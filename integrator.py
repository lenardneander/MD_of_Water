import numpy as np
import simulation
from numba import njit, prange
import pbc

#@njit()
def baoab_step(universe,A,B,C,k,eq_dist,k_theta,eq_angle,L,dt,T,force,gamma=0, nbls=False, rcut=8.
                                                                     , r_skin=1.5):

    """Determine the positions and momenta for all particles for the next timestep with a langevin thermostat
    universe:   class
                contains all informations of the system
    A:          float
                Parameter for long range force
    B:          float
                Parameter for short range force
    C:          float
                Parameter for coulomb force
    k:          float
                Strength of the harmonic potential
    eq_dist:    float
                Equilibrium distance
    k_theta:    float
                Spring constant for angle
    eq_angle:   float
                Equilibrium angle in radians
    L:          float
                Box-Size
    dt:         float
                Timestep
    T:          float
                Temperature
    force:      ndarray
                Force from previous timestep
    gamma:      float
                Coupling parameter for the thermostat (gamma=0 no thermostat)
    returns:    2d-array
                forces for each particle
    """


    x = universe.get_positions()         #assign positions

    N = len(x)

    p = universe.get_momenta()        #Assign velocities  (modify later with universe.get_momenta())

    m = universe.get_masses()              #Get masses
    
    r_verlet = rcut + r_skin
   
    p = p+force*dt/2  #B
    x = x + p / m[:, None] * dt / 2     #A
    p = np.exp(-gamma*dt)*p + 0.0911837*np.sqrt((m[:, None]*T*(1-np.exp(-2*gamma*dt))))*np.random.normal(0,1,[N,3]) #O
    x = x + p / m[:, None] * dt / 2      #A
    universe.positions = x
    if not nbls:
        force = simulation.force_field(universe,A,B,C,k,eq_dist,k_theta,eq_angle,L)
    else:
        force = simulation.nbls_force_field(universe,A,B,C,k,eq_dist,k_theta,eq_angle,L, coulomb_radius=rcut, lj_radius=rcut, verlet_radius=r_verlet)
    
    p = p + force * dt / 2  #B
        
        
    return x, p, force


@njit()
def lj_forces(dist: np.ndarray, normed_vecs: np.ndarray, A: float, B: float, rc: float = 1e6, f_max = np.inf):
    """Determine the LJ-forces between all particles
    dist:       2d-array
                Distance-matrix between particle pairs (including pbcs)
    normed_vecs:3D-array of shape [n_oxy, n_oxy, 3] containg the normed vectors
                between the oxygen atoms
    A:          float
                Parameter for long range force
    B:          float
                Parameter for short range force
    rc:         float
                Cutoff radius for the LJ potential
    f_max:      float
                Maximum force in case particles come to close within one dt
    returns:    2d-array
                forces for each particle
    """

    N = len(dist)  # Dimension of array
    forces = np.zeros((N, 3))  # Define force array

    for i in range(N - 1):  # Loop over all matrix rows
        # get the upper triangle of distances in this row
        dist_row = dist[i, i + 1:]
        vec_row = normed_vecs[i, i + 1:]

        # select only entries under the cutoff radius
        rc_mask = np.argwhere((dist_row < rc) & (dist_row != 0.))

        # if the mask is empty, do nothing
        if rc_mask.size == 0:
            continue
        else:
            # weird indexing stuff
            rc_mask = rc_mask[:,0]

        dist_row = dist_row[rc_mask]
        vec_row = vec_row[rc_mask]

        r_inv = 1 / np.expand_dims(dist_row, axis=-1)  # calculate inverse distances
        r6_inv = r_inv ** 6  # distances**6
        force = -6 * r_inv * r6_inv * (A * 2 * r6_inv - B) * vec_row  # Calculate forces according to derrivative of LJP

        # fill in results
        forces[i] += np.sum(force, axis=0)
        forces[rc_mask + (i + 1)] -= force

    return forces


@njit()
def coulomb_forces(dist: np.ndarray, normed_vecs: np.ndarray, C: float, qs: np.ndarray, rc: float = 1e6, f_max = np.inf):
    """Determine the Coulomb-forces between all particles
    dist:       2d-array
                Distance-matrix between particle pairs (including pbcs)
    normed_vecs:3D-array of shape [n_atom, n_atom, 3] containg the normed vectors
                between all atoms
    C:          float
                Parameter for coulomb force
    qs:         array
                charges of each particle
    rc:         float
                Cutoff radius
    f_max:      float
                Maximum force in case particles come to close within one dt
    returns:    2d-array
                forces for each particle
                """

    N = len(dist)  # Dimension of array
    forces = np.zeros((N, 3))  # Define force array

    for i in range(N - 1):  # Loop over all matrix rows

        # get the upper triangle of distances in this row
        dist_row = dist[i, i + 1:]
        vec_row = normed_vecs[i, i + 1:]

        # select only entries under the cutoff radius
        rc_mask = np.argwhere(dist_row < rc)

        # if the mask is empty, do nothing
        if rc_mask.size == 0:
            continue
        else:
            # weird indexing stuff
            rc_mask = rc_mask[:,0]

        dist_row = dist_row[rc_mask]
        vec_row = vec_row[rc_mask]

        r_inv = 1 / np.expand_dims(dist_row, axis=-1)  # calculate inverse distances

        num_zeros = (2 - (i % 3))                      #Set intramolecular coulomb forces to zero
        zeros = np.array([0] * num_zeros)
        charge_factor = qs[i] * qs[i + 1::]      
        charge_factor[0:num_zeros] = zeros

        charge_factor = np.expand_dims(charge_factor, axis=-1)[rc_mask]     #Cutoff
        force = -C * charge_factor * r_inv ** 2 * vec_row  # Calculate forces according to derrivative of coulomb law

        # fill in results
        forces[i] += np.sum(force, axis=0)
        forces[rc_mask+(i+1)] -= force

    return np.clip(forces, -f_max, +f_max)   #cutoff force if too high

@njit()
def bond_forces(ho_dists, ho_vecs, k, eq_dist):
    """ Determine the bond forces for all water molecules
    ho_dists:   2D array of shape [n_mols, 2] of the distance of the oxygen and the two hydrogens
    ho_vecs:    3D array of shape [n_mols, 2, 3] normed vectors pointing from to hydrogens to oxygen
    k:          float
                Strength of the harmonic potential

    eq_dist:    float
                Equilibrium distance
         
    returns:    2d-array with dimensions (N, 3)
                Forces for each particle
    """

    # Dimension of array
    N = len(ho_dists)

    # Define force array
    forces = np.zeros((N, 3, 3))

    # calculate forces for the hydrogens
    forces[:,1:,:] = np.expand_dims(-k*(ho_dists - eq_dist), axis=-1)*(-ho_vecs)

    # and for the oxygen
    forces[:,0,:] = -forces[:,1:,:].sum(axis=1)

    return forces

@njit()
def angle_forces(dist, vect, k_theta, eq_angle):
    """ Determine the angle forces for all water molecules 
    dist:       2D array of shape [n_mol, 2] containing the distances between oxygen
                and the 2 hydrogens of mol i at index i
                Distances of H-O-H atoms (including pbcs)

    vect:       3D array of shape [n_mol, 2, 3] containing the normed vectors from the hydrogens to the oxygens in mol i
                Vectors of H-O-H atoms (including pbcs)

    k_theta:    float
                Stiffness of angle

    eq_angle:   float
                Equilibrium angle
                  
    returns:    3D array with dimensions (N_mol, 3, 3)
                Forces for atom j in molecule i at index [i,j]
    """

    # Dimension of array
    N = len(dist)

    # Define force array
    forces = np.zeros((N, 3, 3))

    # Calculate the relevant angles
    angle = np.arccos((vect[:,0]* vect[:,1]).sum(axis=-1))

    # get normal vectors for each of the hydrogens
    normal_vec_both = np.cross(vect[:,0], vect[:,1])
    normal_OH1 = -np.cross(vect[:, 0], normal_vec_both)
    normal_OH2 = -np.cross(-vect[:, 1], normal_vec_both)

    # norm these vectors
    normal_OH1 /= np.expand_dims(np.sqrt((normal_OH1 ** 2).sum(axis=-1)), axis=-1)
    normal_OH2 /= np.expand_dims(np.sqrt((normal_OH2 ** 2).sum(axis=-1)), axis=-1)

    # get the force magnitude and assign the force to the atoms
    force_magnitude = np.expand_dims(-k_theta * (angle-eq_angle), axis=-1)

    forces[:,1] = force_magnitude*normal_OH1
    forces[:,2] = force_magnitude*normal_OH2
    forces[:,0] = -(force_magnitude*normal_OH1 + force_magnitude*normal_OH2)

    return forces

# somehow parallel-numba compiles this lovely function into something useless, so no parallel execution here :(
@njit(parallel=False)
def nbls_coulomb_forces(dist: np.ndarray, normed_vecs: np.ndarray, neighborlist_idxs: np.ndarray, C: float, qs: np.ndarray, rc: float = 1e6):
    """Determine the Coulomb-forces between all particles
    dist:               1d-array distances between particles
    normed_vecs:        2d array of shape [n_pairs, 3] with normed vectors
    neighborlist_idxs:  2d array of shape [n_pairs, 2] containing the indices of from_atom, to_atom the vectors
                        and distances correspond to
    C:          float
                Parameter for coulomb force
    qs:         array
                charges of each particle
    rc:         float
                Cutoff radius

    returns:    2d-array
                forces for each particle
                """

    n_atoms = len(qs)  # Dimension of array

    # disselect based on cutoff
    rc_mask = (dist < rc)

    reduced_dist = dist[rc_mask]
    reduced_vecs = normed_vecs[rc_mask]
    reduced_nbls = neighborlist_idxs[rc_mask]

    # calculate forces per pair
    force_magnitudes = -C*qs[reduced_nbls[:,0]]*qs[reduced_nbls[:,1]]*((1/reduced_dist)**2)

    nbls_forces = np.expand_dims(force_magnitudes, axis=-1)*reduced_vecs

    # fill in results
    no_pairs = reduced_nbls.shape[0]
    forces = np.zeros((n_atoms, 3))

    # for loop go brrrrrrr
    for i in range(no_pairs):  # Loop over all matrix rows

        from_atom = reduced_nbls[i,0]
        to_atom = reduced_nbls[i,1]

        forces[from_atom] += nbls_forces[i]
        forces[to_atom] -= nbls_forces[i]

    return forces
