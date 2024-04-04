import numpy as np
import integrator as ig
from tqdm import tqdm
from m_shake import shake_step
import pbc




def force_field(universe,A,B,C,k,eq_dist,k_theta,eq_angle,L, rcut=1e8):

    """Determine the forces for all particles
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
    eq_anlge:   float
                Equilibrium angle in radians
    L:          float
                Box-Size
    rcut:       float, cutoff for the forcefield
    returns:    2d-array
                forces for each particle
    """

    # get the charges and the vectors
    qs = universe.get_charges().copy()
    all_atom_vectors = pbc.pbc_distance(universe.get_positions(),L)

    # apply pbcs here once to the vectors
    # not sure if this really works, but it should
    #all_atom_vectors += -np.round(all_atom_vectors/(L/2))*(L)

    #get the all atom distance matrix and norm the vectors once
    all_atom_distances = np.linalg.norm(all_atom_vectors, axis=-1)
    all_atom_vectors /= all_atom_distances[:,:,None]

    #reindex it for the different contributions
    # select only the oxy atoms for LJ
    # ie, these are the vectors and distance matrix of only the oxygen atoms
    oxy_vecs, oxy_dists = universe.vecs_dists_to_oxy_format(all_atom_vectors, all_atom_distances)

    # for the intramolecular things, get a matrix containing the vectors pointing from oxy to each of the hydrogens
    # ho_vecs is a [n_mol, 2, 3] array with vector from o to hj in mol i at ho_vecs[i,j]
    ho_vecs, ho_dists = universe.vecs_dists_to_ho_format(all_atom_vectors, all_atom_distances)

    # compute the different force contributions with them
    coulomb_force = ig.coulomb_forces(all_atom_distances, all_atom_vectors, C, qs, rc=rcut)
    lj_force = ig.lj_forces(oxy_dists, oxy_vecs, A, B, rc=rcut)
    bond_force = ig.bond_forces(ho_dists, ho_vecs, k, eq_dist)
    angle_force = ig.angle_forces(ho_dists, ho_vecs,k_theta,eq_angle)

    # reindex the mol and oxy contributions to the all atom format
    lj_force = universe.from_oxygen_format(lj_force)
    bond_force = universe.from_mol_format(bond_force)
    angle_force = universe.from_mol_format(angle_force)

    # add them all up
    force = bond_force  + lj_force + coulomb_force + angle_force

    return force

def nbls_force_field(universe,A,B,C,k,eq_dist,k_theta,eq_angle,L, lj_radius=4, coulomb_radius=6, verlet_radius=8):

    """Determine the forces for all particles
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
    eq_anlge:   float
                Equilibrium angle in radians
    L:          float
                Box-Size
    returns:    2d-array
                forces for each particle
    """

    force_radius = max(lj_radius, coulomb_radius)

    # get the charges and the vectors
    qs = universe.get_charges()
    vectors, distances, neighborlist_idxs = universe.get_nbls_vectors_distances(verlet_radius, force_radius)
    ho_vecs, ho_dists = universe.get_ho_vectors()

    # norm the vectors
    vectors /= distances[:,None]
    ho_vecs /= ho_dists[:,:,None]

    # reindex the intermol vectors for the ljp
    oxy_vecs, oxy_dists = universe.nbls_to_oxy_format(vectors, distances)

    # compute the different force contributions with them
    lj_force = ig.lj_forces(oxy_dists, oxy_vecs, A, B, rc=lj_radius)
    coulomb_force = ig.nbls_coulomb_forces(distances, vectors, neighborlist_idxs, C, qs, rc=coulomb_radius)

    bond_force = ig.bond_forces(ho_dists, ho_vecs, k, eq_dist)
    angle_force = ig.angle_forces(ho_dists, ho_vecs,k_theta,eq_angle)

    # reindex the mol and oxy contributions to the all atom format
    lj_force = universe.from_oxygen_format(lj_force)
    bond_force = universe.from_mol_format(bond_force)
    angle_force = universe.from_mol_format(angle_force)

    # add them all up
    force = bond_force + angle_force + lj_force + coulomb_force

    return force




def step(universe,A,B,C,k,eq_dist,k_theta,eq_angle,L,dt,T, nbls=False, rcut=8, rskin=2):
    """Determine the forces for all particles
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
    eq_anlge:   float
                Equilibrium angle in radians
    L:          float
                Box-Size
    dt:         float
                Timestep
    T:          float
                Temperature
    returns:    2d-array
                new positions and momenta
    """


    pos = universe.get_positions()         #assign positions

    mom = universe.get_momenta()        #Assign velocities  (modify later with universe.get_momenta())

    m = universe.get_masses()              #Get masses

    if nbls:
        r_verlet = rcut + rskin
        f = nbls_force_field(universe,A,B,C,k,eq_dist,k_theta,eq_angle,L
                             , lj_radius=rcut, coulomb_radius=rcut, verlet_radius=r_verlet)     #Calculate force field
    else:
        f = force_field(universe,A,B,C,k,eq_dist,k_theta,eq_angle,L)

    return ig.baoab_step(dt,pos,mom,f,m,T)

def m_shake_step(universe,A,B,C,k,eq_dist,k_theta,eq_angle,L,dt,T, bond_eq, gamma=0.1, nbls=False, rcut=8, rskin=2):
    """Determine the forces for all particles
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
    eq_anlge:   float
                Equilibrium angle in radians
    L:          float
                Box-Size
    dt:         float
                Timestep
    T:          float
                Temperature
    gamma:      float coupling constant for the thermostat
    bond_eq:    flaot, equilibrium constant to keep the bonds fixed to
    nbls:       bool, whether to use a neighborlist
    rcut:       float, cutoff radius for the forcefield
    rskin:      float, verlet skin radius for the neighborlists
    returns:    2d-array
                new positions and momenta
    """


    pos = universe.get_positions()         #assign positions

    mom = universe.get_momenta()        #Assign velocities  (modify later with universe.get_momenta())

    m = universe.get_masses()              #Get masses

    if nbls:
        r_verlet = rcut + rskin
        f = nbls_force_field(universe,A,B,C,k,eq_dist,k_theta,eq_angle,L
                             , lj_radius=rcut, coulomb_radius=rcut, verlet_radius=r_verlet)     #Calculate force field
    else:
        f = force_field(universe,A,B,C,k,eq_dist,k_theta,eq_angle,L, rcut=rcut)

    x, p, failed = shake_step(pos, mom, m, f, dt, bond_eq, universe.molecule_idxs, pbcs=universe.cell[0], T=T
                              , gamma=gamma)
    return x, p, failed


def simulate(universe,A,B,C,k,eq_dist,k_theta,eq_angle,L,T,dt,steps, nbls=False, m_shake=False, bond_eq=1.012
             , gamma=0.1, rcut=1e8):
    """Determine the forces for all particles
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
    T:          float
                Temperature
    dt:         float
                Timestep
    gamma:      coupling parameter for the thermostat
    rcut:       cutoff for the forcefield
    steps:      int
                number of timesteps
    returns:    array
                positions and momenta for each timestep
    """

    pos_list = []
    momenta_list = []

    rskin = 1.5      # this value seems fine.
    r_verlet = rcut + rskin
    failed = False
    failed_counter = 0
    
    if not nbls:
        force = force_field(universe,A,B,C,k,eq_dist,k_theta,eq_angle,L)
    else:
        force = nbls_force_field(universe,A,B,C,k,eq_dist,k_theta,eq_angle,L, coulomb_radius=rcut, lj_radius=rcut, verlet_radius=r_verlet)
    
    for i in tqdm(range(steps)):
        if m_shake:
            if not failed:
                x, p, failed = m_shake_step(universe,A,B,C,k,eq_dist,k_theta,eq_angle,L,dt,T, bond_eq,
                                            nbls=nbls, rcut=rcut, rskin=rskin, gamma=gamma)
            else:
                if not nbls:
                    force = force_field(universe,A,B,C,k,eq_dist,k_theta,eq_angle,L)
                else:
                    force = nbls_force_field(universe,A,B,C,k,eq_dist,k_theta,eq_angle,L, coulomb_radius=rcut, lj_radius=rcut, verlet_radius=r_verlet)
    
                x, p, force = step(universe, A, B, C, k, eq_dist, k_theta, eq_angle, L, dt, T, nbls=nbls, rcut=rcut,
                            rskin=rskin)
                failed = False
                failed_counter+=1
        else:
            x, p, force = ig.baoab_step(universe,A,B,C,k,eq_dist,k_theta,eq_angle,L,dt,T,force,gamma=gamma, nbls=nbls, rcut=rcut
                                                                     , r_skin=rskin)

        x = pbc.pbc_shift(x,L)
        pos_list.append(x)
        momenta_list.append(p)
        universe.set_positions(x)
        universe.momenta = p

    print(failed_counter)
    
    return np.array(pos_list), np.array(momenta_list)
