import numpy as np
import integrator as ig
from DataStructures.core import NumpyData
from DataStructures.generateBox import generate_grid_box
import pbc




def ljp(dist: np.ndarray, A: float, B:float,p_max=np.inf, rc=1e8):
    """Determine the LJ-forces between all particles
    dist:       2d-array
                Distance-matrix between particle pairs (including pbcs)
    A:          float
                Parameter for long range force
    B:          float
                Parameter for short range force
                
    returns:    2d-array
                forces for each particle
    """
    
   
    N = len(dist)                                               #Dimension of array 
    energy = 0                            #Define force array
    

    for i in range(N-1):                                          #Loop over all matrix rows
        # get the upper triangle of distances in this row
        dist_row = dist[i, i+1:]
        rc_mask = dist_row > rc
        
        dist_row[rc_mask] = rc


        r_inv = 1 / dist_row  # calculate inverse distances
        r6_inv = r_inv**6                                       #distances**6
        energy +=  np.sum(r6_inv * (A *  r6_inv - B))  #Calculate forces according to derrivative of LJP

        
        

    return np.clip(energy, -p_max, +p_max)

def kinetic_energy(ps,ms):

    return np.sum(ps**2 / (2 * ms[:, None]) )


def coulomb_energy(dist: np.ndarray, C: float, qs: np.ndarray, rc=1e8):
    """Determine the Coulomb-forces between all particles
    dist:       2d-array
                Distance-matrix between particle pairs (including pbcs)
    C:          float
                Parameter for coulomb force
    qs:         array
                charges of each particle

    returns:    2d-array
                forces for each particle
                """

    N = len(dist)  # Dimension of array
    energy = 0

    for i in range(N-1):  # Loop over all matrix rows

        # get the upper triangle of distances in this row
        dist_row = dist[i, i+1:]
        rc_mask = dist_row > rc
        
        dist_row[rc_mask] = rc


        r_inv = 1 / dist_row  # calculate inverse distances
        num_zeros = (2 - (i % 3))                      #Set intramolecular coulomb forces to zero
        zeros = np.array([0] * num_zeros)
        charge_factor = qs[i] * qs[i + 1::]      
        charge_factor[0:num_zeros] = zeros
        energy += np.sum(C * charge_factor * r_inv)   # Calculate forces according to derrivative of coulomb law

       

    return energy



def bond_energy(ho_dists, k, eq_dist):
    """ Determine the bond forces for all water molecules
    ho_dists:   2D array of shape [n_mols, 2] of the distance of the oxygen and the two hydrogens
    k:          float
                Strength of the harmonic potential

    eq_dist:    float
                Equilibrium distance
         
    returns:    2d-array with dimensions (N, 3)
                Forces for each particle
    """
    return np.sum(1/2*k*(ho_dists - eq_dist)[:,:,None]**2)


def angle_energy(vect, k_theta, eq_angle):
    """ Determine the angle forces for all water molecules 

    vect:       3D array of shape [n_mol, 2, 3] containing the normed vectors from the hydrogens to the oxygens in mol i
                Vectors of H-O-H atoms (including pbcs)

    k_theta:    float
                Stiffness of angle

    eq_angle:   float
                Equilibrium angle
                  
    returns:    3D array with dimensions (N_mol, 3, 3)
                Forces for atom j in molecule i at index [i,j]
    """


    angle = np.arccos((vect[:,0]* vect[:,1]).sum(axis=-1))

    return np.sum(k_theta/2 * (angle-eq_angle)**2)


def output_energy(x,p,box,dt,A,B,C,k,eq_dist,k_theta,eq_angle,L,rcut):
    """returns energy from a simulation trajectory
    x:          numpy-array
                contains position of particles for different timesteps
    p:          numpy-array
                contains momenta of particles for different timesteps
    box:        class
                contains all informations of the system
    dt:         int
                output frequency
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
    returns:    numpy arrays
                returns E_kin, E_ljp, E_coul, E_bond, E_angle
    """
    ljp_list = []
    kin_list = []
    coul_list = []
    bond_list = []
    angle_list = []
    m = box.get_masses()
    qs = box.get_charges()

    for i in range (0,len(x),dt):
        dist_vec = pbc.pbc_distance(x[i],L)
        dist = np.linalg.norm(dist_vec,axis=2)
        ho_vecs, ho_dists = box.vecs_dists_to_ho_format(dist_vec, dist)
        bond_list.append(bond_energy(ho_dists,k,eq_dist))
        dist_ljp = np.linalg.norm(pbc.pbc_distance(x[i][::3],L),axis=2)
        ljp_list.append(ljp(dist_ljp,A,B,rc=rcut))  
        kin_list.append(kinetic_energy(p[i],m)) 
        coul_list.append( coulomb_energy(dist, C, qs, rc=rcut) )
        angle_list.append(angle_energy(ho_vecs,k_theta,eq_angle))


    return np.array(kin_list), np.array(ljp_list), np.array(coul_list), np.array(bond_list), np.array(angle_list)


def bond_angle_distribution(x,box,L,dt):

    bond_dist = []
    angle_list = []

    for i in range(0,len(x),dt):

        dist_vec = pbc.pbc_distance(x[i],L)
        dist = np.linalg.norm(dist_vec,axis=2)

        ho_vecs, ho_dists = box.vecs_dists_to_ho_format(dist_vec, dist)

        bond_dist.append(ho_dists)
        angle_list.append(np.arccos((ho_vecs[:,0] * ho_vecs[:,1]).sum(axis=-1)))
    

    return np.array(bond_dist).flatten(), np.array(angle_list).flatten()


    



