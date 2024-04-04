from ase import Atoms
from ase.visualize import view
from DistanceCalculators.numba_calculators import *
import mdtraj as md
from numba import jit, prange

import numpy as np


class DataTemplate:

    def __init__(self, cell: np.ndarray, T: float = 300):
        """
        This is just a template initializing the simulation box and setting up everything else as None
        :param cell: a np.array with 3 float entries specifying the 3 box lengths (let`s stick to cubic cells no?)
        :param T:    temperature in Kelvin
        """

        self.cell = cell

        # these are all meant to be filled in later when adding molecules
        self.positions = None
        self.momenta = None
        self.elements = None
        self.charges = None
        self.masses = None

        # this is maybe for avoiding errors
        self.charge_dict = None
        self.mass_dict = None

        # these are meant to contain indices
        self.molecule_idxs = None
        self.atom_idxs = None

        # Neighborlist things
        self.neighbor_list_init = False
        self.verlet_radius = None
        self.force_radius = None
        self.neighborlist_idxs = None
        self.last_nbls_update_posis = None

        # topology and trajectory objects for MD Traj saving and loading
        self.topology = None
        self.trajectory = None

        # temperature
        self.T = T

class NumpyData(DataTemplate):
    """
    Our very first DataStructure ^^
    """

    # just copy the init from the template
    def __init__(self, cell: np.ndarray, T: float = 300):
        super().__init__(cell, T=T)

    def add_molecule(self, o_position: np.ndarray, h1_position: np.ndarray, h2_position: np.ndarray,
                     o_mom=None, h1_mom=None, h2_mom=None):
        """
        Add a water molecule
        :param o_position: [3] np.array with xyz koords for O
        :param h1_position: [3] np.array with xyz koords for H1
        :param h2_position: [3] np.array with xyz koords for H2
        :param o_mom: optional momentum vector for O
        :param h1_mom: optional momentum vector for H1
        :param h2_mom: optional momentum vector for H2
        :return: Niente
        """

        # first check whether we have already added some molecule
        if self.atom_idxs is None:
            max_current_at_idx = -1
        else:
            max_current_at_idx = self.atom_idxs.max()

        # setup indices for the new molecule
        new_idxs = np.arange(3) + max_current_at_idx + 1

        # generate arrays for the atomic numbers, positions and momenta of our 3 new atoms
        new_elements = np.array([8, 1, 1])
        new_positions = np.asarray([o_position, h1_position, h2_position]).astype(float)

        if o_mom is None:
            # set o_velo to a value drawn from the boltzmann dist
            o_mom = np.random.normal(0, 0.274 * np.sqrt(self.T), size=(3)) * self.mass_dict[8.]

        # same for the other hydrogens
        if h1_mom is None:
            h1_mom = np.random.normal(0, 0.274 * np.sqrt(self.T), size=(3)) * self.mass_dict[1.]
        if h2_mom is None:
            h2_mom = np.random.normal(0, 0.274 * np.sqrt(self.T), size=(3)) * self.mass_dict[1.]

        new_momenta = np.asarray([o_mom, h1_mom, h2_mom]).astype(float)

        if self.positions is None:
            # if we haven`t initialized all the lists yet, we will have to do so know:
            self.elements = new_elements
            self.positions = new_positions
            self.atom_idxs = new_idxs
            self.molecule_idxs = new_idxs[None, :]
            self.momenta = new_momenta
        else:
            # otherwise just append the new info to the old lists
            self.elements = np.concatenate((self.elements, new_elements))
            self.positions = np.concatenate((self.positions, new_positions))
            self.momenta = np.concatenate((self.momenta, new_momenta))
            self.atom_idxs = np.concatenate((self.atom_idxs, new_idxs))
            self.molecule_idxs = np.concatenate((self.molecule_idxs, new_idxs[None, :]))

        # To avoid errors if you add molecules after setting up the charges, let the code automatically setup charges,
        # if we supplied a charge dict earlier
        if self.charge_dict is not None:
            self.setup_charges(self.charge_dict)

        if self.mass_dict is not None:
            self.setup_masses(self.mass_dict)

        self.mk_mdtraj_topology()

        trajectory = md.Trajectory(self.positions, topology=self.topology, unitcell_angles=[90, 90, 90],
                             unitcell_lengths=self.cell)
        self.trajectory = trajectory


    def setup_charges(self, charge_dict: dict):
        """
        This is meant to setup a list of partial charges for the atoms
        :param charge_dict: A dict containing {atomic_number:partial_charge}-pairs.
        :return: None
        """

        # store the dict internally
        self.charge_dict = charge_dict

        if self.elements is not None:
            # set up an array containing the charges if we already have the elements pinned down
            charges = np.zeros_like(self.elements, dtype=float)
            for key in charge_dict.keys():
                charges[self.elements == key] = charge_dict[key]

            # assign them internally
            self.charges = charges

    def setup_masses(self, mass_dict: dict):
        """
        This is meant to setup a list of masses for the atoms
        :param mass_dict: A dict containing {atomic_number:mass}-pairs.
        :return: None
        """

        # store the dict internally
        self.mass_dict = mass_dict

        if self.elements is not None:
            # set up an array containing the masses if we already have the elements pinned down
            masses = np.zeros_like(self.elements, dtype=float)
            for key in mass_dict.keys():
                masses[self.elements == key] = mass_dict[key]

            # assign them internally
            self.masses = masses

    def get_vectors(self, pbcs: bool=False):
        """
        :param: pbcs, whether to use pbcs
        :return: np.ndarray of shape [n_atoms, n_atoms, 3] with the vector pointing from atom i to j at index i,
        """

        if not pbcs:
            pbcs = 1e6
        else:
            pbcs = self.cell[0]

        #get the vectors between atoms without pbcs
        vectors, distances = numba_vectors_distances(self.positions, self.positions, pbcs=pbcs)

        return vectors

    def vecs_dists_to_oxy_format(self, all_vecs: np.ndarray, all_dists: np.ndarray):
        """
        Selects only the oxygen atoms (for the LJ computation) from the all atom matrices
        :param all_vecs: a array of shape [n_atoms, n_atoms, 3] containing all the vectors from atom i to j
        :param all_dists: a array of shape [n_atoms, n_atoms] containing all the distances
        :return: oxy_vecs, oxy_dists
        where oxy_vecs is a [n_oxy_atom, n_oxy_atom, 3] array with all the oxygen atoms selcted
        and oxy_dists is the same for the distances
        """
        # get a mask and reindex everything
        o_idxs = np.argwhere(self.elements == 8).flatten()
        x_idxs, y_idxs = np.meshgrid(o_idxs, o_idxs)

        o_vecs = all_vecs[y_idxs, x_idxs, :]
        o_dists = all_dists[y_idxs, x_idxs]

        return o_vecs, o_dists

    def vecs_dists_to_ho_format(self, all_vecs: np.ndarray, all_dists: np.ndarray):
        """
        Selects only the HO vectors (for the intramolecular computation) from the all atom matrices
        :param all_vecs: a array of shape [n_atoms, n_atoms, 3] containing all the vectors from atom i to j
        :param all_dists: a array of shape [n_atoms, n_atoms] containing all the distances
        :return: ho_vecs, ho_dists
        where ho_vecs is a [n_molecules, 2, 3] array with the vector from Hj to O of molecule i
        at index ho_vecs[i,j] (j in 0,1)
        ho_dists is the same, just with the dists of these
        """

        # get the indices first
        mol_os = self.molecule_idxs[:,0]
        mol_h1s = self.molecule_idxs[:,1]
        mol_h2s = self.molecule_idxs[:,2]

        # fill them all into an empty array
        ho_vectors = np.zeros((self.molecule_idxs.shape[0], 2,3))
        ho_vectors[:, 0] = -all_vecs[mol_os, mol_h1s, :]
        ho_vectors[:, 1] = -all_vecs[mol_os, mol_h2s, :]

        # same for the dists
        ho_dists = np.zeros((self.molecule_idxs.shape[0], 2))
        ho_dists[:, 0] = all_dists[mol_os, mol_h1s]
        ho_dists[:, 1] = all_dists[mol_os, mol_h2s]

        return ho_vectors, ho_dists

    def from_oxygen_format(self, array: np.ndarray):
        """
        This is a helper that turns a [n_oxy_atoms,...] array and only returns a zero padded [n_atom, ...] array
        (containing the oxygen entries in the right place)
        :param array: a [n_oxy_atoms, ...] np.ndarray
        :return: a [n_atom,  ...] array
        """
        result = np.zeros((self.elements.shape[0], *array.shape[1:]))
        result[self.elements == 8] = array
        return result

    def from_mol_format(self, array: np.ndarray):
        """
        This is a helper that takes a [n_mols, 3, ...] array and turns it into the all atom format
        :param array: a [n_atom, 3, ...] np.ndarray containing the entries of O,H1,H2 in that order (along the second axis)
        :return: a [n_atoms, ...] array containing the entries at the right entries
        """
        return array.reshape([-1, *array.shape[2:]])[self.molecule_idxs.flatten()]

    def get_nbls_vectors_distances(self, verlet_radius: float, force_radius: float):
        """
        :param verlet_radius: The neighborlist radius
        :param force_radius: The cutoff radius of the forces
        :return: np.ndarray of shape [n_atoms, n_atoms, 3] with the vector pointing from atom i to j at index i,
        """

        # if there is no neighborlist, we need to init some things
        if not self.neighbor_list_init:
            self.neighborlist_idxs = numba_make_neighborlist(self.positions, verlet_radius, self.molecule_idxs, self.cell[0])
            self.last_nbls_update_posis = self.positions
            self.neighbor_list_init = True

        # check for penetrated verlet skin and rebuild the neighborlist if necessary
        if self.nbls_skin_penetrated(verlet_radius, force_radius):
            self.neighborlist_idxs = numba_make_neighborlist(self.positions, verlet_radius, self.molecule_idxs, self.cell[0])
            self.last_nbls_update_posis = self.positions

        vectors, distances = numba_neighborlist_vectors_distances(self.positions, self.neighborlist_idxs, self.cell[0])

        return vectors, distances, self.neighborlist_idxs

    def nbls_skin_penetrated(self, verlet_radius: float, force_radius: float):
        """
        :param verlet_radius: The neighborlist radius
        :param force_radius: The cutoff radius of the forces
        :return: bool, whether the neighborlist skin has been penetrated
        """
        # calc the distance travelled after the last update:
        vec_displacements = self.positions - self.last_nbls_update_posis
        displacement_lengths = np.sqrt((vec_displacements**2).sum(axis=-1))

        # get the skin radius
        skin_radius = verlet_radius - force_radius

        # return true if any displacement is larger than this.
        return (displacement_lengths > skin_radius).any()

    def get_ho_vectors(self):
        """
        A utility function to calculate the intramolecular vectors/distances pointing from o to the two hydrogens
        :return: ho_vecs, ho_dists
                with ho_vecs as a np.ndarray of shape [n_mol, 2, 3] with entry [i,j] being the vector pointing to
                            hydrogen j in molecule i
                and ho_dists: np.ndarray of shape [n_mol, 2] with the distances at the same place as with ho_vecs
        """

        return get_ho_vectors_dists(self.positions, self.molecule_idxs, self.cell[0])

    def nbls_to_oxy_format(self, nbls_vecs: np.ndarray, nbls_dists: np.ndarray):
        """
        A utility function to convert from a neighborlist style vector/distance representation to a dense array
                    of oxygen vectors, distances
        :param nbls_vecs: a array of shape [n_nbls_pairs, 3] containing all the vectors with the nbls
        :param all_dists: a array of shape [n_nbls_pairs] containing all the distances with the nbls
        :return: oxy_vecs, oxy_dists
        oxy_vecs: np.ndarray of shape [n_oxy_atoms, n_oxy_atoms, 3] with entry [i,j] being the vector pointing from
                    oxygen atom i to oxygen atom j (only the upper triangle of the matrix tho)
        oxy_dists: np.ndarray of shape [n_oxy_atoms, n_oxy_atoms] with entry [i,j] being the distance between
                    oxygen atom i and oxygen atom j (only the upper triangle of the matrix tho)
        """

        oxy_idxs = self.atom_idxs[self.elements == 8]

        return nbls_to_oxy_format(nbls_vecs, nbls_dists, oxy_idxs, self.neighborlist_idxs)

    def view(self):
        """
        Opens a small viewer with the current box
        :return: Niente
        """
        atoms = Atoms(self.elements, self.positions, cell=self.cell)

        view(atoms)

    def get_molecule_positions(self):
        """
        :return: mol_positions
        where mol_positions is a [n_mol, 3, 3] array where the i, jth vector corresponds to the position of atom j of
        molecule i
        """

        # molecule_idxs is handy for fancy indexing
        return self.positions[self.molecule_idxs]

    def get_elements(self):
        """
        :return: array with the atomic numbers
        """
        return self.elements

    def get_masses(self):
        """
        :return: array with the atomic masses
        """
        return self.masses

    def get_charges(self):
        """
        :return: array with the atomic charges
        """
        return self.charges

    def get_positions(self):
        """
        :return: array with the atomic positions
        """
        return self.positions

    def get_momenta(self):
        """
        :return: array with the atomic momenta
        """
        return self.momenta

    def set_positions(self, new_positions: np.ndarray):
        """
        :param new_positions: the new atomic positions
        :return:
        """
        self.positions = new_positions

    def set_momenta(self, new_momenta: np.ndarray):
        """
        :param new_momenta: the new atomic momenta
        :return:
        """
        self.momenta = new_momenta

    def to_file(self, filename: str):
        """
        This will write the current waterbox to a gromacs gro file using MDanalysis
        :param filename: A gro file (and path) to write to
        :return:
        """

        # we have to make some slight modifications because of unit conversions (ie AA to nm)
        # and also shift the atoms because pdb files go from 0 to L whereas we go from -L/2 to L/2
        trajectory = self.trajectory
        trajectory.xyz /= 10
        trajectory.unitcell_lengths /= 10.
        trajectory.save_pdb(filename)

    def from_file(self, filename: str, mass_dict: dict, charge_dict: dict, frame: int = 0):
        """
        This will fill in all data from a gromacs pdb file using MDTraj
        :param filename: A file (and path) from where to read the data
        :param frame: the index of the timeframe to use
        :param mass_dict: a dict containing the atomic masses (ie. atomic number : mass)
        :param charge_dict: the same, but with the charges
        :return: Niente
        """

        trajectory = md.load_pdb(filename, no_boxchk=True)

        self.mass_dict = mass_dict
        self.charge_dict = charge_dict

        print(f'PDB file with {trajectory.n_frames} frames, {trajectory.n_residues} molecules and {trajectory.n_atoms}'
              f' atoms. \n Selecting Frame {frame} at t = {trajectory[frame].time} ')

        positions = trajectory.xyz[frame]*10
        self.cell = trajectory.unitcell_lengths[0]*10

        for molecule in trajectory.topology.residues:
            h_indices = []
            o_index = 0
            for atom in molecule.atoms:
                if atom.element.number == 1:
                    h_indices.append(atom.serial)
                if atom.element.number == 8:
                    o_index = atom.serial
            self.add_molecule(positions[o_index], positions[h_indices[0]], positions[h_indices[1]])

        self.trajectory = trajectory
        self.topology = trajectory.topology

    def mk_mdtraj_topology(self):
        """
        This just setups a MD-Traj topology object with our current box.
        :return: Niente
        """
        atom_idxs = self.atom_idxs
        mol_idxs = self.molecule_idxs
        elements = self.elements

        n_mols = mol_idxs.shape[0]

        topology = md.Topology()

        only_chain = topology.add_chain()

        for i in range(n_mols):
            tmp_residue = topology.add_residue(f'H2O_', only_chain, resSeq=i)
            atom_list = []
            for j in range(3):
                atom_idx = mol_idxs[i, j]
                atomic_number = elements[atom_idxs == atom_idx][0]
                element = md.element.Element.getByAtomicNumber(atomic_number)
                tmp_atom = topology.add_atom(str(atom_idx), element=element, residue=tmp_residue, serial=atom_idx)
                atom_list.append(tmp_atom)

                if atomic_number == 8:
                    oxygen_index = j

            for j in range(3):
                if j == oxygen_index:
                    continue
                topology.add_bond(atom_list[j], atom_list[oxygen_index], order=1)

        self.topology = topology

    def to_mdtraj(self):
        """
        Sets up a MD Traj Trajectory with our current Trajectory
        :return: A MD Traj Trajectory with fixed topology
        """

        trajectory = md.Trajectory(self.positions, topology=self.topology, unitcell_angles=[90, 90, 90],
                             unitcell_lengths=self.cell)
        self.trajectory = trajectory

        return trajectory

    def add_timestamp_to_trajectory(self, new_positions: np.ndarray, time: float):
        """
        Adds positions to the MD Traj Trajectory (for saving them)
        :param new_positions: np.ndarray (n_atom, 3) containing the new positions
        :param time: a float representing the simulation time of this timestamp
        :return: Niente
        """
        extra_timestep = self.trajectory[0]
        extra_timestep.xyz[0] = new_positions
        extra_timestep.time = np.asarray([time])

        self.trajectory = self.trajectory + extra_timestep