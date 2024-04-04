import numpy as np
from numba import njit, prange


@njit(parallel=False)
def numba_isin(test_array: np.array, test_set: set):
    """
    A compiled helper to replace the slow np.isin method
    :param: test_array -> 1d array of values
    :param: test_set -> a (python) set of values to check against
    :return: mask, a 1d array of size values.size true, where array[i] is in set, false otherwise
    """

    mask = np.empty(test_array.shape[0], dtype=np.bool_)

    for i in prange(test_array.shape[0]):
        if test_array[i] in test_set:
            mask[i] = True
        else:
            mask[i] = False

    return mask

@njit(parallel=True, fastmath=True)
def numba_vectors_distances(positions1: np.ndarray, positions2: np.ndarray, pbcs: np.ndarray):
    """
    This calculates the vectors between two sets of atoms using more cpu cores.
    :param positions1: [n_atom1, 3] array of positions of the first set of atoms
    :param positions2: [n_atom2, 3] array of positions of the second set of atoms
    :param pbcs: float of the box_sizes
    :return: vectors, with vectors being a [n_atom1, n_atom2, 3] array of vectors, where i,j contains
    the vector from atom i in the first array to atom j in the second array, (only  the upper triangle is calculated)
    """

    # get some basic infos first
    natoms1 = positions1.shape[0]
    natoms2 = positions2.shape[0]
    dimensions = positions1.shape[1]

    # empty arrays for the results
    vectors = np.zeros((natoms1, natoms2, dimensions))
    distances = np.zeros((natoms1, natoms2))

    for i in prange(natoms1):
        # get the vectors row wise
        tmp_vecs = positions2[(i + 1):] - positions1[i]

        # apply pbcs
        tmp_vecs = np.where(tmp_vecs > (pbcs / 2), tmp_vecs - pbcs, tmp_vecs)
        tmp_vecs = np.where(tmp_vecs < (-pbcs / 2), tmp_vecs + pbcs, tmp_vecs)

        # fill in the results
        vectors[i, (i + 1):] = tmp_vecs
        distances[i, (i+1):] = np.sqrt((tmp_vecs**2).sum(axis=-1))

    return vectors, distances


@njit(parallel=False, fastmath=True)
def numba_make_neighborlist(positions, verlet_radius, molecule_idxs, pbcs):
    """
    WARNING: This is deprecated as its not capable of dealing with pbcs!
    Creates a neighborlist (as an array of indices) and also returns the vectors and distances of the underlying
    calculation
    :param positions: a [n_atom, 3] np ndarray with atomic positions
    :param verlet_radius: the intended verlet radius
    :return: neighbor_list, vectors, distances
    with neighbor_list as a [n_neighbor_pairs, 2] np ndarray containing the indices of neighbor atoms
    vectors as the vectors between the neighbors as a [n_atom, n_atom, 3] array
    and distances as a [n-atom, n_atom] array with distances between the atoms
    """
    # get distances first
    vectors, distances = numba_vectors_distances(positions, positions, pbcs)

    # numba doesnt like literals?
    two = 2

    # make some empty arrays for results
    neighbor_list = np.zeros((distances.shape[0] ** 2, two), dtype='i8')
    idx_mask = np.arange(distances.shape[0])
    current_no_idx_pairs = 0

    # for all atoms find partners in upper triangle and fill the atomic indices into the neighbor_list
    for i in range(distances.shape[0]):
        partner_idxs = idx_mask[(i + 1):][distances[i, (i + 1):] <= verlet_radius]

        # find mol
        mol_idx = int(i / 3)
        mol_atom_set = set(molecule_idxs[mol_idx])

        # disselect molecule combinations
        mol_atom_mask = numba_isin(partner_idxs, mol_atom_set)

        partner_idxs = partner_idxs[~(mol_atom_mask)]

        # make the neighborlist pairs and fill them in
        no_partners = partner_idxs.shape[0]

        neighbor_list[current_no_idx_pairs:current_no_idx_pairs + no_partners, 1] = partner_idxs
        neighbor_list[current_no_idx_pairs:current_no_idx_pairs + no_partners, 0] = i

        # increment to keep track of the total number of pairs
        current_no_idx_pairs += no_partners

    # cut out the nonexistent parts of the list
    neighbor_list = neighbor_list[:current_no_idx_pairs]

    return neighbor_list


@njit(parallel=True, fastmath=True)
def numba_neighborlist_vectors_distances(positions: np.ndarray, neighborlist_idxs: np.ndarray, pbcs: np.ndarray):
    """
    Calculates vectors and distances based on a verlet neighbor list
    :param positions: [n_atom, 3] np.ndarray containing the atomic positions
    :param neighborlist_idxs: [n_neighbor_pais, 2] np.ndarray containing indices of neighbors
    :param pbcs: float of the box_sizes
    :return:
    """
    # get some general info first
    niter = neighborlist_idxs.shape[0]
    natoms = positions.shape[0]
    dims = positions.shape[1]

    # number of vectors to compute in each (parallel) for loop
    vecs_per_iter = 1000

    # number of for loops
    for_iter = niter // vecs_per_iter + 1

    # empty array for results
    vectors = np.zeros((neighborlist_idxs.shape[0], dims))
    distances = np.zeros((neighborlist_idxs.shape[0]))

    # somehow splitting the positions into 2 array speeds things up
    x_posis = positions[neighborlist_idxs[:, 0]]
    y_posis = positions[neighborlist_idxs[:, 1]]

    # iterate through all atoms
    for i in prange(for_iter):

        # current instance starting index
        idx = i * vecs_per_iter
        # number of vecs for this loop instance
        offset = vecs_per_iter

        # careful with array borders
        if i == for_iter - 1:
            offset = niter - idx

        # calculate all vectors and distances in one batch, np vectorized ops are fast
        tmp_vecs = y_posis[idx:idx + offset] - x_posis[idx:idx + offset]

        # apply pbcs
        tmp_vecs = np.where(tmp_vecs > (pbcs / 2), tmp_vecs - pbcs, tmp_vecs)
        tmp_vecs = np.where(tmp_vecs < (-pbcs / 2), tmp_vecs  + pbcs, tmp_vecs)

        vectors[idx:idx+offset] = tmp_vecs
        distances[idx:idx+offset] = np.sqrt((tmp_vecs**2).sum(axis=-1))

    return vectors, distances

@njit(parallel=True)
def nbls_to_oxy_format(nbls_vecs: np.ndarray, nbls_dists: np.ndarray, oxy_idxs: np.ndarray,
                      neighborlist_idxs: np.ndarray):
    """
    A utility function to convert from a neighborlist style vector/distance representation to a dense array
    of oxygen vectors, distances
    (This function is also just here because numba doesnt like attributes of custom classes)
    :param nbls_vecs: a array of shape [n_nbls_pairs, 3] containing all the vectors with the nbls
    :param nbls_dists: a array of shape [n_nbls_pairs] containing all the distances with the nbls
    :param oxy_idxs: a array of shape [n_oxy_atoms] containing the indices of oxygen atoms
    :param neighborlist_idxs: a array of shape [n_nbls_pairs, 2] containing the indices of atom pairs the nbls vecs
    / dists belong to
    :return: oxy_vecs, oxy_dists
            oxy_vecs: np.ndarray of shape [n_oxy_atoms, n_oxy_atoms, 3] with entry [i,j] being the vector pointing from
                        oxygen atom i to oxygen atom j (only the upper triangle of the matrix tho)
            oxy_dists: np.ndarray of shape [n_oxy_atoms, n_oxy_atoms] with entry [i,j] being the distance between
                        oxygen atom i and oxygen atom j (only the upper triangle of the matrix tho)
            """

    no_oxys = oxy_idxs.shape[0]
    no_pairs = nbls_vecs.shape[0]

    oxy_set = set(oxy_idxs)

    # fill them all into an empty array
    oxy_vectors = np.zeros((no_oxys, no_oxys, 3))
    oxy_dists = np.zeros((no_oxys, no_oxys))

    for i in prange(no_pairs):
        if neighborlist_idxs[i,0] in oxy_set and neighborlist_idxs[i,1] in oxy_set:
            x_idx = neighborlist_idxs[i, 0]//3
            y_idx = neighborlist_idxs[i, 1]//3
            oxy_vectors[x_idx, y_idx] = nbls_vecs[i]
            oxy_dists[x_idx, y_idx] = nbls_dists[i]

    return oxy_vectors, oxy_dists


@njit(parallel=True)
def get_ho_vectors_dists(positions:np.ndarray, molecule_idxs: np.ndarray, pbcs:float):
    """
    A utility function to calculate the intramolecular vectors/distances pointing from o to the two hydrogens
    :param: positions: [n_atom, 3] array of atomic positions
    :param: molecule_idxs: [n_mol, 3] array with the indices of the atoms in molecule i at index i
                            (the atoms have to be ordered O, H1, H2)
    :param: pbcs: The box size (only supports cube boxes)
    :return: ho_vecs, ho_dists
            with ho_vecs as a np.ndarray of shape [n_mol, 2, 3] with entry [i,j] being the vector pointing to
                        hydrogen j in molecule i
            and ho_dists: np.ndarray of shape [n_mol, 2] with the distances at the same place as with ho_vecs
    """

    # make some placeholders
    ho_vecs = np.zeros((molecule_idxs.shape[0], 2, 3))

    # get some indices
    oxy_indices = molecule_idxs[:, 0]
    h1_indices = molecule_idxs[:, 1]
    h2_indices = molecule_idxs[:, 2]

    # calculate the vectors
    ho_vecs[:, 0] = (positions[oxy_indices] - positions[h1_indices])
    ho_vecs[:, 1] = (positions[oxy_indices] - positions[h2_indices])

    # apply pbcs
    ho_vecs = np.where(ho_vecs > (pbcs / 2), ho_vecs - pbcs, ho_vecs)
    ho_vecs = np.where(ho_vecs < (-pbcs / 2), ho_vecs + pbcs, ho_vecs)

    #calc the distances
    ho_dists = np.sqrt((ho_vecs**2).sum(axis=-1))

    return ho_vecs, ho_dists