import numpy as np


def np_vectors_distances(positions1: np.ndarray, positions2: np.ndarray):
    """
    Note: I tried all of this using scipy.spatial.cdist, which got the distances around 10 times faster, but no idea,
    how to get the vectors through scipy :/ (The distance metrics will delete the vector signs :/)
    This calculates the vectors and distances between one set of atoms and a different set of atoms
    :param positions1: [n_atom1, 3] array of positions of the first set of atoms
    :param positions2: [n_atom2, 3] array of positions of the second set of atoms
    :return: vectors, distances, with vectors being a [n_atom1, n_atom2, 3] array of vectors, where i,j contains
    the vector from atom i in the first array
    to atom j in the second array
    and distances being a [n_atom, n_atom] matrix
    """
    # None will create a newaxis and through broadcasting, the subtraction will calculate all the vectors
    vectors = positions1[:,None,:] - positions2[None, :,:]

    return vectors
