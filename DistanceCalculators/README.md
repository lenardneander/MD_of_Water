# Distance Calculators
As the name suggests this contains some files with helpers for calculating interatomic distances and vectors.  

## Numpy Calculators
### numpy_vectors_distaces:
Quick numpy implementation of the N²-Distance calculation algorithm. (i. e. just calculating distances from every atom 
in the first positions array to every other atom in the second positions array)
### np_list_vectors_distances:
Same as above, just taking arrays of atom-pairs as its argument, so [n_list, n_atom1, 3] and [n_list, n_atom2, 3] 
instead of [n_atom1, 3] and [n_atom2, 3] as above. Calculates distances only between atoms at the same index in the first axis.
### np_molecule_distances_angles
Takes as input a [n_mols, 3, 3] array of watermolecule-atomic positions, first axis: molecule_idx, second:atoms, 
third:vectors. Returns again: vectors, distances (inside the individual molecules) as well as the bond angle in the molecules as a third array.

## Numba Calculators
Numba-parallelized version of the N²-Algorithm as well as numba parallelized functions to construct neighborlists and 
to calculate vectors and distances using them.
### numba_vectors_distances:
Same as np_vectors_distances just using numba parallelization.
### numba_make_neighborlist:
will create a neighborlist with a given verlet radius and also return the vectors, distances of the underlying 
N²-calculation.
### numba_neighborlist_vectors_distances
Calculates vectors, distances using a verlet-neighborlist.
