import numpy as np
import DataStructures.core as Data

def rotation_matrix(angle: float, direction:'str' ):
    """
    This will generate a rotation matrix, that will either rotate along the x, y or z axis
    :param angle: the rotation angle in radians
    :param direction: either 'x', 'y', 'z'
    :return: The [3,3] rotation matrix
    """
    if direction == 'x':
        return np.asarray([
            [1, 0, 0],
            [0,np.cos(angle), -np.sin(angle)],
            [0,np.sin(angle), np.cos(angle)]
        ])
    elif direction == 'y':
        return np.asarray([

            [ np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)]
        ])
    elif direction == 'z':
        return np.asarray([
            [np.cos(angle), -np.sin(angle),0],
            [np.sin(angle), np.cos(angle),0],
            [0, 0, 1]
        ])


def generate_grid_box(box_lengths: np.ndarray, grid_spacings: np.ndarray, bond_length: float, angle: float
                      , random_orientation: bool = False, T: float = 300):
    """
    This sets up a simulation box with equally spaced water molecules
    :param box_lengths: the box lengths in xyz koords
    :param grid_spacings: the spacings of the water molecules in xyz directions
    :param bond_length: the intended bond length
    :param angle: the bond angle (in radians)
    :param random_orientation: whether to use a random orientation
    :param T: temperature for random initial momenta (in K)
    :return: a NumpyData Object filled with water molecules
    """

    # set up a data object with the specified box lengths
    data_box = Data.NumpyData(box_lengths, T=T)

    # setup some random masses
    mass_dict = {
        1: 1.,
        8: 16.
    }
    data_box.setup_masses(mass_dict)

    # figure out how many water molecules fit in each direction:
    no_water_molecules = np.floor(box_lengths / grid_spacings).astype(int)

    # setup the displacements for the h atoms:
    # a small matrix to rotate one vector by the bond angle:
    angle_rotate_matrix = rotation_matrix(angle, 'y')
    h1_displacement = np.asarray([bond_length, 0, 0])
    h2_displacement = np.matmul(angle_rotate_matrix, h1_displacement)

    # put everything into a single array
    h2o_displacements = np.asarray([np.zeros(3),
                                    h1_displacement,
                                    h2_displacement])

    # iterate through the grid and place a water molecule at every point
    for x in range(no_water_molecules[0]):
        for y in range(no_water_molecules[1]):
            for z in range(no_water_molecules[2]):

                #o is at current koord
                current_koord = np.asarray([x,y,z]) * grid_spacings
                current_displacements = h2o_displacements

                if random_orientation:
                    # generate 3 rotation matrices for rotating along the axis with a random angle
                    random_angles = np.random.rand(3)*2*np.pi
                    xrot_matrix = rotation_matrix(random_angles[0], 'x')
                    yrot_matrix = rotation_matrix(random_angles[0], 'y')
                    zrot_matrix = rotation_matrix(random_angles[0], 'z')

                    #multiply them all together
                    tot_rot_mat = np.matmul(xrot_matrix, yrot_matrix.T)
                    tot_rot_mat = np.matmul(tot_rot_mat, zrot_matrix.T)

                    # and rotate the displacements
                    current_displacements = np.matmul(current_displacements, tot_rot_mat)

                h2o_koords = current_koord + current_displacements

                data_box.add_molecule(h2o_koords[0], h2o_koords[1], h2o_koords[2])

    return data_box
