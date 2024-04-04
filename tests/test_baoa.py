from integrator import baoa_step
import numpy as np

def test_baoa():
    #setup parameters
    N = 5000
    x_0 = np.zeros((N, 1))
    p_0 = np.ones((N, 1))
    m = np.ones(N)
    dt = 0.1
    timesteps = int(300 // dt)

    starting_energy = N/2
    stability_threshold = starting_energy * 0.015

    #create some placeholder arrays
    x_result = []
    p_result = []
    energies_pot = []
    energies_kin = []

    #setup x and p
    x_tmp = x_0
    p_tmp = p_0

    # run a bunch of steps for the harmonic oscillator
    for i in range(timesteps):

        force = -2 * x_tmp
        x_tmp, p_tmp = baoa_step(dt, x_tmp, p_tmp, force, m)

        # get the energies here
        e_potential = (x_tmp ** 2).sum(axis=-1)
        e_kinetic = (1 / (2 * m[:, None]) * (p_tmp ** 2)).sum(axis=-1)

        # record everything
        x_result.append(x_tmp)
        p_result.append(p_tmp)
        energies_pot.append([e_potential])
        energies_kin.append([e_kinetic])

    x_result = np.asarray(x_result)
    p_result = np.asarray(p_result)
    energies_pot = np.squeeze(np.asarray(energies_pot))
    energies_kin = np.squeeze(np.asarray(energies_kin))

    # because the baoa p-values are half a timestep behind, we have to shift them:
    energies_pot[1:] = (energies_pot[1:] + energies_pot[:-1]) / 2
    energies_pot = energies_pot[1:]
    energies_kin = energies_kin[1:]

    #get the total energy of all particles
    total_energy = (energies_pot + energies_kin).sum(axis=-1)

    assert (total_energy.max() - total_energy.min()) < stability_threshold, print(f"energy max: {total_energy.max()} "
                                                                                  f"min: {total_energy.min()}")