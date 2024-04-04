from DataStructures.core import NumpyData
from DataStructures.generateBox import generate_grid_box


def traj_to_pdb(dt_out,traj,filename,box,N,dt,steps,L,g,T):
    """Creates a pdb file from a simulated trajectory with used parameters
    dt_out:     int
                output frequency of the frames
    traj:       nump array
                positions of the particles for each timestep
    filename:   string
                desired name of the out file
    box:        class
                universe to create pdb file from
    N:          int
                number of atoms (not molecules)
    dt:         float
                timestep of the simulation
    steps:      int
                number of time steps of the simulation
    L:          float
                Box-Size
    g:          float
                grid size
    T:          float
                Temperature
    returns:    pdb file in path
    """

    for i in range(0,len(traj),dt_out):
        box.add_timestamp_to_trajectory(traj[i], 3)

    #save it
    path = 'Simulation_Results/'+filename+'_N='+str(N/3)+'_dt='+str(dt)+'_steps='+str(steps)+'_L='+str(L)+'_g='+str(g)+'_T='+str(T)+'.pdb'

    box.to_file(path)







