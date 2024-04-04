# Data handling

This folder is dedicated our DataStructures and Input generation.

## DataStructures
### NumpyData
An array based DataStructure Template storing information about the SimulationBox in the form of Dense Numpy Arrays.
Capable of Storing:  
cell information   
atomic positions, numbers and charges   
momenta  
molecule topology   
verlet-neighborlists   
#### Features:
Basic File IO in the form of pdb files, so store atomic positions, numbers, cell information and multiple timesteps.
as a file and read this info from a file. NOTE: Masses and charges cant be stored and have to be setup after reading from file.  
Calculating intermolecular distances and vectors through a numba-parallelized version of the basic N² Algorithm.  
Initializing and calculation of distances through a verlet-list.  
Calculating intramolecular distances, vectors and angles.  
A small ase-viewer to look at the waterbox. ^^  

## Input Generation
### generate_grid_box
This will setup a NumpyData Object with a box of equally spaced watermolecules all with the same bond lengths and bond angle.
Optionally, the watermolecules will be assigned a random orientation.
  
## Example Usage:
Have a look at example_NumpyData.ipynb