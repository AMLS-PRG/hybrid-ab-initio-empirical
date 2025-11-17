from deepmd_jax.md import Simulation, TrajDump, TrajDumpSimulation
import ase.io
import numpy as np
from ase import units
import os

atoms=ase.io.read('../../../../liquid-water.data', format="lammps-data")

atomic_numbers=atoms.get_array('type')
type_idx=np.array(atomic_numbers) - 1

box=np.array(atoms.get_cell())

initial_position=atoms.get_positions()

sim = TrajDumpSimulation(
    model_path='../../../../../training-process/SCAN/model.pkl',    # Has to be an 'energy' or 'dplr' model
    box=box,                           # Angstroms
    type_idx=type_idx,                 # here the index-element map (e.g. 0-Oxygen, 1-Hydrogen) must match the dataset used to train the model
    mass=[16, 2],    # Oxygen, Hydrogen
    routine='NPT',                     # 'NVE', 'NVT', 'NPT' (Nosé-Hoover)
    dt=0.5,                            # femtoseconds
    initial_position=initial_position, # Angstroms
    temperature=340,                   # Kelvin # type: ignore
    report_interval=100,               # Report every 100 steps
    pressure=1.0,
)

n_steps=20000000 # 10 ns

# print positions and velocities every 200 steps in xyz format
traj_file = "traj.xyz"
if os.path.exists(traj_file):
    os.remove(traj_file)
trajectory = sim.run(
            n_steps,
            TrajDump(atoms, traj_file,
                      200000, vel=True),
)
