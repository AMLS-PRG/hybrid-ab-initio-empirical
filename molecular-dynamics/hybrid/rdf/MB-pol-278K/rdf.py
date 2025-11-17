import numpy as np
import matplotlib.pyplot as plt
import ase.io as aseio
from ase import Atoms
from ase.geometry.analysis import Analysis
from ase.neighborlist import neighbor_list
from ase.visualize import view
import os

filename = "traj.xyz"
traj = aseio.iread(filename)
configs=[]
for atoms in traj:
    configs.append(atoms)

def compute_rdf(atoms: Atoms,
                        species: str = "O",
                        r_min: float = 0.0,
                        r_max: float = 10.0,
                        dr: float = 0.1,
                        pbc: bool = True):
    """
    Compute the partial RDF g_{species-species}(r) for an ASE Atoms object.
    
    Parameters
    ----------
    atoms : Atoms
        ASE Atoms object containing positions and cell information.
    species : str
        Chemical symbol of the species (e.g. "O") for which to compute the partial RDF.
    r_min : float
        Minimum distance to start the histogram (Å).
    r_max : float
        Maximum distance to end the histogram (Å).
    dr : float
        Width of the histogram bins (Å).
    pbc : bool
        Whether to apply minimum-image convention (periodic boundary conditions).
    
    Returns
    -------
    r : np.ndarray
        Bin centers array.
    rdf : np.ndarray
        Computed partial RDF values for species-species.
    """
    # Encuentra índices de átomos del tipo deseado
    syms = atoms.get_chemical_symbols()
    idx = [i for i, s in enumerate(syms) if s == species]
    if len(idx) < 2:
        raise ValueError(f"No hay suficientes átomos de tipo '{species}' para calcular RDF.")
    
    # Calcula distancias entre esos átomos
    # get_all_distances devuelve matriz NxN; extraemos solo filas y columnas de 'idx'
    D = atoms.get_all_distances(mic=pbc)
    # Tome pares i<j dentro de idx
    pairs = [(i, j) for i in idx for j in idx if i < j]
    d = np.array([D[i, j] for i, j in pairs])
    
    # Histograma
    edges = np.arange(r_min, r_max + dr, dr)
    hist, edges = np.histogram(d, bins=edges)
    r = 0.5 * (edges[:-1] + edges[1:])
    
    # Densidad parcial ρ = N_species / V
    V = atoms.get_volume()
    N_sp = len(idx)
    rho_sp = N_sp / V
    
    # Volumen de concha esférica
    shell_vol = 4.0 * np.pi * r**2 * dr
    
    # Normalización
    norm = (N_sp * rho_sp * shell_vol) / 2.0
    
    # RDF parcial
    rdf = hist / norm
    
    return r, rdf

rdf_OO = []
for conf in configs:
    dummy = compute_rdf(conf, species="O", r_min=2.245, r_max=5.005, dr=0.01)
    rdf_OO.append(dummy[1])
r = dummy[0]

os.makedirs("rdf", exist_ok=True)
np.save("rdf/r.npy", r)
np.save("rdf/rdf_OO.npy", rdf_OO)