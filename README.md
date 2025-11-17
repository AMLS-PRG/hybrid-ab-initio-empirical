# Hybrid ab initio and empirical machine learning models for the potential energy surface
### Authors: Pablo Peña-Cano, and Pablo M. Piaggi

Input and analysis files to reproduce simulations of manuscript "Hybrid ab initio and empirical machine learning models for the potential energy surface" by Peña-Cano and Piaggi.

## Brief description of folder contents:
* `molecular dynamics`: Input and output files of molecular dynamics simulations.
* `notebooks`: Jupyter notebooks for analysis and figure production.
* `training-data`: Ab initio and experimental data in format suitable for the DeePMD-JAX.
* `training-process`: Training scripts for the DeePMD-kit.

## Software setting up
Note: You need to first have **CUDA 12** installed for GPU support.

1. Download the repository.
```
git clone https://github.com/AMLS-PRG/hybrid-ab-initio-empirical
```
2. Install the DeePMD-JAX software.
```
cd hybrid-ab-initio-empirical
git clone https://github.com/AMLS-PRG/deepmd-jax
python -m venv .venv
source .venv/bin/activate
cd deepmd-jax
pip install --upgrade pip
pip install -e .
```

Python libraries required for the DeePMD-JAX software:
* JAX 0.6.2
* Flax 0.10.6
* Optax 0.2.5
* JAX MD 0.2.27
* ASE 3.26.0
* Matplotlib 3.10.7
* gpustat 1.1.1
* IPyKernel 7.1.0
* seaborn 0.13.2
* pandas 2.3.3

## Download ab initio data
For reproducing the simulations you need to download the ab initio datasets. You can find them in the following repositories:
* SCAN: 
* MB-pol: https://github.com/sigbjobo/Quantum-phase-diagram-of-water/tree/main/Model/TrainingSet/training

You may copy the dataset inside the folders in `training-data/ab-initio/`.

## Workflow
As explained in the manuscript, the training of a hybrid machine learning model requires performing multiple simulations. In the following we explain the procedure in three steps:

### Step 1. Train a model using only the ab initio data:
Enter the folder `training-process/SCAN/` (or `training-process/MB-pol/`) and execute the file `train.py`. This script obtains the data from the `training-data/ab-initio/SCAN/` folder.

After the training, the model is saved in the `model.pkl` file.

### Step 2. Run a molecular dynamics simulation and extract a dataset:
Enter one of the folders in `molecular-dynamics/ab-initio/SCAN/` and run a molecular dynamics simulations with the file `simulate.py`, which uses `molecular-dynamics/liquid-water.data` as the initial configuration of the system and the previously obtained `model.pkl` file.

When the simulation ends, we need to extract a dataset with the magnitudes that will be needed for training the hybrid model (i.e., system's box, atomic coordinates, energy and the observable). A Jupyter notebook (`extract_data.ipynb`) is prepared into the `dataset/` folder for a direct extraction of the data.

Note: You can find some Jupyter notebooks in `notebooks/sim-analysis/` that may help you analyze the output files of the simulations.

### Step 3. Train the hybrid model:
To train the hybrid model we use both the ab initio data (e.g., `training-data/ab-initio/SCAN/`) and the previous dataset (e.g., `molecular-dynamics/ab-initio/SCAN/300K/dataset/`). Enter one of the folders in `training-process/hybrid/` and
execute the file `train.py`.

Some important parameters of the hybrid model training are:
* `train_data_path`: A list with the ab initio data folders.
* `train_data_path_obs`: A string or a list with the dataset obtained in step 2. When fitting observables at multiple temperatures, this parameter needs to be a list which elements are also lists with the paths of data folders at a specific temperature.
* `temperautes`: A float or a list with temperature values in Kelvin.
* `target_observable`: A float or a string with the path of the target value of the observable. For multiple temperatures, a list with the taret values for its temperature. It is important that the units of the target are the same as the ones of the `train_data_path_obs` files.
* `batch_size_observable`: An integer with the size of the mini-batch used in the observable optimization part of the iteration during training.
* `obs_step_every`: An integer that specifies how often the observable optimization step is performed.

Ones the hybrid model is trained, we test its precision in a molecular dynamics simulation (`molecular-dynamics/hybrid/` folder).


Please e-mail me if you have trouble reproducing the results.