
This script is developeded for the batch execution of equilibrium simulations, temperature-variant molecular dynamics (MD) simulations, and steered molecular dynamics (SMD) simulations under the GROMACS and CHARMM36 force fields, aimed at the rapid screening of protein molecular properties.

## Install dependencies
The script has been tested on Python 3.10.

`pip install -r requirements.txt`

## Usage

Place multiple PDB files and the script in the same folder, modify the config file to adjust the simulation parameters, and run `python batch_run.py` to initiate the execution.