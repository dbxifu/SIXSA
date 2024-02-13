# SIXSA
Simulation based Inference for X-ray Spectral Analysis package from Barret & Dupourqué (2024). This repository contains example scripts for reproducing some of the results of the paper, and allows interested users to test this approach on their own data.

# Install 
We recommend the users to start from a fresh Python 3.10 [conda environment](https://conda.io/projects/conda/en/latest/user-guide/install/index.html). 

```
conda create -n sixsa python=3.10
conda activate sixsa
```

Once the environment is set up, create a new directory and clone this GitHub repository.

```
git clone https://github.com/dbxifu/SIXSA
cd sixsa
```

Now, you can install the required dependencies using `poetry`

```
pip install poetry
poetry install --no-root
```

# Run the demo 

The demo scripts can be run to reproduce some figures from Barret & Dupourqué (2024). Be sure to run this in the `sixsa` environment

```
python run_mri.py # will run multiple round inference on a spectrum of 2000 counts.
python run_sri.py # will run single round inference on a spectrum of either 2000 or 20000 counts.
```
In this first release, it is possible to run single round inference and multiple round inference on reference simulated pha files. 
Only the restricted prior, based on the number of counts in the spectrum to lie within a range specified in the yml files is implemented.
Output files are generated in PDF format, as to evaluate the different steps of the processing. 
