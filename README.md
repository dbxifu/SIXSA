# SIXSA
Simulation based Inference for X-ray Spectral Analysis package from [Barret & Dupourqué (2024, Astronomy and Astrophysics, in press](https://ui.adsabs.harvard.edu/abs/2024arXiv240106061B/abstract)). This repository contains example scripts for reproducing some of the results of the paper, and allows interested users to test this approach on their own data.
The core of the python scripts are build upon the [sbi](https://sbi-dev.github.io/sbi/) python package. The simulations are performed with an early release of [jaxspec](https://jaxspec.readthedocs.io/en/latest/) (Dupourqué et al., A&A, in preparation).
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

The demo scripts can be run to reproduce figures from [Barret & Dupourqué (2024, A&A, in press)](https://ui.adsabs.harvard.edu/abs/2024arXiv240106061B/abstract), to get a feel of the power of SBI-NPE. Be sure to run this in the `sixsa` environment

```
python SIXSA_CODES/run_demo
```
In this first release, it is possible to run single round inference and multiple round inference on some reference 
simulated pha files. The run will produce a set of PDF files that are available in SIXSA_OUTPUTS. In running single 
round inference, the code will generate the posteriors for a set of 500 spectra. The code will also produce a pickle file 
that saves the results of the run, that can be uploaded for re-use.
The inputs for each run is defined in the yaml files provided (SIXSA_YML_INPUT_FILES). You can change the parameters of the input files, 
but beware that reducing too much the training sample size may introduce erratic behavior. There is always a 
gain in training the network with sufficient samples.
Remember that there are random realizations involved and therefore the results from one run to the other can differ 
(still within errors). 

The run times can also differ from one round to the other, but stay reasonably similar. 
Note that time listed in the paper have been obtained by running the code on a 2.9 GHz 6-Core Intel Core i9 MacBook Pro. 
We are working on ways to speed up the code, while increasing its efficiency.
