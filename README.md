# QuantumPolyspectra: a Python Package for the Analysis and Simulation of Quantum Measurements 
by M. Sifft and D. Hägele

The QuantumPolyspectra package is open-source software for analyzing and simulating quantum measurements in terms of so-called quantum polyspectra. Here we refer to the polyspectra as second to fourth order spectra (powerspectrum, bispectrum, and 2D cut through trispectrum). The simulation of measurement traces (integration of the stochastic master equation) is implemented via the QuTiP toolbox whereas the calculation of polyspectra from Hamiltonians or measurements traces recorded in the lab is performed as described in [this paper](https://link.aps.org/doi/10.1103/PhysRevB.98.205143) and [this paper](https://arxiv.org/abs/2011.07992) which also shows the utilization of quantum polyspectra to extract Hamiltonian parameters from a quantum measurement. 

## Documentation
The package is divided into two parts: the **generation** module and the **analysis** module. 
### Generation Module
This module connects any measurement trace as defined by a time-independet stochastic master equation with its corresponding polyspectra. Notice that spectra can be inferred via an actual simulation of the measurement trace by integration of the SME or (much quicker) by directly evaluating the ???formulas as shown [here](https://link.aps.org/doi/10.1103/PhysRevB.98.205143). 
### Analysis Module
This module allows for a convenient calculation of polyspectra from any measurement performed in the laboratory using state-of-the-art cumulant estimators and window function. Error estimation is done automatically. All routines are implemented using the ArrayFire library which allows the code to run on any CPU and GPU (Nvidia and AMD). GPUs are highly recommended for measurement trace exceeding 3 GB (binary size, not as .csv). The module also comes with a helper function for the conversion between .csv files to .h5 files which are needed to run the routines. Moreover, it come pre-equiped with a function for the estimation of parameters of telegraph noise.

### Examples
Examples for every function of the package are currently added to the folder Examples

## Support
The development of the QuantumPolyspectra package is supported by the working group Spectroscopy of Condensed Matter of the Faculty of Physics and Astronomy at the Ruhr University Bochum.

## Dependencies
For the package multiple libraries are used for the numerics and displaying the results:
* NumPy
* SciPy
* Pandas
* Cachetools
* QuTiP
* MatPlotLib
* Plotly
* tqdm
* Numba
* Lmfit
* h5py
* ArrayFire
* labellines
