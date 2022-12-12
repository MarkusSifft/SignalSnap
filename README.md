# SignalSnap: Signal Analysis Made Easy 
by M. Sifft and D. Hägele

The SignalSnap package is open-source software for analyzing signals in the spectral domain. Here, a few outstanding 
features of SignalSnap:
* Calculation of a powerspectrum within a few lines of code
* Errors of spectral values are automatically calculated 
* Calculation of higher-order spectra or polyspectra (3rd and 4th order) (+ error estimation) in a few lines of code
* Support for just-in-time import hdf data (dataset does not have to fit in RAM)
* All calculation can be performed on GPU (see Arrayfire) 
* Advanced plotting options for two-dimensional higher-order spectra 
* Usage of unbiased estimators for higher-order cumulants (see Literature)
* Efficient implementation of the confined Gaussian window for an optimal time-bandwidth product (see Literature)
* Special functions for the verification of the stationarity of a signal

## Documentation
A comprehensive documentation of SignalSnap will follow soon. 
### Generation Module
This module connects any measurement trace as defined by a time-independent stochastic master equation with its corresponding polyspectra. Notice that spectra can be inferred via an actual simulation of the measurement trace by integration of the SME or (much quicker) by directly evaluating the ???formulas as shown [here](https://link.aps.org/doi/10.1103/PhysRevB.98.205143). 
### Analysis Module
This module allows for a convenient calculation of polyspectra from any measurement performed in the laboratory using state-of-the-art cumulant estimators and window function. Error estimation is done automatically. All routines are implemented using the ArrayFire library which allows the code to run on any CPU and GPU (Nvidia and AMD). GPUs are highly recommended for measurement trace exceeding 3 GB (binary size, not as .csv). The module also comes with a helper function for the conversion between .csv files to .h5 files which are needed to run the routines. Moreover, it come pre-equiped with a function for the estimation of parameters of telegraph noise.

### Examples
Examples for every function of the package are currently added to the folder Examples

## Support
The development of the SignalSnap package is supported by the working group Spectroscopy of Condensed Matter of the 
Faculty of Physics and Astronomy at the Ruhr University Bochum.

## Dependencies
For the package multiple libraries are used for the numerics and displaying the results:
* NumPy
* SciPy
* MatPlotLib
* tqdm
* Numba
* h5py
* ArrayFire

## Literature
Unbiased estimators are used for the calculation of higher-order cumulants. Their derivation can be found in
[this paper](https://arxiv.org/abs/2011.07992). An explanation for how the spectra are calculated can be found in
Appendix B of [this paper](https://doi.org/10.1103/PhysRevResearch.3.033123).