# SignalSnap: Signal Analysis In Python Made Easy 
by M. Sifft and D. Hägele

The SignalSnap Python package is open-source software for analyzing signals in the spectral domain. Here, a few outstanding 
features of SignalSnap:
* Calculation of a powerspectrum within a few lines of code
* Errors of spectral values are automatically calculated 
* Calculation of higher-order spectra or polyspectra (3rd and 4th order) (+ error estimation) in a few lines of code
* Support for just-in-time import from hdf data (dataset does not have to fit in RAM) 
* Function for conversion of Numpy array to hdf data is also provided
* All calculation can be performed on GPU (NVidia and AMD) (see Arrayfire) 
* Advanced plotting options for two-dimensional higher-order spectra 
* Usage of unbiased estimators for higher-order cumulants (see Literature)
* Efficient implementation of the confined Gaussian window for an optimal time-bandwidth product (see Literature)
* Special functions for the verification of the stationarity of a signal

## Installation
SignalSnap is available on `pip` and can be installed with 
```bash
pip install signalsnap
```

## Documentation
A comprehensive documentation of SignalSnap will follow soon. 

### Examples
Examples for every function of the package are currently added to the folder Examples. Here are a few lines 
to get you started. Let's say you have your signal/datset as a Numpy array called `y`.  

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