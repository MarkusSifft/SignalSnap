# SignalSnap: Signal Analysis In Python Made Easy 
by M. Sifft and D. Hägele

The SignalSnap Python package is open-source software for analyzing signals in the spectral domain. Here, a few outstanding 
features of SignalSnap:
* Calculation of a powerspectrum within a few lines of code
* Errors of spectral values are automatically calculated 
* Calculation of higher-order spectra or polyspectra (3rd and 4th order) (+ error estimation) in a few lines of code
* Support for just-in-time import from hdf data (dataset does not have to fit in RAM)
* Function for conversion of Numpy array to hdf data is also provided
* Correlations between two time series can be calculated
* All calculation can be performed on GPU (NVidia and AMD) (see Arrayfire) 
* Advanced plotting options for two-dimensional higher-order spectra 
* Usage of unbiased estimators for higher-order cumulants (see Literature)
* Efficient implementation of the confined Gaussian window for an optimal RMS time-bandwidth product (see Literature)
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
to get you started. We will generate some white noise as signal/dataset store it as Numpy array called `y`.

```python
import SignalSnap as snp
import numpy as np
rng = np.random.default_rng()

# ------ Generation of white noise -----
unit = 'Hz'
fs = 10e3 # sampling rate
N = 1e5 # number of points
t = np.arange(N) / fs
y = rng.normal(scale=1, size=t.shape)
```

Now we creat a spectrum object and feed it with the data. This object will store the dataset, 
later the spectra and errors, all freely chosen variables and contains 
the methods for calculating the spectra, plotting and storing.

```python
spec = snp.Spectrum(data=y, delta_t=1/fs)
T_window = 0.02 # these are now ms since the unit of choice are kHz
f_max = 5e3 # kHz
f, s, serr = spec.calc_spec(order_in=[2,3,4], T_window=T_window, f_max=f_max, backend='cpu')
```
The output will show you the actual length of a window (in case your T_window is not a multiple of 1/fs), the maximum 
frequency (Nyquist frequency) and the number of point of the calculated spectrum. The data points in the first window 
are plotted, so you can verify the window length (which is also given in points by chunk shape). The function will 
return `f` the frequencies at which the spectrum has been calculated, `s` the spectral values, and `serr` the error 
of the spectra value (1 sigma).

Visualization of the results is just as easy as the calculation.

```python
fig = spec.plot(order_in=[2,3,4], f_max=f_max/2)
```


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