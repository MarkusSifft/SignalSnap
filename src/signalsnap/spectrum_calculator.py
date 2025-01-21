# This file is part of signalsnap: Signal Analysis In Python Made Easy
# Copyright (c) 2020 and later, Markus Sifft and Daniel Hägele.
#
# This software is provided under the terms of the 3-Clause BSD License.
# For details, see the LICENSE file in the root of this repository or
# https://opensource.org/licenses/BSD-3-Clause

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pickle

import arrayfire as af
from arrayfire.arith import conjg as conj
from arrayfire.interop import from_ndarray as to_gpu
from arrayfire.signal import fft_r2c
from arrayfire.statistics import mean

from numba import njit
from scipy.fft import rfftfreq
from tqdm.auto import tqdm

from .spectrum_config import SpectrumConfig
from .plot_config import PlotConfig

try:
    import torch
except ImportError:
    print("Failed to import torch. This is only a problem when you want to use the CUDA backend.")


class MissingValueError(Exception):
    """Base class for missing value exceptions"""
    pass


def pickle_save(path, obj):
    """
    Helper function to pickle system objects

    Parameters
    ----------
    path : str
        Location of saved data
    obj : System obj

    """
    f = open(path, mode='wb')
    pickle.dump(obj, f)
    f.close()


def load_spec(path):
    """
    Helper function to load pickled objects.

    Parameters
    ----------
    path : str
        Path to pkl file.

    Returns
    -------
    Returns the object.

    """
    f = open(path, mode='rb')
    obj = pickle.load(f)
    f.close()
    return obj


def to_hdf(dt, data, path, group_name, dataset_name):
    """
    Helper function to generated h5 file from numpy array.

    Parameters
    ----------
    dt : float
        Inverse sampling rate of the signal (is saved as attribute "dt" to the dataset)
    data : array
        E.g. simulation results
    path : str
        Path for the data to be saved at
    group_name : str
        Name of the group in the h5 file
    dataset_name : str
        Name of the dataset in the h5 file

    """
    with h5py.File(path, "w") as f:
        grp = f.create_group(group_name)
        d = grp.create_dataset(dataset_name, data=data)
        d.attrs['dt'] = dt


# ----------- Old algorithm to calculate a_w3 ------------------------------------------

# @njit(parallel=False)
# def calc_a_w3(a_w_all, f_max_ind, m):
#     """
#     Preparation of a_(w1+w2) for the calculation of the bispectrum

#     Parameters
#     ----------
#     a_w_all : array
#         Fourier coefficients of the signal
#     f_max_ind : int
#         Index of the maximum frequency to be used for the calculation of the spectra
#     m : int
#         Number of windows per spectrum

#     Returns
#     -------
#     a_w3 : array
#         Matrix of Fourier coefficients
#     """

#     a_w3 = 1j * np.ones((f_max_ind // 2, f_max_ind // 2, m))
#     for i in range(f_max_ind // 2):
#         a_w3[i, :, :] = a_w_all[i:i + f_max_ind // 2, 0, :]
#     return a_w3.conj()

# --------------------------------------------------------------------------


def c1(a_w):
    """calculation of c1 / mean """
    # C_1 = < a_w >
    s1 = mean(a_w, dim=2)

    return s1[0]


@njit
def find_start_index_interlaced(data, T_window):
    """
    Returns the index of the first datapoint to be contained in the first interlaced window

    Parameters
    ----------
    data : array
        timestamps of detector clicks
    T_window : float
        window length in seconds (or unit of choice)

    Returns
    -------

    """
    end_time = T_window / 2

    if end_time > data[-1]:
        raise ValueError(
            "Not even half a window fits your data. Your resolution is either way too high or your data way too short.")

    if end_time <= data[0]:
        return 0

    i = 0
    while True:
        if data[i] > end_time:
            return i - 1
        i += 1


@njit
def find_end_index(data, start_index, T_window, m, frame_number, j):
    """

    Parameters
    ----------
    data : array
        timestamps of detector clicks
    start_index : int
        index of the last timestamp in the whole dataset that fitted into the prior window (zero in case of first window)
    T_window : float
        window length in seconds (or unit of choice)
    m : int
        number of windows to calculate the cumulant estimator from
    frame_number : int
        keeps track of the current frame (1 frame = m windows)
    j : int
        number of the current window in the frame

    Returns
    -------

    """
    end_time = T_window * (m * frame_number + (j + 1))
    end_time_interlaced = T_window * (m * frame_number + (j + 1)) + T_window / 2

    if end_time > data[-1] or end_time_interlaced > data[-1]:
        return -1, -1

    i = 0
    while True:
        if data[start_index + i] > end_time:
            start_index_out = start_index + i
            break
        i += 1

    i = 0
    while True:
        if data[start_index_out + i] > end_time_interlaced:
            start_index_interlaced_out = start_index_out + i
            break
        i += 1

    return start_index_out, start_index_interlaced_out


@njit
def g(x_, N_window, L, sigma_t):
    """
    Helper function to calculate the approx. confined gaussian window as defined in https://doi.org/10.1016/j.sigpro.2014.03.033

    Parameters
    ----------
    x_ : array
        points at which to calculate the function
    N_window : int
        length of window in points
    L : int
        N_window + 1
    sigma_t : float
        parameter of the approx. confined gaussian window (here chosen to be 0.14)

    Returns
    -------

    """
    return np.exp(-((x_ - N_window / 2) / (2 * L * sigma_t)) ** 2)


@njit
def calc_window(x, N_window, L, sigma_t):
    """
    Helper function to calculate the approx. confined gaussian window as defined in https://doi.org/10.1016/j.sigpro.2014.03.033

    Parameters
    ----------
    x : array
        points at which to calculate the function
    N_window : int
        length of window in points
    L : int
        N_window + 1
    sigma_t : float
        parameter of the approx. confined gaussian window (here chosen to be 0.14)

    Returns
    -------

    """
    return g(x, N_window, L, sigma_t) - (g(-0.5, N_window, L, sigma_t) * (
            g(x + L, N_window, L, sigma_t) + g(x - L, N_window, L, sigma_t))) / (
            g(-0.5 + L, N_window, L, sigma_t) + g(-0.5 - L, N_window, L, sigma_t))


@njit
def cgw(N_window, fs=None, ones=False, sigma_t=0.14):
    """
    Helper function to calculate the approx. confined gaussian window as defined in https://doi.org/10.1016/j.sigpro.2014.03.033

    Parameters
    ----------
    ones : bool
        if true, the window is simply set to one resulting in a rectangular window
    fs : float
        sampling rate of the signal
    N_window : int
        length of window in points

    Returns
    -------

    """
    x = np.linspace(0, N_window, N_window)
    L = N_window + 1
    window = calc_window(x, N_window, L, sigma_t)
    if ones:
        window = np.ones(N_window)

    norm = np.sum(window ** 2) / fs

    return window / np.sqrt(norm), norm


@njit
def apply_window(window_width, t_clicks, fs, sigma_t=0.14):
    """
    This function take the timestamps of the detector and applies the window function as an envelope treating the
    clicks as steps with height one.

    Parameters
    ----------
    window_width : float
        timely width of the window in unit of choice
    t_clicks : array
        timestamps of the dataset that lie within the current window
    fs : float
        sampling rate of the signal
    sigma_t : float
        parameter of the approx. confined gaussian window (here chosen to be 0.14)

    Returns
    -------

    """

    # ----- Calculation of g_k -------

    # TODO Hier gibts es auf jeden Fall Sachen, die nur einmal außerhalb dieser Funktion berechnet
    # werden könnten
    dt = 1 / fs

    N_window = window_width / dt
    L = N_window + 1

    x = t_clicks / dt
    window = calc_window(x, N_window, L, sigma_t)
    # if ones:
    #    window = np.ones(N_window)

    # ------ Normalizing by calculating full window with given resolution ------

    N_window_full = 70
    dt_full = window_width / N_window_full

    # window_full, norm = cgw(N_window_full, 1 / dt_full, ones=ones)
    window_full, norm = cgw(N_window_full, 1 / dt_full, sigma_t=sigma_t)

    return window / np.sqrt(norm), window_full


def unit_conversion(f_unit):
    """
    Helper function to automatically convert units.

    Parameters
    ----------
    f_unit : str
        Frequency unit

    Returns
    -------
    Returns the corresponding time unit.
    """

    if f_unit == 'Hz':
        t_unit = 's'
    elif f_unit == 'kHz':
        t_unit = 'ms'
    elif f_unit == 'MHz':
        t_unit = 'us'
    elif f_unit == 'GHz':
        t_unit = 'ns'
    elif f_unit == 'THz':
        t_unit = 'ps'
    elif f_unit == 'mHz':
        t_unit = 's $\cdot 10$'
    return t_unit


class SpectrumCalculator:
    """
    SpectrumCalculator class stores signal data, calculated spectra and error of spectral values.
    Allows for the calculation of the polyspectra of the signal and their visualization.
    Also hold methods for saving spectrum objects.

    Parameters
    ----------
    path : str
        Path to h5 file with stored signal
    group_key : str
        Group key for h5 file
    dataset : str
        Name of the dataset in h5 file
    data : array
        Signal to be analyzed as Numpy array
    corr_data : array
        Second signal used as correlation data as Numpy array
    corr_path : str
        Path to h5 file with stored second signal for correlation
    corr_group_key : str
        Group key for h5 file with correlation signal
    corr_dataset : str
        Name of the dataset in h5 file with correlation signal

    Attributes
    ----------
    corr_data_path: str
        Path to h5 file with a second stored signal to be correlated with the first on (see path)
    T_window : float
        Length of window in seconds (or unit of choice) used for the calculation of
        any spectra
    path : str
        Path to h5 file with stored signal
    freq : array
        Stores frequencies at with the spectra are calculated
    f_max : float
        Used to limit the upper frequency to calculate spectral values at (especially important for high sampling rates)
    fs : float
        Stores the sampling rate of the signal
    f_lists: dict
        Used for the calculation of Poisson spectra. Stores all frequency values for the calculation of the
        non-uniform discreet Fourier transformation. (Also used for plotting with broken frequency axis.)
    S : dict
        Stores spectra values (final storage at the end of the calculation)
    S_gpu : dict
        Temporarily stores spectra values during calculation on the GPU
    S_err : dict
        Stores errors of spectra values (final storage at the end of the calculation)
    S_err_gpu : dict
        Temporarily stores errors of spectra values during calculation on the GPU
    S_errs : dict
        Temporarily stores m_var spectra used for the calculation of the errors
    S_stationarity_temp : dict
        Stores m_stationarity spectra which are then averaged and used for the stationarity plot
    S_stationarity : dict
        Stores the m_stationarity averaged spectra used for the stationarity plot
    group_key : str
        Group key of the dataset within the h5 file
    dataset : str
        Name of the dataset in the h5 file
    window_points : int
        Number of points in a window
    m : int
        Number of windows for the estimation of the cumulant
    m_var : int
        m_var single spectra are used at a time for the calculation of the spectral errors
    m_stationarity : int
        Number of spectra which are then averaged and used for the stationarity plot
    first_frame_plotted : bool
        If true the first frame is plotted during the calculation of the spectra
    delta_t : float
        Inverse of the sampling rate of the signal
    data : array / pointer
        Stores the full dataset or is a pointer to the data in the h5 file (recommended of the dataset
        is larger than your RAM)
    corr_data : array / pointer
        Stores the full second dataset or is a pointer to the data in the h5 file (recommended of the dataset
        is larger than your RAM)
    corr_group_key : str
        Group key of the second dataset within the h5 file
    corr_dataset : str
        Name of the second dataset in the h5 file
    main_data : array / pointer
        (outdated / for old methods) Stores the full dataset or is a pointer to the data in the h5 file (recommended of the dataset
        is larger than your RAM)
    err_counter : dict
        Counts the number single spectra already stored for the calculation of errors
     stationarity_counter : dict
        Counts the number single spectra already stored for the averaging for the stationarity plot
    """

    def __init__(self, config: SpectrumConfig):

        self.use_naive_estimator = None
        self.config = config
        self.plot_config = None
        self.T_window = None
        self.freq = {2: None, 3: None, 4: None}
        self.fs = None
        self.f_lists = {2: None, 3: None, 4: None}
        self.S = {1: None, 2: None, 3: None, 4: None}
        self.S_gpu = {1: None, 2: None, 3: None, 4: None}
        self.S_err = {1: None, 2: None, 3: None, 4: None}
        self.S_err_gpu = {1: None, 2: None, 3: None, 4: None}
        self.S_errs = {1: None, 2: [], 3: [], 4: []}
        self.S_stationarity_temp = {1: None, 2: None, 3: None, 4: None}
        self.S_stationarity = {1: [], 2: [], 3: [], 4: []}
        self.window_points = None
        self.m = {1: None, 2: None, 3: None, 4: None}
        self.m_var = {1: None, 2: None, 3: None, 4: None}
        self.m_stationarity = {1: None, 2: None, 3: None, 4: None}
        self.first_frame_plotted = False
        self.main_data = None
        self.err_counter = {1: 0, 2: 0, 3: 0, 4: 0}
        self.stationarity_counter = {1: 0, 2: 0, 3: 0, 4: 0}
        self.t_unit = unit_conversion(config.f_unit)
        self.number_of_error_estimates = 0

    # ==================== New algorith from here =================================

    def a_w3_gen(self, f_max_ind, m):
        '''
        This new approch needs to generate the initialization only once and isn't
        called in every step.

        Parameters
        ----------
        f_max_ind : int
            Index of the maximum frequency to be used for the calculation of the
            spectra
         m : int
            Number of windows per spectrum

        Returns
        -------
        a_w3 : array
            A place holder fo the matrix of Fourier coefficients
        '''

        if self.config.full_bispectrum:
            n = 2 * (f_max_ind // 2) - 1
            a_w3 = np.ones((f_max_ind // 2, n, m), dtype=complex) * 1j

        else:
            a_w3 = np.ones((f_max_ind // 2, f_max_ind // 2, m), dtype=complex) * 1j

        return a_w3

    def index_generation_to_aw_3(self, f_max_ind):
        '''
        Constructs an index matrix to correctly place elements of the Fourier coefficients
        of the signal in the desired order, accounting for potential spectrum symmetry.

        Parameters
        ----------
        f_max_ind : int
            Index of the maximum frequency to be used for the calculation of the spectra.

        Returns
        -------
        indices : ndarray
            - If `self.config.full_bispectrum` is True:
                A 2D index matrix of shape (n, 2n-1), where
                n = f_max_ind // 2. The first row spans indices
                from -(n-1) to +(n-1), and subsequent rows
                increment accordingly, aligning with the shifted
                Fourier coefficient arrangement for full bispectrum.
            - Otherwise:
                A 2D index matrix of shape (n, n) with elements
                given by the sum of their row and column indices,
                corresponding to the standard arrangement of Fourier
                coefficients.
        '''

        if self.config.full_bispectrum:
            n = f_max_ind // 2
            row_indices = np.arange(n)[:, None]  # Shape (n, 1)
            col_indices = np.arange(-(n - 1), n)  # Shape (2n-1,)
            indices = row_indices + col_indices  # Resulting shape (n, 2n-1) # MAYBE FALSCH

        else:
            indices = np.arange(f_max_ind // 2)[:, None] + np.arange(f_max_ind // 2)

        return indices

    def calc_a_w3(self, a_w_all, f_max_ind, m, a_w3, indices, backend):
        '''
        Preparation of a_(w1+w2) for the calculation of the bispectrum

        Parameters
        ----------
        a_w_all : array
            Fourier coefficients of the signal
        f_max_ind : int
            Index of the maximum frequency to be used for the calculation of the spectra
        m : int
            Number of windows per spectrum
        a_w3 : array
            Place holder generated by a_w3_gen function
        indices : array
            Index matrix generated by indi function
        backend : str
            backend : {'cpu', 'opencl', 'cuda'}
            backend for computation. opencl is not an option for fast computing of this
            function

        Returns
        -------
        a_w3 : array
            Matrix of Fourier coefficients
        '''

        if backend != 'cuda':
            # ----------------------this runs on cpu------------------------------
            #       !!! EVEN IF OPENCL IS USED IT WILL RUN ONY CPU !!!
            # Use advanced indexing to replace certain elements with values
            # from a_w_all
            a_w3[np.arange(f_max_ind // 2), :, :] = a_w_all[indices, 0, :m]
            a_w3 = a_w3.conj()

        elif backend == 'cuda':
            # ----------------------this runs on NVIDIA ONLY----------------------
            # Use advanced indexing and Torch to replace certain elements
            # with values from a_w_all
            device = torch.device('cuda')

            a_w_all = torch.from_numpy(a_w_all).to(device)

            a_w3 = torch.from_numpy(a_w3).to(device)
            indices = torch.from_numpy(indices).to(device)

            a_w3[torch.arange(f_max_ind // 2), :, :] = a_w_all[indices, 0, :m]
            a_w3 = a_w3.conj()

            # Convert back to numpy array
            a_w3 = a_w3.cpu().numpy()

        return a_w3

    # ==================== End of the new algorith =================================

    def calc_single_window(self):
        """
        Return a single example of the window function for normalization purposes.

        This method uses the following attributes from the `config` class:
        - delta_t: The inverse of the sampling rate of the signal (float).
        - window_width: The timely width of the window in the unit of choice (float).
        - sigma_t: Parameter of the approx. confined Gaussian window (float, typically 0.14).

        Returns
        -------
        window_full : array_like
            Calculated window function.
        N_window_full : int
            Size of the window function.

        Notes
        -----
        Ensure that the `config` attributes are properly set before calling this method.
        """

        # ----- Calculation of g_k -------

        N_window = self.T_window / self.config.delta_t
        L = N_window + 1

        # if ones:
        #    window = np.ones(N_window)

        # ------ Normalizing by calculating full window with given resolution ------

        N_window_full = 70
        dt_full = self.T_window / N_window_full

        # window_full, norm = cgw(N_window_full, 1 / dt_full, ones=ones)
        window_full, norm = cgw(N_window_full, 1 / dt_full, sigma_t=self.config.sigma_t)

        return window_full, N_window_full

    def plot_first_frame(self, chunk, window_size):
        """
        Helper function for plotting one window during the calculation of the spectra.
        Useful for checking data and correct window length.

        Parameters
        ----------
        chunk : array_like
            One frame of the dataset.
        window_size : int
            Size of window in points. Represents the number of data points in each window of the spectrum.

        Notes
        -----
        The inverse sampling rate of the signal (`delta_t`) is retrieved from the `config` class within the method.
        """
        first_frame = chunk[:window_size]
        t = np.arange(len(first_frame)) * self.config.delta_t
        plt.figure(figsize=(14, 3))

        plt.rc('text', usetex=False)
        plt.rc('font', size=12)
        plt.rcParams["axes.axisbelow"] = False

        plt.title(f'data in first window ({first_frame.shape[0]} points)')

        plt.plot(t, first_frame)

        plt.xlim([0, t[-1]])
        plt.xlabel('t / (' + self.t_unit + ')')
        plt.ylabel('amplitude')

        plt.show()

    def c2(self, a_w, a_w_corr):
        """
        Calculation of the second cumulant (C2) for the power spectrum.

        Parameters
        ----------
        a_w : array_like
            The Fourier-transformed data window.
        a_w_corr : array_like
            The Fourier-transformed correlation window.

        Returns
        -------
        s2 : array_like
            The calculated second cumulant (C2).

        Notes
        -----
        The second cumulant (C2) is calculated based on the parameters provided in the `config` attribute.
        The calculation follows the formula:
        C_2 = m / (m - 1) * (<a_w * a_w*> - <a_w> <a_w*>)
        Where <...> denotes the mean, and * denotes complex conjugate.

        If the `coherent` flag in `config` is set to True, a modified calculation is performed.
        """

        m = self.config.m

        mean_1 = mean(a_w * conj(a_w_corr), dim=2)

        if self.config.coherent:
            s2 = mean_1

        else:

            mean_2 = mean(a_w, dim=2)

            if self.config.corr_data is None and self.config.corr_path is None:
                mean_3 = conj(mean_2)
            else:
                mean_3 = mean(conj(a_w_corr), dim=2)

            if self.use_naive_estimator:
                s2 = mean_1 - mean_2 * mean_3

            else:
                s2 = m / (m - 1) * (mean_1 - mean_2 * mean_3)

        return s2

    def c3(self, a_w1, a_w2, a_w3):
        """
        Calculation of c3 for bispectrum (see arXiv:1904.12154)
        # C_3 = m^2 / (m - 1)(m - 2) * (< a_w1 * a_w2 * a_w3 >
        #                                      sum_123
        #       - < a_w1 >< a_w2 * a_w3 > - < a_w1 * a_w2 >< a_w3 > - < a_w1 * a_w3 >< a_w2 >
        #          sum_1      sum_23           sum_12        sum_3         sum_13      sum_2
        #       + 2 < a_w1 >< a_w2 >< a_w3 >)
        #             sum_1   sum_2   sum_3
        # with w3 = - w1 - w2

        Parameters
        ----------
        a_w1 : array
            Fourier coefficients of signal as vertical array.
        a_w2 : array
            Fourier coefficients of signal as horizontal array.
        a_w3 : array
            a_w1+w2 as matrix.

        Returns
        -------
        Returns the c3 estimator as matrix.

        Notes
        -----
        The number of windows `m` used for the calculation is obtained from `self.config.m`.
        """

        m = self.config.m

        # ones = to_gpu(np.ones_like(a_w1.to_ndarray()))
        d_1 = af.tile(af.transpose(a_w1),
                      a_w2.shape[0], )  # af.matmulNT(ones, a_w1) # copies a_w1 vertically # MAYBE MAYBE FALSCH
        d_2 = af.tile(a_w2, 1, a_w1.shape[0])  # copies a_w2 horizontally # MAYBE MAYBE FALSCH
        d_3 = a_w3
        # ================ moment ==========================
        if self.config.coherent:
            d_12 = d_1 * d_2
            d_123 = d_12 * d_3
            s3 = mean(d_123, dim=2)

        # ==================================================
        else:
            # Calculate products for efficiency
            d_12 = d_1 * d_2
            d_13 = d_1 * d_3
            d_23 = d_2 * d_3
            d_123 = d_12 * d_3

            # Compute means
            d_means = [mean(d, dim=2) for d in [d_1, d_2, d_3, d_12, d_13, d_23, d_123]]
            d_1_mean, d_2_mean, d_3_mean, d_12_mean, d_13_mean, d_23_mean, d_123_mean = d_means

            if self.use_naive_estimator:
                s3 = (d_123_mean - d_12_mean * d_3_mean -
                      d_13_mean * d_2_mean - d_23_mean * d_1_mean +
                      2 * d_1_mean * d_2_mean * d_3_mean)

            else:
                # Compute c3 estimator using the equation provided
                s3 = m ** 2 / ((m - 1) * (m - 2)) * (d_123_mean - d_12_mean * d_3_mean -
                                                     d_13_mean * d_2_mean - d_23_mean * d_1_mean +
                                                     2 * d_1_mean * d_2_mean * d_3_mean)
        return s3

    # ==================== new compact algorithm for c4 =================================
    def c4(self, a_w, a_w_corr):
        """
            Calculation of c4 for trispectrum based on equation 60 in arXiv:1904.12154.

            Parameters
            ----------
            a_w : array
                Fourier coefficients of the signal.
            a_w_corr : array
                Fourier coefficients of the signal or a second signal.

            Returns
            -------
            s4 : array
                The c4 estimator as a matrix.

            Notes
            -----
            The value of `m`, the number of windows used for the calculation of one spectrum,
            is obtained from the `config` object associated with this instance.
            """

        m = self.config.m

        x = a_w
        z = a_w_corr

        y = conj(x)
        w = conj(z)

        if self.config.coherent:
            sum_11c22c = af.matmulNT(x * y, z * w)
            sum_11c22c_m = mean(sum_11c22c, dim=2)
            s4 = sum_11c22c_m

        else:
            x_mean = x - af.tile(mean(x, dim=2), d0=1, d1=1, d2=x.shape[2])
            y_mean = y - af.tile(mean(y, dim=2), d0=1, d1=1, d2=x.shape[2])
            z_mean = z - af.tile(mean(z, dim=2), d0=1, d1=1, d2=x.shape[2])
            w_mean = w - af.tile(mean(w, dim=2), d0=1, d1=1, d2=x.shape[2])

            xyzw = af.matmulNT(x_mean * y_mean, z_mean * w_mean)
            xyzw_mean = mean(xyzw, dim=2)

            xy_mean = mean(x_mean * y_mean, dim=2)
            zw_mean = mean(z_mean * w_mean, dim=2)
            xy_zw_mean = af.matmulNT(xy_mean, zw_mean)

            xz_mean = mean(af.matmulNT(x_mean, z_mean), dim=2)
            yw_mean = mean(af.matmulNT(y_mean, w_mean), dim=2)
            xz_yw_mean = xz_mean * yw_mean

            xw_mean = mean(af.matmulNT(x_mean, w_mean), dim=2)
            yz_mean = mean(af.matmulNT(y_mean, z_mean), dim=2)
            xw_yz_mean = xw_mean * yz_mean

            if self.use_naive_estimator:
                s4 = xyzw_mean - (xy_zw_mean + xz_yw_mean + xw_yz_mean)

            else:
                s4 = m ** 2 / ((m - 1) * (m - 2) * (m - 3)) * (
                        (m + 1) * xyzw_mean -
                        (m - 1) * (
                                xy_zw_mean + xz_yw_mean + xw_yz_mean
                        )
                )

        return s4

    def c4_old(self, a_w, a_w_corr):
        """
        Calculation of c4 for trispectrum based on equation 60 in arXiv:1904.12154.

        Parameters
        ----------
        a_w : array
            Fourier coefficients of the signal.
        a_w_corr : array
            Fourier coefficients of the signal or a second signal.

        Returns
        -------
        s4 : array
            The c4 estimator as a matrix.

        Notes
        -----
        The value of `m`, the number of windows used for the calculation of one spectrum,
        is obtained from the `config` object associated with this instance.
        """

        m = self.config.m

        a_w_conj = conj(a_w)  #y
        a_w_conj_corr = conj(a_w_corr)  #w

        ones = to_gpu(np.ones_like(a_w.to_ndarray()[:, :, 0]))

        sum_11c22c = af.matmulNT(a_w * a_w_conj, a_w_corr * a_w_conj_corr)  # d1
        sum_11c22c_m = mean(sum_11c22c, dim=2)

        if self.config.coherent:
            s4 = sum_11c22c_m

        else:
            sum_11c2 = af.matmulNT(a_w * a_w_conj, a_w_corr)  # d2
            sum_11c2_m = mean(sum_11c2, dim=2)
            sum_122c = af.matmulNT(a_w, a_w_corr * a_w_conj_corr)  # d3
            sum_122c_m = mean(sum_122c, dim=2)
            sum_1c22c = af.matmulNT(a_w_conj, a_w_corr * a_w_conj_corr)  # d4
            sum_1c22c_m = mean(sum_1c22c, dim=2)
            sum_11c2c = af.matmulNT(a_w * a_w_conj, a_w_conj_corr)  # d5
            sum_11c2c_m = mean(sum_11c2c, dim=2)

            sum_11c = a_w * a_w_conj  # d6
            sum_11c_m = mean(sum_11c, dim=2)
            sum_22c = a_w_corr * a_w_conj_corr  # d6
            sum_22c_m = mean(sum_22c, dim=2)
            sum_12c = af.matmulNT(a_w, a_w_conj_corr)  # d7
            sum_12c_m = mean(sum_12c, dim=2)
            sum_1c2 = af.matmulNT(a_w_conj, a_w_corr)  # d8
            sum_1c2_m = mean(sum_1c2, dim=2)

            sum_12 = af.matmulNT(a_w, a_w_corr)  # d9
            sum_12_m = mean(sum_12, dim=2)
            sum_1c2c = af.matmulNT(a_w_conj, a_w_conj_corr)  # d9
            sum_1c2c_m = mean(sum_1c2c, dim=2)

            sum_1_m = mean(a_w, dim=2)  # d10
            sum_1c_m = mean(a_w_conj, dim=2)  # d11
            sum_2_m = mean(a_w_corr, dim=2)  # d10
            sum_2c_m = mean(a_w_conj_corr, dim=2)  # d11

            sum_11c_m = af.matmulNT(sum_11c_m, ones)  # d6'
            sum_22c_m = af.matmulNT(ones, sum_22c_m)  # d6''
            sum_1_m = af.matmulNT(sum_1_m, ones)  # d10'
            sum_1c_m = af.matmulNT(sum_1c_m, ones)  # d11'
            sum_2_m = af.matmulNT(ones, sum_2_m)  # d10''
            sum_2c_m = af.matmulNT(ones, sum_2c_m)  # d11''

            s4 = m ** 2 / ((m - 1) * (m - 2) * (m - 3)) * (
                    (m + 1) * sum_11c22c_m - (m + 1) * (sum_11c2_m * sum_2c_m + sum_11c2c_m * sum_2_m +
                                                        sum_122c_m * sum_1c_m + sum_1c22c_m * sum_1_m)
                    - (m - 1) * (sum_11c_m * sum_22c_m + sum_12_m * sum_1c2c_m + sum_12c_m * sum_1c2_m)
                    + 2 * m * (sum_11c_m * sum_2_m * sum_2c_m + sum_12_m * sum_1c_m * sum_2c_m +
                               sum_12c_m * sum_1c_m * sum_2_m + sum_22c_m * sum_1_m * sum_1c_m +
                               sum_1c2c_m * sum_1_m * sum_2_m + sum_1c2_m * sum_1_m * sum_2c_m)
                    - 6 * m * sum_1_m * sum_1c_m * sum_2_m * sum_2c_m
            )

        return s4

    def add_random_phase(self, a_w, window_size):
        """(Experimental function) Adds a random phase proportional to the frequency to deal with ultra-coherent signals.

        This method uses the number of windows per frame (`m`) and the inverse of the sampling rate (`delta_t`) from the config class.

        Parameters
        ----------
        a_w : array_like
            Fourier coefficients of the window.
        window_size : int
            Size of the window in points.

        Returns
        -------
        array_like
            Phased Fourier coefficients.

        Notes
        -----
        The attributes `m` and `delta_t` are taken from the config class associated with this instance.
        """

        m = self.config.m

        random_factors = np.random.uniform(high=window_size * self.config.delta_t, size=m)
        freq_all_freq = rfftfreq(int(window_size), self.config.delta_t)
        freq_mat = np.tile(np.array([freq_all_freq]).T, m)
        factors = np.exp(1j * 2 * np.pi * freq_mat * random_factors)
        factors = factors.reshape(a_w.shape)
        factors_gpu = to_gpu(factors)
        a_w_phased = a_w * factors_gpu
        return a_w_phased

    def import_data(self):
        """
        Helper function to load data from h5 file into a numpy array.
        Imports data in the .h5 format with structure group_key -> data + attrs[dt].
        Parameters such as 'full_import', 'path', 'group_key', and 'dataset' are obtained from the config attribute.

        Returns
        -------
        numpy.ndarray
            Simulation result if 'full_import' is True; otherwise, returns a pointer to the data in the h5 file.
        float
            Inverse sampling rate (only if 'delta_t' in config is None).

        Notes
        -----
        Ensure that the config attribute is properly set before calling this method, and that the paths and keys correspond to existing elements in the h5 file.
        """

        main = h5py.File(self.config.path, 'r')
        if self.config.group_key == '':
            main_data = main[self.config.dataset]
        else:
            main_group = main[self.config.group_key]
            main_data = main_group[self.config.dataset]
        if self.config.delta_t is None:
            self.config.delta_t = main_data.attrs['dt']
        if self.config.full_import:
            return main_data[()]
        else:
            return main_data

    def import_corr_data(self):
        """
        Helper function to load data from h5 file into a numpy array.
        Imports data in the .h5 format with structure group_key -> data + attrs[dt].
        Parameters such as 'full_import', 'path', 'group_key', and 'dataset' are obtained from the config attribute.

        Returns
        -------
        numpy.ndarray
            Simulation result if 'full_import' is True; otherwise, returns a pointer to the data in the h5 file.
        float
            Inverse sampling rate (only if 'delta_t' in config is None).

        Notes
        -----
        Ensure that the config attribute is properly set before calling this method, and that the paths and keys correspond to existing elements in the h5 file.
        """

        main = h5py.File(self.config.corr_path, 'r')
        if self.config.corr_group_key == '':
            main_data = main[self.config.corr_dataset]
        else:
            main_group = main[self.config.corr_group_key]
            main_data = main_group[self.config.corr_dataset]
        if self.config.delta_t is None:
            self.config.delta_t = main_data.attrs['dt']
        if self.config.full_import:
            return main_data[()]
        else:
            return main_data

    def save_spec(self, save_path, remove_S_stationarity=False):
        """
        Save the SpectrumCalculator object to a file, removing pointers and the full dataset before saving.

        This method clears certain attributes of the object, including GPU data, main data, and config data,
        before storing the object to the specified path.

        Parameters
        ----------
        save_path : str
            Path where the object will be stored, including the filename and extension.

        Examples
        --------
        >>> spectrum_obj.save_spec('path/to/file.pkl', remove_S_stationarity=False)

        Notes
        -----
        It's important to ensure that the save path is writable and has the correct permissions.
        """
        self.S_gpu = None
        self.S_err_gpu = None
        self.main_data = None
        self.data = None
        if remove_S_stationarity:
            self.S_stationarity = None

        # Only set to None if the attribute exists
        if hasattr(self, 'config'):
            self.config.corr_data = None
            self.config.data = None

        self.S_errs = None
        self.S_stationarity_temp = None
        pickle_save(save_path, self)

    def store_single_spectrum(self, single_spectrum, order):

        """
        Helper function to store the spectra of single frames afterwards used for calculation of errors and overlaps.

        Parameters
        ----------
        single_spectrum : array
            SpectrumCalculator of a single frame.
        order : {2, 3, 4}
            Order of the spectra to be calculated.

        Notes
        -----
        This method relies on the configuration parameters `m_var` and `m_stationarity` from the `config` object.

        `m_var`: int
            Number of spectra to calculate the variance from (should be set as high as possible).
        `m_stationarity`: int
            Number of spectra after which their mean is stored to verify the stationarity of the data.

        This function also makes use of several instance variables like `S_gpu`, `S_errs`, `err_counter`, etc. to
        perform its operations.

        This method is intended for internal use within the class.
        """

        if self.S_gpu[order] is None:
            self.S_gpu[order] = single_spectrum
        else:
            self.S_gpu[order] += single_spectrum

        if not self.config.turbo_mode:

            if order == 1:
                self.S_errs[order][0, self.err_counter[order]] = single_spectrum
            elif order == 2:
                self.S_errs[order][:, self.err_counter[order]] = single_spectrum
            else:
                self.S_errs[order][:, :, self.err_counter[order]] = single_spectrum
            self.err_counter[order] += 1

            if self.config.m_stationarity is not None:
                if order == 1:
                    self.S_stationarity_temp[order][0, self.stationarity_counter[order]] = single_spectrum
                elif order == 2:
                    self.S_stationarity_temp[order][:, self.stationarity_counter[order]] = single_spectrum
                else:
                    self.S_stationarity_temp[order][:, :, self.stationarity_counter[order]] = single_spectrum
                self.stationarity_counter[order] += 1

                if self.stationarity_counter[order] % self.config.m_stationarity == 0:
                    if order == 1:
                        self.S_stationarity[order].append(af.mean(self.S_stationarity_temp[order], dim=1).to_ndarray())
                    elif order == 2:
                        self.S_stationarity[order].append(af.mean(self.S_stationarity_temp[order], dim=1).to_ndarray())
                    else:
                        self.S_stationarity[order].append(af.mean(self.S_stationarity_temp[order], dim=2).to_ndarray())
                    self.stationarity_counter[order] = 0

            if self.err_counter[order] % self.config.m_var == 0:
                if order == 1 or order == 2:
                    dim = 1
                else:
                    dim = 2

                if order == self.orders[0]:
                    self.number_of_error_estimates += 1

                # S_err_gpu_real = af.sqrt(self.config.m_var / (self.config.m_var - 1) * (
                #        af.mean(af.real(self.S_errs[order]) * af.real(self.S_errs[order]), dim=dim) -
                #        af.mean(af.real(self.S_errs[order]), dim=dim) * af.mean(
                #    af.real(self.S_errs[order]), dim=dim)))
                # S_err_gpu_imag = af.sqrt(self.config.m_var / (self.config.m_var - 1) * (
                #        af.mean(af.imag(self.S_errs[order]) * af.imag(self.S_errs[order]), dim=dim) -
                #        af.mean(af.imag(self.S_errs[order]), dim=dim) * af.mean(
                #    af.imag(self.S_errs[order]), dim=dim)))

                S_err_gpu_real = self.config.m_var / (self.config.m_var - 1) * (
                        af.mean(af.real(self.S_errs[order]) * af.real(self.S_errs[order]), dim=dim) -
                        af.mean(af.real(self.S_errs[order]), dim=dim) * af.mean(
                    af.real(self.S_errs[order]), dim=dim))
                S_err_gpu_imag = self.config.m_var / (self.config.m_var - 1) * (
                        af.mean(af.imag(self.S_errs[order]) * af.imag(self.S_errs[order]), dim=dim) -
                        af.mean(af.imag(self.S_errs[order]), dim=dim) * af.mean(
                    af.imag(self.S_errs[order]), dim=dim))

                self.S_err_gpu = (S_err_gpu_real + 1j * S_err_gpu_imag) / self.config.m_var

                if self.S_err[order] is None:
                    self.S_err[order] = self.S_err_gpu.to_ndarray()
                else:
                    self.S_err[order] += self.S_err_gpu.to_ndarray()

                self.err_counter[order] = 0

    def calc_overlap(self, t_unit=None, imag=False, scale_t=1):

        """
        Calculate and plot the overlap between all m_stationarity spectra and the overall mean spectrum.
        This can be used to identify slow drifts and singular events that differ from the mean spectrum.

        Parameters
        ----------
        t_unit : str, optional
            Unit of time for plotting on the x-axis. If not specified, `self.t_unit` is used. Default is None.
        imag : bool, optional
            If True, the imaginary part of all spectra is used for calculation and plotting. Default is False.
        scale_t : float, optional
            Factor to scale the time axis. Default is 1.

        Returns
        -------
        t : ndarray
            Time array for overlap_s2, overlap_s3, overlap_s4.
        t_main : ndarray
            Time array for the main data.
        overlap_s2 : ndarray
            Overlap for s2.
        overlap_s3 : ndarray
            Overlap for s3.
        overlap_s4 : ndarray
            Overlap for s4.

        Notes
        -----
        The function normalizes the overlaps and the main data for visualization, and then plots them
        along with the specified labels and titles. The time units and scaling can be controlled
        through the parameters.
        """

        if t_unit is None:
            t_unit = self.t_unit

        plt.figure(figsize=(28, 13))

        overlap_s2 = [np.var(self.S_stationarity[2][:, i] * self.S[2]) for i in range(self.S_stationarity[2].shape[1])]

        overlap_s3 = [np.var(self.S_stationarity[3][:, :, i] * self.S[3]) for i in
                      range(self.S_stationarity[3].shape[2])]

        overlap_s4 = [np.var(self.S_stationarity[4][:, :, i] * self.S[4]) for i in
                      range(self.S_stationarity[4].shape[2])]

        t = np.linspace(0, self.config.delta_t * self.main_data.shape[0],
                        self.S_stationarity[4][1, 1, :].shape[0]) / scale_t
        t_main = np.linspace(0, self.config.delta_t * self.main_data.shape[0], self.main_data.shape[0]) / scale_t

        if imag:
            overlap_s2 = np.imag(overlap_s2)
            overlap_s3 = np.imag(overlap_s3)
            overlap_s4 = np.imag(overlap_s4)

        plt.plot(t, overlap_s2 / max(overlap_s2), label='s2')
        plt.plot(t, overlap_s3 / max(overlap_s3), label='s3')
        plt.plot(t, overlap_s4 / max(overlap_s4), label='s4')

        plt.plot(t_main, self.main_data / max(self.main_data))
        plt.legend()
        plt.xlabel(t_unit)
        plt.ylabel('normalized')
        if not imag:
            plt.title('real part')
        else:
            plt.title('imaginary part')
        plt.show()
        return t, t_main, overlap_s2, overlap_s3, overlap_s4

    def fourier_coeffs_to_spectra(self, orders, a_w_all_gpu, f_max_ind, f_min_ind,
                                  single_window, window=None, chunk_corr_gpu=None,
                                  window_points=None):
        """
        Helper function to calculate the (1,2,3,4)-order cumulant from the Fourier coefficients of the windows in
        one frame.

        This function computes the single spectra of various orders based on Fourier coefficients. If correlation
        data is available, it is used to calculate the second and fourth-order spectra.

        Parameters
        ----------
        orders : list of {1, 2, 3, 4}
            Orders of the spectra to be calculated.
        a_w_all_gpu : array_like
            A matrix containing the Fourier coefficients of the windows.
        f_max_ind : int
            Index of the maximum frequency in the frequency array to calculate the spectral values at.
        single_window : array_like
            Values of the window function used to normalize spectra.
        window : array_like, optional
            Matrix containing multiple single_windows used to apply the window function to the whole frame. Default is None.
        chunk_corr_gpu : array_like, optional
            Matrix containing one frame of the correlation dataset. Default is None.
        window_points : array_like, optional
            Points of the window function used in random phase addition. Default is None.

        Returns
        -------
        None
            The function stores the calculated single spectra internally and does not return any value.

        Notes
        -----
        This function makes use of the `config` class object that stores the configuration and parameters
        required for the calculations, including 'm', 'm_var', 'm_stationarity', 'coherent', 'random_phase', etc.
        """

        for order in orders:
            if order == 1:
                a_w = af.lookup(a_w_all_gpu, af.Array(list(range(f_min_ind, f_max_ind))), dim=0)
                single_spectrum = c1(a_w) / self.config.delta_t / single_window.mean() / single_window.shape[0]

            elif order == 2:
                if self.config.f_lists is None:
                    a_w = af.lookup(a_w_all_gpu, af.Array(list(range(f_min_ind, f_max_ind))), dim=0)
                else:
                    a_w = a_w_all_gpu

                if self.config.corr_data is not None:
                    a_w_all_corr = fft_r2c(window * chunk_corr_gpu, dim0=0, scale=self.config.delta_t)
                    a_w_corr = af.lookup(a_w_all_corr, af.Array(list(range(f_min_ind, f_max_ind))), dim=0)
                    single_spectrum = self.c2(a_w, a_w_corr) / (
                            self.config.delta_t * (single_window ** order).sum())

                else:
                    single_spectrum = self.c2(a_w, a_w) / (
                            self.config.delta_t * (single_window ** order).sum())

            elif order == 3:

                a_w1 = af.lookup(a_w_all_gpu, af.Array(list(range(f_min_ind, f_max_ind // 2))), dim=0)
                a_w2 = a_w1

                # a_w3 = to_gpu(calc_a_w3(a_w_all_gpu.to_ndarray(), f_max_ind, self.config.m))

                # ======== New algorithm divides the steps ==========
                if self.config.corr_data is not None:
                    a_w_all_corr = fft_r2c(window * chunk_corr_gpu, dim0=0, scale=self.config.delta_t)
                    t0 = a_w_all_corr.to_ndarray()
                else:
                    t0 = a_w_all_gpu.to_ndarray()

                if self.config.full_bispectrum:
                    a_w1 = af.join(0, af.conjg(af.flip(a_w1[1:], 0)), a_w1)
                    t0 = np.concatenate((t0, np.conj(t0[:0:-1])))

                t1 = self.calc_a_w3(t0, f_max_ind, self.config.m, self.a_w3_init, self.indi, self.config.backend)
                a_w3 = to_gpu(t1)
                # ===================================================

                single_spectrum = self.c3(a_w1, a_w2, a_w3) / (self.config.delta_t * (single_window ** order).sum())

            else:  # order 4

                a_w = af.lookup(a_w_all_gpu, af.Array(list(range(f_min_ind, f_max_ind))), dim=0)

                if self.config.corr_data is not None:
                    a_w_all_corr = fft_r2c(window * chunk_corr_gpu, dim0=0, scale=self.config.delta_t)
                    if self.config.random_phase:
                        a_w_all_corr = self.add_random_phase(a_w_all_corr, window_points)

                    a_w_corr = af.lookup(a_w_all_corr, af.Array(list(range(f_min_ind, f_max_ind))), dim=0)
                else:
                    a_w_corr = a_w

                single_spectrum = self.c4(a_w, a_w_corr) / (self.config.delta_t * (single_window ** order).sum())

            self.store_single_spectrum(single_spectrum, order)

    def __prep_f_and_S_arrays(self, orders, f_all_in):
        """
        Helper function to calculate the frequency array and initialize arrays for the storage of spectra and errors.

        This method prepares the frequency array and initializes the error and stationarity arrays based on the given
        parameters and configuration. The `orders` parameter defines which spectra will be calculated, and the arrays
        are set up accordingly.

        Parameters
        ----------
        orders : list of {1, 2, 3, 4}
            Orders of the spectra to be calculated.
        f_all_in : array_like
            An array containing all possible frequencies given the window size and sampling rate OR the list of
            frequencies at which the spectra should be calculated in the calc_spec_poisson function.
        f_max_ind : int
            Index of the maximum frequency in f_all_in array to calculate the spectral values at.

        Returns
        -------
        None

        Notes
        -----
        - This function also initializes `self.S_errs` and `self.S_stationarity_temp`, based on the current configuration.
        - `self.config.m_var` and `self.config.m_stationarity` are taken from the class configuration and determine the
          shape and behavior of the arrays.
        - The print statement at the end indicates the number of points if the order list contains more than one item and
          does not include 1.
        """

        f_max_ind = f_all_in.shape[0]

        for order in orders:
            if order == 3:
                self.freq[order] = f_all_in[:int(f_max_ind // 2)]
                if self.config.full_bispectrum:
                    self.freq[order] = np.concatenate((-self.freq[order][:0:-1], self.freq[order]))
            else:
                self.freq[order] = f_all_in

            if not self.config.turbo_mode:

                if order == 1:
                    self.S_errs[1] = to_gpu(1j * np.ones((1, self.config.m_var)))
                elif order == 2:
                    self.S_errs[2] = to_gpu(1j * np.ones((f_max_ind, self.config.m_var)))
                elif order == 3:
                    if self.config.full_bispectrum:
                        k = 2 * (f_max_ind // 2) - 1
                        self.S_errs[3] = to_gpu(1j * np.ones((f_max_ind // 2, k, self.config.m_var)))
                    else:
                        self.S_errs[3] = to_gpu(1j * np.ones((f_max_ind // 2, f_max_ind // 2, self.config.m_var)))
                elif order == 4:
                    self.S_errs[4] = to_gpu(1j * np.ones((f_max_ind, f_max_ind, self.config.m_var)))

                if self.config.m_stationarity is not None:
                    if order == 1:
                        self.S_stationarity_temp[1] = to_gpu(1j * np.ones((1, self.config.m_stationarity)))
                    elif order == 2:
                        self.S_stationarity_temp[2] = to_gpu(1j * np.ones((f_max_ind, self.config.m_stationarity)))
                    elif order == 3:
                        if self.config.full_bispectrum:
                            k = 2 * (f_max_ind // 2) - 1
                            self.S_stationarity_temp[3] = to_gpu(
                                1j * np.ones((f_max_ind // 2, k, self.config.m_stationarity)))
                        else:
                            self.S_stationarity_temp[3] = to_gpu(
                                1j * np.ones((f_max_ind // 2, f_max_ind // 2, self.config.m_stationarity)))
                    elif order == 4:
                        self.S_stationarity_temp[4] = to_gpu(
                            1j * np.ones((f_max_ind, f_max_ind, self.config.m_stationarity)))

    def __reset_variables(self, orders, f_lists=None):
        """
        Helper function to reset all variables in case spectra are recalculated.

        This function resets variables related to orders, errors, stationarity, and frequency lists using the configuration
        parameters stored in the `config` attribute of the class.

        Parameters
        ----------
        orders : list of {1, 2, 3, 4}
            Orders of the spectra to be calculated. Must be a subset of [1, 2, 3, 4].
        f_lists : list of array_like, optional
            Frequencies at which the spectra will be calculated (can be multiple arrays with different frequency steps).
            Default is None, which will use the frequencies stored in the `config` attribute.

        Returns
        -------
        None
            The method modifies the internal state of the object, updating the error counters, stationarity counters,
            frequency lists, and other attributes related to the specified orders.

        Notes
        -----
        This method utilizes the following attributes from the `config` object:
        - `m`: Spectra for m windows with temporal length T_window are calculated.
        - `m_var`: Number of spectra to calculate the variance from (should be set as high as possible).
        - `m_stationarity`: Number of spectra after which their mean is stored to verify stationarity of the data.
        """
        self.err_counter = {1: 0, 2: 0, 3: 0, 4: 0}
        self.number_of_error_estimates = 0
        self.stationarity_counter = {1: 0, 2: 0, 3: 0, 4: 0}
        for order in orders:
            self.f_lists[order] = f_lists
            self.m[order] = self.config.m
            self.m_var[order] = self.config.m_var
            self.m_stationarity[order] = self.config.m_stationarity
            self.freq[order] = None
            self.S[order] = None
            self.S_gpu[order] = None
            self.S_err_gpu = None
            self.S_err[order] = None
            self.S_errs[order] = []
            self.S_stationarity_temp[order] = []
            self.S_stationarity[order] = []

    def __store_final_spectra(self, orders, n_chunks, n_windows):
        """
        Helper function to move spectra from GPU to RAM at the last step of spectra calculation.

        This method takes the calculated spectra on the GPU and transfers them to the system's RAM, performing some final
        processing and adjustments. It is a part of the internal calculation process and is meant to be called internally
        within the class.

        Parameters
        ----------
        orders : list of {1, 2, 3, 4}
            List of orders of the spectra to be calculated. Only valid orders are 1, 2, 3, and 4.
        n_chunks : int
            Number of calculated spectra chunks. Also used to estimate spectral errors.
        n_windows : int
            Similar to n_chunks; more explanation needed (TODO).

        Notes
        -----
        The method also internally utilizes `m_var`, which is a configuration parameter defining the number of spectra
        used to calculate the variance. It should be set as high as possible for accurate results.

        Returns
        -------
        None
            This method modifies the object's internal state but does not return any value.

        Raises
        ------
        ValueError
            If any of the provided orders are not within the accepted values.
        """
        for order in orders:
            self.S_gpu[order] /= n_chunks
            self.S[order] = self.S_gpu[order].to_ndarray()

            if not self.config.turbo_mode:
                # self.S_err[order] /= n_windows // self.config.m_var * np.sqrt(n_windows)

                self.S_err[order] = 1 / self.number_of_error_estimates * (
                        np.sqrt(self.S_err[order].real) + 1j * np.sqrt(self.S_err[order].imag))

                if self.config.interlaced_calculation:
                    self.S_err[order] /= np.sqrt(2)

    def __find_datapoints_in_windows(self, start_index, start_index_interlaced, frame_number, enough_data):
        """
        Helper function for the calc_spec_poisson function. It locates all click times within a window.

        Parameters
        ----------
        start_index : int
            Index (in the dataset) of the last timestamp in the previous window.
        frame_number : int
            Number of the current frame/spectra to be calculated.
        enough_data : bool
            Used to terminate calculation if the last window is longer than the latest timestamp.

        Returns
        -------
        windows : list of array or None
            Click times within each window. None if no click times found in a window.
        start_index : int
            Updated starting index for the next iteration.
        enough_data : bool
            Updated flag indicating whether there's enough data to continue.

        Notes
        -----
        This method utilizes the `config` attribute of the class, which should contain the configuration parameters.
        For example, `self.config.m` represents the number of windows with temporal length `T_window`,
        and `self.data` represents the full dataset of time stamps.
        """
        windows = []
        windows_interlaced = []
        for i in range(self.config.m):
            end_index, end_index_interlaced = find_end_index(self.data, start_index,
                                                             self.T_window, self.config.m, frame_number, i)

            if end_index == -1:
                enough_data = False
                break
            else:
                if start_index == end_index:
                    windows.append(None)
                else:
                    windows.append(self.data[start_index:end_index])
                start_index = end_index

                if start_index_interlaced == end_index_interlaced:
                    windows_interlaced.append(None)
                else:
                    windows_interlaced.append(self.data[start_index_interlaced:end_index_interlaced])
                start_index_interlaced = end_index_interlaced

        return windows, windows_interlaced, start_index, start_index_interlaced, enough_data

    def process_order(self):
        """
        Process the order of calculation based on the configuration.

        Returns
        -------
        list of int
            List containing integers representing the orders to be processed. If the configuration
            specifies 'all', the list will contain [1, 2, 3, 4]. Otherwise, it will return the
            value specified in the configuration.

        Notes
        -----
        The order is retrieved from the configuration object associated with the instance.
        """
        if self.config.order_in == 'all':
            return [1, 2, 3, 4]
        else:
            return self.config.order_in

    def reset_and_backend_setup(self):
        """
        Resets internal variables and sets up the backend for computation.

        Returns
        -------
        list of int
            List of orders that are to be processed.

        Side Effects
        ------------
        - The backend is set based on the configuration's backend value.
        - Internal variables are reset by calling the private method `__reset_variables`.

        Notes
        -----
        This method relies on the configuration object associated with the instance.
        """
        af.set_backend(self.config.backend)
        orders = self.process_order()
        self.orders = orders
        self.__reset_variables(orders, f_lists=self.config.f_lists)
        return orders

    def setup_data_calc_spec(self, orders):
        """
        Set up the data and calculate spectral parameters for the analysis.

        This method configures parameters related to the windowing and frequency
        calculations for spectral analysis. It calculates and returns several
        values, including the window points, frequency array, frequency mask,
        maximum frequency index, and number of windows.

        Notes
        -----
        - `self.config.corr_shift` is modified within this function to convert the shift from seconds to time steps.
        - The number of windows is reduced if `self.config.corr_shift` is present.
        - The actual `T_window` value is printed during execution.
        - The function raises a warning if `self.config.f_max` is too high and adjusts it accordingly.
        - Ensure that the associated configuration (`self.config`) has been initialized properly before calling this method.

        Returns
        -------
        window_points : int
            The number of data points in each window.
        freq_all_freq : ndarray
            Array of all frequency points considered.
        f_mask : ndarray (boolean)
            Mask array for frequencies less than or equal to `self.config.f_max`.
        f_max_ind : int
            Index corresponding to the maximum frequency.
        n_spectra : int
            Number of windows for spectral analysis, considering shifts and overlaps.

        See Also
        --------
        SpectrumConfig : Configuration class containing the associated parameters.
        """

        f_max_actual = 1 / (2 * self.config.delta_t)

        if self.config.f_max is None:
            self.config.f_max = f_max_actual

        window_length_factor = f_max_actual / (self.config.f_max - self.config.f_min)

        # Spectra for m windows with temporal length T_window are calculated.
        self.T_window = (self.config.spectrum_size - 1) * 2 * self.config.delta_t * window_length_factor
        self.config.corr_shift /= self.config.delta_t  # conversion of shift in seconds to shift in dt
        n_data_points = self.data.shape[0]
        window_points = int(np.round(self.T_window / self.config.delta_t))

        if self.config.turbo_mode:
            self.config.m = int(n_data_points // window_points - 0.5)
            m = self.config.m

        else:
            # Set m to be as high as possible for the given m_var in the config if m is not given
            # Check if enough data points are there to perform the calculation (added window_points // 2 due to interlaced calculation)
            if not window_points * self.config.m + window_points // 2 < n_data_points:
                m = (n_data_points - window_points // 2) // window_points
                if m < max(orders):
                    # Spectral resolution has to be decreased
                    max_spectrum_size = window_points // (2 * window_length_factor) + 1
                    raise ValueError("Not enough data points for the set spectrum_size. The maximum nuber of "
                                     f"points in the spectrum of order {max(orders)} is {max_spectrum_size}.")

                print(f"Values have been changed due to too little data. old: m = {self.config.m}, new: m = {m}")
                self.config.m = m
            else:
                m = self.config.m

        # Check m_var and m_stationarity
        number_of_spectra = n_data_points // (window_points * m + window_points // 2)
        if number_of_spectra < self.config.m_var and not self.config.turbo_mode:
            m_var = n_data_points // (window_points * m + window_points // 2)
            if m_var < 2:
                raise ValueError(f"Not enough data points to estimate error from {self.config.m_var} spectra. Consider "
                                 f"decreasing the resolution of the spectra or the variable m_var.")
            else:
                print(
                    f"Values have been changed due to too little data. old: m_var = {self.config.m_var}, new: m_var = {m_var}")
            self.config.m_var = m_var

        if self.config.m_stationarity is not None and not self.config.turbo_mode:
            if number_of_spectra < self.config.m_stationarity:
                raise ValueError(
                    f"Not enough data points to calculate {self.config.m_stationarity} different spectra "
                    f"to visualize changes in the power spectrum over time. Consider "
                    f"decreasing the resolution of the spectra or the variable m_stationary.")

        if self.config.verbose:
            print('T_window: {:.3e} {}'.format(window_points * self.config.delta_t, self.t_unit))
        self.window_points = window_points

        n_spectra = int(np.floor(n_data_points / (m * window_points)))
        n_spectra = int(
            np.floor(n_spectra - self.config.corr_shift / (
                    m * window_points)))  # number of windows is reduced if corr shifted

        self.fs = 1 / self.config.delta_t
        freq_all_freq = rfftfreq(int(window_points), self.config.delta_t)
        if self.config.verbose:
            print('Maximum frequency: {:.3e} {}'.format(np.max(freq_all_freq), self.config.f_unit))

        # ------ Check if f_max is too high ---------
        f_mask = freq_all_freq <= self.config.f_max
        f_max_ind = sum(f_mask)

        # ------ Find index of f_min --------
        f_mask = freq_all_freq < self.config.f_min
        f_min_ind = sum(f_mask)

        return m, window_points, freq_all_freq, f_max_ind, f_min_ind, n_spectra

    def check_minimum_window_length(self, n_parts=4):

        # -------data setup---------
        if self.config.data is None:
            self.data = self.import_data()
        else:
            self.data = self.config.data
        if self.config.delta_t is None:
            raise MissingValueError('Missing value for delta_t')

        # Function to calculate autocorrelation using FFT
        def autocorrelation_via_fft(x):
            n = len(x)
            # print('n:', n)
            f = np.fft.fft(x - np.mean(x))
            result = np.fft.ifft(f * np.conjugate(f)).real
            # print('1:', result.shape)
            result = result[:n]
            result /= result[0]  # Normalize the result
            return result

        # Splitting the dataset into 10 parts
        data_length = self.data.shape[0]
        part_length = data_length // n_parts
        datasets = [self.data[i * part_length:(i + 1) * part_length] for i in range(n_parts)]

        # Calculating autocorrelation for each part using FFT
        autocorrelations_fft = np.array([autocorrelation_via_fft(dataset) for dataset in datasets])

        # Calculating mean and standard deviation for each delta_t
        mean_autocorr_fft = np.mean(autocorrelations_fft, axis=0)

        # Plotting the autocorrelation with error bars
        delta_t = self.config.delta_t
        n_points = part_length // 2  # Restrict to realistic delta_t values

        # plt.errorbar(np.arange(n_points) * delta_t, mean_autocorr_fft[:n_points], yerr=std_autocorr_fft[:n_points], fmt='-o', label='Mean autocorrelation ± σ')
        # plt.plot(np.arange(n_points) * delta_t, mean_autocorr_fft[:n_points], '-', label='Mean autocorrelation ± σ')
        # plt.xlabel('Time lag (delta_t)')
        # plt.ylabel('Autocorrelation')
        # plt.title('Autocorrelation with Error Bars using FFT')
        # plt.legend()
        ##plt.ylim([0,200])
        # plt.yscale('log')
        # plt.show()

        # Finding the largest significant delta_t
        largest_significant_delta_t = np.where(mean_autocorr_fft[:n_points] < np.zeros(n_points))[0][0]

        print("The largest delta_t for significant autocorrelation:", largest_significant_delta_t * delta_t, 's',
              'corresponds to', 1 / (largest_significant_delta_t * delta_t), self.config.f_unit)
        print("A window length of at least:", 10 * largest_significant_delta_t * delta_t, self.t_unit, 'is recommended')

    def calc_spec(self):
        """
        Calculation of spectra of orders 1 to 4 with the ArrayFire library.

        This function relies on the configuration parameters defined in the `SpectrumConfig` class.

        Notes
        -----
        - `order_in`, `spectrum_size`, `f_max`, etc.:
            Refer to the `SpectrumConfig` class for details on these parameters.
        - The computation could be expensive for `spectrum_size` greater than 1000 for orders 3 and 4.
        - Several experimental and TODO flags are available; consult the `SpectrumConfig` class for guidance.
        - Ensure that `delta_t` is set; a MissingValueError will be raised if it is None.

        Returns
        -------
        freq : array_like
            Frequencies at which the spectra are calculated.
        S : array_like
            Calculated spectra for the given orders.
        S_err : array_like
            Errors associated with the calculated spectra.

        See Also
        --------
        SpectrumConfig : Class containing all configuration parameters used in this method.
        """

        orders = self.reset_and_backend_setup()

        # -------data setup---------
        if self.config.data is None:
            self.data = self.import_data()
        else:
            self.data = self.config.data
        if self.config.delta_t is None:
            raise MissingValueError('Missing value for delta_t')

        n_chunks = 0

        if self.config.corr_path is not None:
            self.config.corr_data = self.import_corr_data()

        m, window_points, freq_all_freq, f_max_ind, f_min_ind, n_windows = self.setup_data_calc_spec(orders)
        self.window_points = window_points
        self.n_windows = n_windows

        for order in orders:
            self.m[order] = m

        # ===== New alogrithm needs to intitialize matrices only once =====
        self.a_w3_init = self.a_w3_gen(f_max_ind, self.config.m)
        self.indi = self.index_generation_to_aw_3(f_max_ind)
        # ==================================================================

        single_window, _ = cgw(int(window_points), self.fs, sigma_t=self.config.sigma_t)
        window = to_gpu(np.array(m * [single_window]).flatten().reshape((window_points, 1, m), order='F'))

        self.__prep_f_and_S_arrays(orders, freq_all_freq[f_min_ind:f_max_ind])

        for i in tqdm(np.arange(0, n_windows, 1), leave=False):

            # Calculate a spectra ones without shift and ones interlaced.
            if self.config.interlaced_calculation:
                shift_iterator = [0, window_points // 2]
            else:
                shift_iterator = [0]

            for window_shift in shift_iterator:

                chunk = self.data[
                        int(i * (window_points * m) + window_shift): int((i + 1) * (window_points * m) + window_shift)]

                if not self.first_frame_plotted and self.config.show_first_frame:
                    self.plot_first_frame(chunk, window_points)
                    self.first_frame_plotted = True

                if not chunk.shape[0] == window_points * m:
                    break

                chunk_gpu = to_gpu(chunk.reshape((window_points, 1, m), order='F'))

                if self.config.corr_default == 'white_noise':  # use white noise to check for false correlations
                    chunk_corr = np.random.randn(window_points, 1, m)
                    chunk_corr_gpu = to_gpu(chunk_corr)
                elif self.config.corr_data is not None:
                    chunk_corr = self.config.corr_data[int(i * (window_points * m) + self.config.corr_shift): int(
                        (i + 1) * (window_points * m) + self.config.corr_shift)]
                    chunk_corr_gpu = to_gpu(chunk_corr.reshape((window_points, 1, m), order='F'))
                else:
                    chunk_corr_gpu = None

                # ---------count windows-----------
                n_chunks += 1

                # -------- perform fourier transform ----------
                if self.config.rect_win:
                    ones = to_gpu(
                        np.array(m * [np.ones_like(single_window)]).flatten().reshape((window_points, 1, m), order='F'))
                    a_w_all_gpu = fft_r2c(ones * chunk_gpu, dim0=0, scale=self.config.delta_t)
                else:
                    a_w_all_gpu = fft_r2c(window * chunk_gpu, dim0=0, scale=self.config.delta_t)

                # --------- modify data ---------
                if self.config.filter_func:
                    pre_filter = self.config.filter_func(self.freq[2])
                    filter_mat = to_gpu(
                        np.array(m * [1 / pre_filter]).flatten().reshape((a_w_all_gpu.shape[0], 1, m), order='F'))
                    a_w_all_gpu = filter_mat * a_w_all_gpu

                if self.config.random_phase:
                    a_w_all_gpu = self.add_random_phase(a_w_all_gpu, window_points)

                # --------- calculate spectra ----------
                self.fourier_coeffs_to_spectra(orders, a_w_all_gpu, f_max_ind, f_min_ind, single_window,
                                               window, chunk_corr_gpu=chunk_corr_gpu, window_points=window_points)
                if n_chunks == self.config.break_after:
                    break

        self.__store_final_spectra(orders, n_chunks, n_windows)
        return self.freq, self.S, self.S_err

    def calc_spec_poisson(self, n_reps=10,
                          sigma_t=0.14, exp_weighting=True, T_window=None):
        """
        Calculate spectra using the Poisson method and average over `n_reps` repetitions.

        The parameters required for this function are expected to be contained in the
        configuration class associated with this object. Make sure the relevant parameters
        are set in the configuration class before calling this function.

        Parameters
        ----------
        n_reps : int, optional
            Number of repetitions for spectra calculation. Default is 10.
        f_lists : list of arrays, optional
            Frequencies at which the spectra will be calculated (can include multiple arrays with different frequency steps). Default is None.
        sigma_t : float, optional
            Width of approximate confined Gaussian windows. Default is 0.14.

        Returns
        -------
        freq : array
            Frequencies at which the spectra were calculated.
        S : dict
            Dictionary containing the calculated spectra for different orders.
        S_err : dict
            Dictionary containing the error in the calculated spectra for different orders.

        Notes
        -----
        The parameters `order_in`, `spectrum_size`, `f_max`, `backend`, `m`, `m_var`, `m_stationarity`,
        `full_import`, `scale_t`, and `rect_win` are expected to be available in the configuration class
        associated with this object.
        """

        orders = self.process_order()

        all_S = []
        all_S_err = []

        for i in range(n_reps):
            f, S, S_err = self.calc_spec_poisson_one_spectrum(sigma_t=sigma_t,
                                                              exp_weighting=exp_weighting, T_window=T_window)

            all_S.append(S)
            all_S_err.append(S_err)

        for i in orders:
            self.S[i] = sum([S[i] for S in all_S]) / n_reps
            self.S_err[i] = sum([S_err[i] for S_err in all_S_err]) / n_reps

        return self.freq, self.S, self.S_err

    def setup_data_calc_spec_poisson(self, T_window=None):

        """
        Set up data for the calculation of the spectral analysis using Poisson statistics.

        This function computes various frequency-related parameters needed for the spectral analysis,
        based on the given frequency lists and the configuration provided in `self.config`.

        Parameters
        -------
        T_window : (float, None)
            If None, T_window is estimated from the freqeuncy spacing in f_lists. It can be chosen manually since
            computation time might be lower for a larger window.

        Returns
        -------
        f_list : ndarray
            Array of frequency values.
        f_max_ind : int
            Maximum index of the frequency list.
        n_windows : int
            Number of windows required for the spectral analysis.
        w_list : ndarray
            Array of angular frequency values.
        w_list_gpu : GPU array
            Array of angular frequency values on the GPU.

        Notes
        -----
        The function assumes that `self.config` is an instance of `SpectrumConfig` containing valid
        configuration parameters, including `f_max` and `spectrum_size`.
        The function also relies on GPU functions, so appropriate setup for GPU computation is assumed.
        """

        if self.config.f_lists is not None:
            f_list = np.hstack(self.config.f_lists)
            delta_f = np.abs(f_list - np.roll(f_list, 1)).min()
        else:
            delta_f = self.config.f_max / (self.config.spectrum_size - 1)
            f_list = np.arange(0, self.config.f_max + delta_f, delta_f)

        if T_window is None:
            self.T_window = 1 / delta_f
        else:
            self.T_window = T_window

        f_max_ind = len(f_list)
        w_list = 2 * np.pi * f_list
        w_list_gpu = to_gpu(w_list)
        n_windows = int(self.data[-1] // (self.T_window * self.config.m))

        if self.config.m_var is None:
            self.config.m_var = n_windows

        return f_list, f_max_ind, n_windows, w_list, w_list_gpu

    def calc_spec_poisson_one_spectrum(self, sigma_t=0.14, exp_weighting=True, T_window=None):
        """
        Calculate the Poisson spectrum for one spectrum based on the configuration stored in self.config.

        Parameters
        ----------
        sigma_t : float, optional
            Width of approximate confined Gaussian windows. Default is 0.14.

        Returns
        -------
        freq : array_like
            Frequencies at which the spectra were calculated.
        S : array_like
            Calculated spectra values.
        S_err : array_like
            Errors in the calculated spectra.

        Notes
        -----
        - Most of the parameters required by this function are extracted from the `self.config` object. Make sure to set up
          the configuration appropriately before calling this function.
        - This function is computationally expensive for a large spectrum_size (e.g., more than 1000 points) for order 3 and 4.
        - If `rect_win` is set to True in `self.config`, no window function will be applied to the window.
        """

        orders = self.reset_and_backend_setup()

        # -------data setup---------
        if self.config.data is None:
            self.data = self.import_data()
        else:
            self.data = self.config.data
        if self.config.delta_t is None:
            self.config.delta_t = 1

        f_list, f_max_ind, n_windows, w_list, w_list_gpu = self.setup_data_calc_spec_poisson(T_window=T_window)

        n_chunks = 0
        start_index = 0
        start_index_interlaced = find_start_index_interlaced(self.data, self.T_window)
        enough_data = True

        print('number of points:', f_list.shape[0])
        print('delta f:', f_list[1] - f_list[0])

        self.__prep_f_and_S_arrays(orders, f_list)

        # ===== New alogrithm needs to intitialize matrices only once =====
        self.a_w3_init = self.a_w3_gen(f_max_ind, self.config.m)
        self.indi = self.index_generation_to_aw_3(f_max_ind)
        # ==================================================================

        single_window, N_window_full = self.calc_single_window()
        self.config.delta_t = self.T_window / N_window_full  # 70 as defined in function apply_window(...)

        # zeros_on_gpu = to_gpu(1j * np.zeros_like(w_list))
        a_w_all = 0 * 1j * np.ones((w_list.shape[0], self.config.m))
        a_w_all_gpu = to_gpu(a_w_all.reshape((len(f_list), 1, self.config.m), order='F'))

        for frame_number in tqdm(range(n_windows)):

            windows, windows_interlaced, start_index, start_index_interlaced, enough_data = self.__find_datapoints_in_windows(
                start_index,
                start_index_interlaced,
                frame_number,
                enough_data)

            if not enough_data:
                break

            if self.config.interlaced_calculation:
                iterator = zip([False, True], [windows, windows_interlaced])
            else:
                iterator = zip([False], [windows])

            for is_interlaced, frame in iterator:

                n_chunks += 1

                for i, t_clicks in enumerate(frame):

                    if t_clicks is not None:

                        t_clicks_minus_start = t_clicks - i * self.T_window - self.config.m * self.T_window * frame_number
                        if is_interlaced:
                            t_clicks_minus_start -= self.T_window / 2

                        if self.config.rect_win:
                            t_clicks_windowed = np.ones_like(t_clicks_minus_start)
                        else:
                            t_clicks_windowed, single_window = apply_window(self.T_window,
                                                                            t_clicks_minus_start,
                                                                            1 / self.config.delta_t,
                                                                            sigma_t=self.config.sigma_t)

                        # ------ GPU --------
                        t_clicks_minus_start_gpu = to_gpu(t_clicks_minus_start)

                        if not exp_weighting:
                            # ------- uniformly weighted clicks -------
                            t_clicks_windowed_gpu = to_gpu(t_clicks_windowed).as_type(af.Dtype.c64)

                        else:
                            # ------- exponentially weighted clicks -------
                            exp_random_numbers = np.random.exponential(1, t_clicks_windowed.shape[0])
                            t_clicks_windowed_gpu = to_gpu(t_clicks_windowed * exp_random_numbers).as_type(af.Dtype.c64)

                        temp1 = af.exp(1j * af.matmulNT(w_list_gpu, t_clicks_minus_start_gpu))
                        # temp2 = af.tile(t_clicks_windowed_gpu.T, w_list_gpu.shape[0])
                        # a_w_all_gpu[:, 0, i] = af.sum(temp1 * temp2, dim=1)

                        a_w_all_gpu[:, 0, i] = af.matmul(temp1, t_clicks_windowed_gpu)

                    else:
                        continue
                        # a_w_all_gpu is initialized as zeros
                        # a_w_all_gpu[:, 0, i] = zeros_on_gpu

                f_min_ind = 0
                self.fourier_coeffs_to_spectra(orders, a_w_all_gpu, f_max_ind, f_min_ind, single_window)

        self.__store_final_spectra(orders, n_chunks, n_windows)

        return self.freq, self.S, self.S_err

    def plot(self, plot_config: PlotConfig = None):
        if plot_config:
            self.plot_config = plot_config
        elif self.plot_config is None:
            self.plot_config = PlotConfig()  # Use default plot configuration

        from .spectrum_plotter import SpectrumPlotter
        plotter = SpectrumPlotter(self, self.plot_config)
        return plotter.plot()

    def stationarity_plot(self, plot_config: PlotConfig = None):
        if plot_config:
            self.plot_config = plot_config
        elif self.plot_config is None:
            self.plot_config = PlotConfig()  # Use default plot configuration

        from .spectrum_plotter import SpectrumPlotter
        plotter = SpectrumPlotter(self, self.plot_config)
        return plotter.stationarity_plot()
