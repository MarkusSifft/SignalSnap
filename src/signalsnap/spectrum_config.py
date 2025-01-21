# This file is part of signalsnap: Signal Analysis In Python Made Easy
# Copyright (c) 2020 and later, Markus Sifft and Daniel Hägele.
#
# This software is provided under the terms of the 3-Clause BSD License.
# For details, see the LICENSE file in the root of this repository or
# https://opensource.org/licenses/BSD-3-Clause

import numpy as np


class SpectrumConfig:
    """
    Configuration class for spectrum analysis, storing parameters, data, and performing validity checks.

    Parameters
    ----------
    path : str, optional
        Path to h5 file with stored signal. Default is None.
    group_key : str, optional
        Group key for h5 file. Default is None.
    dataset : str, optional
        Name of the dataset in h5 file. Default is None.
    delta_t : float, optional
        Inverse of the sampling rate of the signal. Must be positive. Default is None.
    data : array_like, optional
        Signal to be analyzed. Default is None.
    corr_data : array_like, optional
        Second signal used for correlation. Default is None.

    Correlation Parameters
    ----------------------
    corr_path : str, optional
        Path to h5 file with second signal for correlation. Default is None.
    corr_group_key : str, optional
        Group key for h5 file with correlation signal. Default is None.
    corr_dataset : str, optional
        Name of the dataset in h5 file with correlation signal. Default is None.
    corr_shift : int, optional
        Non-negative integer or None. Default is 0.

    Frequency Parameters
    --------------------
    f_unit : {'Hz', 'kHz', 'MHz', 'GHz', 'THz', 'mHz'}, optional
        Frequency unit. Default is 'Hz'.
    f_max : float, optional
        Upper frequency limit for spectral values calculation. Must be positive. Default is None.

    Computational Parameters
    ------------------------
    backend : {'cpu', 'opencl', 'cuda'}, optional
        Backend for computation. Default is 'cpu'.
    spectrum_size : int, optional
        Number of points in a window, must be positive. Default is 100.
    order_in : str or list of {1, 2, 3, 4}, optional
        Order for calculation, 'all' or list with numbers between 1 and 4. Default is 'all'.

    Additional Parameters
    ---------------------
    filter_func : callable or False, optional
        Filtering function. Default is False.
    verbose, coherent, random_phase, rect_win, full_import, show_first_frame : bool, optional
        Various boolean flags. Defaults are True, False, False, False, True, True respectively.
    corr_default : {None, 'white noise'}, optional
        Default is None.
    break_after, m, m_var, m_stationarity, window_shift : int, optional
        Various integer parameters with constraints. Defaults are 1e6, 10, 10, None, 1 respectively.
    turbo_mode : bool (experimental)
        If set, no error is calculated and m is set as high as possible, effectively calculating only one spectrum parallelized.
        Default is False.

    Raises
    ------
    ValueError
        If any parameter does not meet its constraints.

    Notes
    -----
    Ensure that the provided paths, group keys, and dataset names are valid, as this class does not handle file reading.
    """

    def __init__(self, path=None, group_key=None, dataset=None, delta_t=None, data=None,
                 corr_data=None, corr_path=None, corr_group_key=None, corr_dataset=None,
                 f_unit='Hz', f_max=None, f_min=0, f_lists=None, backend='cpu', spectrum_size=100, order_in='all',
                 corr_shift=0, filter_func=False, verbose=True, coherent=False, corr_default=None,
                 break_after=1e6, m=10, m_var=10, m_stationarity=None, interlaced_calculation=True,
                 random_phase=False, full_bispectrum=False, sigma_t=0.14,
                 rect_win=False, full_import=True, show_first_frame=True, turbo_mode=False):

        if path is not None and not isinstance(path, str):
            raise ValueError("path must be a string or None.")
        if group_key is not None and not isinstance(group_key, str):
            raise ValueError("group_key must be a string or None.")
        if dataset is not None and not isinstance(dataset, str):
            raise ValueError("dataset must be a string or None.")
        #if delta_t is not None and (not isinstance(delta_t, float) or delta_t <= 0):
        #    raise ValueError("delta_t must be a positive float or None.")
        if data is not None and not isinstance(data, np.ndarray):
            raise ValueError("data must be a numpy array or None.")
        if corr_data is not None and not isinstance(corr_data, np.ndarray):
            raise ValueError("corr_data must be a numpy array or None.")
        if corr_path is not None and not isinstance(corr_path, str):
            raise ValueError("corr_path must be a string or None.")
        if f_unit not in ['Hz', 'kHz', 'MHz', 'GHz', 'THz', 'mHz']:
            raise ValueError("f_unit must be one of 'Hz', 'kHz', 'MHz', 'GHz', 'THz', 'mHz'.")
        if backend not in ['cpu', 'opencl', 'cuda']:
            raise ValueError("backend must be one of 'cpu', 'opencl', 'cuda'.")
        if (not isinstance(spectrum_size, int) or spectrum_size <= 0) and f_lists is None:
            raise ValueError("spectrum_size must be a positive integer.")

        if order_in != 'all' and (not isinstance(order_in, list) or not all(1 <= i <= 4 for i in order_in)):
            raise ValueError("order_in must be 'all' or a list containing one or more numbers between 1 and 4.")
        if order_in == 'all':
            largest_order = 4
        elif isinstance(order_in, list):
            largest_order = max(order_in)
        else:
            raise ValueError("order_in must be 'all' or a list containing one or more numbers between 1 and 4.")

        if filter_func is not False and not callable(filter_func):
            raise ValueError("filter_func must be a callable function or False.")
        if corr_default not in [None, 'white noise']:
            raise ValueError("corr_default must be None or 'white noise'.")
        if isinstance(break_after, float) and break_after.is_integer():
            break_after = int(break_after)
        if not isinstance(break_after, int) or break_after <= 0:
            raise ValueError("break_after must be a positive integer.")
        if m is not None and (not isinstance(m, int) or m < largest_order):
            raise ValueError(
                f"m must be larger or equal to the largest number in order_in ({largest_order}), or larger or equal to 4 if 'all' is used.")
        if m_var is not None and (not isinstance(m_var, int) or m_var <= 2):
            raise ValueError("m_var must be larger or equal to 2.")
        if m_stationarity is not None and (not isinstance(m_stationarity, int) or m_stationarity <= 0):
            raise ValueError("m_stationarity must be a positive integer or None.")
        if not isinstance(interlaced_calculation, bool):
            raise ValueError("window_shift must be a boolean value (True or False).")
        if not isinstance(random_phase, bool):
            raise ValueError("random_phase must be a boolean value (True or False).")
        if not isinstance(rect_win, bool):
            raise ValueError("rect_win must be a boolean value (True or False).")
        if not isinstance(full_import, bool):
            raise ValueError("full_import must be a boolean value (True or False).")
        if not isinstance(show_first_frame, bool):
            raise ValueError("show_first_frame must be a boolean value (True or False).")
        if (f_max is not None and (not isinstance(f_max, (float, int)) or f_max <= 0)) and f_lists is None:
            raise ValueError("f_max must be a positive number or None.")
        if (not isinstance(f_min, (float, int)) or f_min < 0) and f_lists is None:
            raise ValueError("f_min must be a positive number or 0.")
        if corr_shift is not None and (not isinstance(corr_shift, int) or corr_shift < 0):
            raise ValueError("corr_shift must be a non-negative integer or None.")
        if not isinstance(verbose, bool):
            raise ValueError("verbose must be a boolean value (True or False).")
        if not isinstance(coherent, bool):
            raise ValueError("coherent must be a boolean value (True or False).")
        if not isinstance(turbo_mode, bool):
            raise ValueError("turbo_mode must be a boolean value (True or False).")

        if f_min is not None:
            if isinstance(order_in, list):
                if f_min > 0 and 3 in order_in:
                    order_in.remove(3)
                    print("Order 3 has been removed from order_in as f_min must be 0 to calculate the bispectrum.")
            if isinstance(order_in, str):
                if f_min > 0 and order_in == 'all':
                    order_in = [1, 2, 4]
                    print("Order 3 has been removed from order_in as f_min must be 0 to calculate the bispectrum.")

        if f_lists is not None:
            if isinstance(order_in, list):
                if 3 in order_in:
                    order_in.remove(3)
                    print("Order 3 has been removed from order_in as f_min must be 0 to calculate the bispectrum.")
            if isinstance(order_in, str):
                if order_in == 'all':
                    order_in = [1, 2, 4]
                    print("Order 3 has been removed from order_in as f_min must be 0 to calculate the bispectrum.")

        self.path = path
        self.group_key = group_key
        self.dataset = dataset
        self.delta_t = delta_t
        self.data = data
        self.corr_data = corr_data
        self.corr_path = corr_path
        self.corr_group_key = corr_group_key
        self.corr_dataset = corr_dataset
        self.f_unit = f_unit
        self.f_max = f_max
        self.f_lists = f_lists
        self.backend = backend
        self.spectrum_size = spectrum_size
        self.order_in = order_in
        self.f_max = f_max
        self.f_min = f_min
        self.corr_shift = corr_shift
        self.filter_func = filter_func
        self.verbose = verbose
        self.coherent = coherent
        self.corr_default = corr_default
        self.break_after = break_after
        self.m = m
        self.m_var = m_var
        self.m_stationarity = m_stationarity
        self.interlaced_calculation = interlaced_calculation
        self.random_phase = random_phase
        self.rect_win = rect_win
        self.full_import = full_import
        self.show_first_frame = show_first_frame
        self.turbo_mode = turbo_mode
        self.full_bispectrum = full_bispectrum
        self.sigma_t = sigma_t
