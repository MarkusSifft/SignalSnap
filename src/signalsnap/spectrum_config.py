import numpy as np


class SpectrumConfig:
    def __init__(self, path=None, group_key=None, dataset=None, delta_t=None, data=None,
                 corr_data=None, corr_path=None, corr_group_key=None, corr_dataset=None,
                 f_unit='Hz', f_max=None, backend='cpu', spectrum_size=100, order_in='all',
                 corr_shift=0, filter_func=False, verbose=True, coherent=False, corr_default=None,
                 break_after=1e6, m=10, m_var=10, m_stationarity=None, window_shift=1, random_phase=False,
                 rect_win=False, full_import=True, show_first_frame=True):

        if path is not None and not isinstance(path, str):
            raise ValueError("path must be a string or None.")
        if group_key is not None and not isinstance(group_key, str):
            raise ValueError("group_key must be a string or None.")
        if dataset is not None and not isinstance(dataset, str):
            raise ValueError("dataset must be a string or None.")
        if delta_t is not None and (not isinstance(delta_t, float) or delta_t <= 0):
            raise ValueError("delta_t must be a positive float or None.")
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
        if not isinstance(spectrum_size, int) or spectrum_size <= 0:
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
        if not isinstance(break_after, int) or break_after <= 0:
            raise ValueError("break_after must be a positive integer.")
        if not isinstance(m, int) or m < largest_order:
            raise ValueError(
                f"m must be larger or equal to the largest number in order_in ({largest_order}), or larger or equal to 4 if 'all' is used.")
        if not isinstance(m_var, int) or m_var <= 2:
            raise ValueError("m_var must be larger or equal to 2.")
        if m_stationarity is not None and (not isinstance(m_stationarity, int) or m_stationarity <= 0):
            raise ValueError("m_stationarity must be a positive integer or None.")
        if not isinstance(window_shift, int) or window_shift <= 0:
            raise ValueError("window_shift must be a positive integer.")
        if not isinstance(random_phase, bool):
            raise ValueError("random_phase must be a boolean value (True or False).")
        if not isinstance(rect_win, bool):
            raise ValueError("rect_win must be a boolean value (True or False).")
        if not isinstance(full_import, bool):
            raise ValueError("full_import must be a boolean value (True or False).")
        if not isinstance(show_first_frame, bool):
            raise ValueError("show_first_frame must be a boolean value (True or False).")
        if f_max is not None and (not isinstance(f_max, float) or f_max <= 0):
            raise ValueError("f_max must be a positive float or None.")
        if corr_shift is not None and (not isinstance(corr_shift, int) or corr_shift < 0):
            raise ValueError("corr_shift must be a non-negative integer or None.")
        if not isinstance(verbose, bool):
            raise ValueError("verbose must be a boolean value (True or False).")
        if not isinstance(coherent, bool):
            raise ValueError("coherent must be a boolean value (True or False).")

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
        self.backend = backend
        self.spectrum_size = spectrum_size
        self.order_in = order_in
        self.f_max = f_max
        self.corr_shift = corr_shift
        self.filter_func = filter_func
        self.verbose = verbose
        self.coherent = coherent
        self.corr_default = corr_default
        self.break_after = break_after
        self.m = m
        self.m_var = m_var
        self.m_stationarity = m_stationarity
        self.window_shift = window_shift
        self.random_phase = random_phase
        self.rect_win = rect_win
        self.full_import = full_import
        self.show_first_frame = show_first_frame
