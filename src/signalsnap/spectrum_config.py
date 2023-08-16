class SpectrumConfig:
    def __init__(self, path=None, group_key=None, dataset=None, delta_t=None, data=None,
                 corr_data=None, corr_path=None, corr_group_key=None, corr_dataset=None,
                 f_unit='Hz', f_max=None, backend='cpu', spectrum_size=100, order_in='all',
                 corr_shift=0, filter_func=False, verbose=True, coherent=False, corr_default=None,
                 break_after=1e6, m=10, m_var=10, m_stationarity=None, window_shift=1, random_phase=False,
                 rect_win=False, full_import=True, show_first_frame=True):
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