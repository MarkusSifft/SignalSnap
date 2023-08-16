def calc_spec(self, order_in, spectrum_size, f_max, backend='cpu', scaling_factor=1,
                  corr_shift=0, filter_func=False, verbose=True, coherent=False, corr_default=None,
                  break_after=1e6, m=10, m_var=10, m_stationarity=None, window_shift=1, random_phase=False,
                  rect_win=False, full_import=True, show_first_frame=True):
    af.set_backend(backend)

    if order_in == 'all':
        orders = [1, 2, 3, 4]
    else:
        orders = order_in

    self.__reset_variables(orders, m, m_var, m_stationarity)

    # -------data setup---------
    if self.data is None:
        self.data = self.import_data(self.path, self.group_key, self.dataset, full_import=full_import)
    if self.delta_t is None:
        raise MissingValueError('Missing value for delta_t')

    n_chunks = 0
    f_max_actual = 1 / (2 * self.delta_t)
    window_length_factor = f_max_actual / f_max

    # Spectra for m windows with temporal length T_window are calculated.
    self.T_window = (spectrum_size - 1) * 2 * self.delta_t * window_length_factor

    corr_shift /= self.delta_t  # conversion of shift in seconds to shift in dt

    window_points = int(np.round(self.T_window / self.delta_t))
    print('Actual T_window:', window_points * self.delta_t)
    self.window_points = window_points

    if self.corr_data is None and not corr_default == 'white_noise' and self.corr_path is not None:
        corr_data = self.import_data(self.corr_data_path, self.corr_group_key, self.corr_dataset,
                                     full_import=full_import)
    elif self.corr_data is not None:
        corr_data = self.corr_data
    else:
        corr_data = None

    n_data_points = self.data.shape[0]
    n_windows = int(np.floor(n_data_points / (m * window_points)))
    n_windows = int(
        np.floor(n_windows - corr_shift / (m * window_points)))  # number of windows is reduced if corr shifted

    self.fs = 1 / self.delta_t
    freq_all_freq = rfftfreq(int(window_points), self.delta_t)
    if verbose:
        print('Maximum frequency:', np.max(freq_all_freq))

    # ------ Check if f_max is too high ---------
    f_mask = freq_all_freq <= f_max
    f_max_ind = sum(f_mask)

    single_window, _ = cgw(int(window_points), self.fs)
    window = to_gpu(np.array(m * [single_window]).flatten().reshape((window_points, 1, m), order='F'))

    self.__prep_f_and_S_arrays(orders, freq_all_freq[f_mask], f_max_ind, m_var, m_stationarity)

    for i in tqdm(np.arange(0, n_windows - 1 + window_shift, window_shift), leave=False):

        chunk = scaling_factor * self.data[int(i * (window_points * m)): int((i + 1) * (window_points * m))]
        if not self.first_frame_plotted and show_first_frame:
            plot_first_frame(chunk, self.delta_t, window_points, self.t_unit)
            self.first_frame_plotted = True

        chunk_gpu = to_gpu(chunk.reshape((window_points, 1, m), order='F'))

        if corr_default == 'white_noise':  # use white noise to check for false correlations
            chunk_corr = np.random.randn(window_points, 1, m)
            chunk_corr_gpu = to_gpu(chunk_corr)
        elif self.corr_data is not None:
            chunk_corr = scaling_factor * corr_data[int(i * (window_points * m) + corr_shift): int(
                (i + 1) * (window_points * m) + corr_shift)]
            chunk_corr_gpu = to_gpu(chunk_corr.reshape((window_points, 1, m), order='F'))
        else:
            chunk_corr_gpu = None

        if n_chunks == 0:
            if verbose:
                print('chunk shape: ', chunk_gpu.shape[0])

        # ---------count windows-----------
        n_chunks += 1

        # -------- perform fourier transform ----------
        if rect_win:
            ones = to_gpu(
                np.array(m * [np.ones_like(single_window)]).flatten().reshape((window_points, 1, m), order='F'))
            a_w_all_gpu = fft_r2c(ones * chunk_gpu, dim0=0, scale=1)
        else:
            a_w_all_gpu = fft_r2c(window * chunk_gpu, dim0=0, scale=1)

        # --------- modify data ---------
        if filter_func:
            pre_filter = filter_func(self.freq[2])
            filter_mat = to_gpu(
                np.array(m * [1 / pre_filter]).flatten().reshape((a_w_all_gpu.shape[0], 1, m), order='F'))
            a_w_all_gpu = filter_mat * a_w_all_gpu

        if random_phase:
            a_w_all_gpu = add_random_phase(a_w_all_gpu, window_points, self.delta_t, m)

        # --------- calculate spectra ----------
        self.__fourier_coeffs_to_spectra(orders, a_w_all_gpu, f_max_ind, m, m_var, m_stationarity, single_window,
                                         window, chunk_corr_gpu=chunk_corr_gpu, coherent=coherent,
                                         random_phase=random_phase, window_points=window_points)

        if n_chunks == break_after:
            break

    self.__store_final_spectra(orders, n_chunks, n_windows, m_var)

    return self.freq, self.S, self.S_err

def calc_spec_poisson_one_spectrum(self, order_in, spectrum_size, f_max, f_lists=None, backend='opencl', m=10, m_var=10,
                                       m_stationarity=None, full_import=False, scale_t=1,
                                       sigma_t=0.14, rect_win=False):
    af.set_backend(backend)

    if order_in == 'all':
        orders = [1, 2, 3, 4]
    else:
        orders = order_in

    self.__reset_variables(orders, m, m_var, m_stationarity, f_lists)

    # -------data setup---------
    if self.data is None:
        self.data = self.import_data(self.path, self.group_key, self.dataset, full_import=full_import)
    if self.delta_t is None:
        self.delta_t = 1

    n_chunks = 0
    f_min = f_max / (spectrum_size - 1)
    self.T_window = 1 / f_min

    if f_lists is not None:
        f_list = np.hstack(f_lists)
    else:
        f_list = None

    self.delta_t *= scale_t
    if f_list is None:
        f_list = np.arange(0, f_max + f_min, f_min)

    start_index = 0

    enough_data = True
    f_max_ind = len(f_list)
    w_list = 2 * np.pi * f_list
    w_list_gpu = to_gpu(w_list)
    n_windows = int(self.data[-1] * scale_t // (self.T_window * m))
    print('number of points:', f_list.shape[0])
    print('delta f:', f_list[1] - f_list[0])

    self.__prep_f_and_S_arrays(orders, f_list, f_max_ind, m_var, m_stationarity)

    single_window, N_window_full = calc_single_window(self.T_window / scale_t,
                                                      1 / self.delta_t,
                                                      sigma_t=sigma_t)
    for frame_number in tqdm(range(n_windows)):

        windows, start_index, enough_data = self.__find_datapoints_in_windows(self.data, m, start_index,
                                                                              self.T_window / scale_t, frame_number,
                                                                              enough_data)
        if not enough_data:
            break

        n_chunks += 1

        a_w_all = 1j * np.ones((w_list.shape[0], m))
        a_w_all_gpu = to_gpu(a_w_all.reshape((len(f_list), 1, m), order='F'))

        for i, t_clicks in enumerate(windows):

            if t_clicks is not None:

                t_clicks_minus_start = t_clicks - i * self.T_window / scale_t - m * self.T_window / scale_t * frame_number

                if rect_win:
                    t_clicks_windowed = np.ones_like(t_clicks_minus_start)
                else:
                    t_clicks_windowed, single_window, N_window_full = apply_window(self.T_window / scale_t,
                                                                                   t_clicks_minus_start,
                                                                                   1 / self.delta_t,
                                                                                   sigma_t=sigma_t)

                # ------ GPU --------
                t_clicks_minus_start_gpu = to_gpu(t_clicks_minus_start * scale_t)

                # ------- uniformly weighted clicks -------
                # t_clicks_windowed_gpu = to_gpu(t_clicks_windowed).as_type(af.Dtype.c64)

                # ------- exponentially weighted clicks -------
                exp_random_numbers = np.random.exponential(1, t_clicks_windowed.shape[0])
                t_clicks_windowed_gpu = to_gpu(t_clicks_windowed * exp_random_numbers).as_type(af.Dtype.c64)

                temp1 = af.exp(1j * af.matmulNT(w_list_gpu, t_clicks_minus_start_gpu))
                # temp2 = af.tile(t_clicks_windowed_gpu.T, w_list_gpu.shape[0])
                # a_w_all_gpu[:, 0, i] = af.sum(temp1 * temp2, dim=1)

                a_w_all_gpu[:, 0, i] = af.matmul(temp1, t_clicks_windowed_gpu)

            else:
                a_w_all_gpu[:, 0, i] = to_gpu(1j * np.zeros_like(w_list))

        self.delta_t = self.T_window / N_window_full  # 70 as defined in function apply_window(...)

        self.__fourier_coeffs_to_spectra(orders, a_w_all_gpu, f_max_ind, m, m_var, m_stationarity, single_window)

    assert n_windows == n_chunks, 'n_windows not equal to n_chunks'

    self.__store_final_spectra(orders, n_chunks, n_windows, m_var)

    return self.freq, self.S, self.S_err