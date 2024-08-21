# This file is part of signalsnap: Signal Analysis In Python Made Easy
# Copyright (c) 2020 and later, Markus Sifft and Daniel Hägele.
#
# This software is provided under the terms of the 3-Clause BSD License.
# For details, see the LICENSE file in the root of this repository or
# https://opensource.org/licenses/BSD-3-Clause

import numpy as np


class PlotConfig:
    """
    Class for configuring the plotting of spectral data for various orders.

    Parameters
    ----------
    plot_orders : list of {2, 3, 4}, optional
        Spectral orders to be plotted. Multiple orders can be chosen. Default is (2, 3, 4).
    plot_f_max : float, optional
        Sets the upper limit of the frequency axis. If set higher than the Nyquist frequency,
        the Nyquist frequency will be chosen as limit. Default is None.
    f_min : float, optional
        Sets the lower limit of the frequency axis. Default is None.
    sigma : float, optional
        Sets the number of standard deviations as error to be shown in the two-dimensional
        plots of orders 3 and 4. Default is 1.
    green_alpha : float, optional
        Sets the alpha value for the green error tiling (value between 0 and 1). Default is 0.3.
    arcsinh_plot : bool, optional
        If set, the spectral values are scaled with an arcsinh function. Default is False.
    arcsinh_const : float, optional
        Constant to set amount of arcsinh scaling. The lower, the stronger. Default is 0.02.
    contours : bool, optional
        If set, contours are shown in the 2D plots. Default is False.
    s3_filter : float, optional
        Applies a Gaussian filter of width s3_filter to the third-order data. Default is 0.
    s4_filter : float, optional
        Applies a Gaussian filter of width s4_filter to the fourth-order data. Default is 0.
    s2_data, s2_err, s3_data, s3_err, s4_data, s4_err : array, optional
        Spectral data and errors for the respective orders can be provided and are then used
        instead of the calculated values stored in the object. Default is None.
    s2_f, s3_f, s4_f : array, optional
        Frequency values for the respective orders can be provided and are then used instead of
        the values stored in the object. Default is None.
    imag_plot : bool, optional
        If set, the imaginary part of the spectral values is plotted. Default is False.
    plot_error : bool, optional
        If set, 1 to 5 sigma error bands are shown in the power spectrum. Default is True.
    broken_lims : list of lists, optional
        The lower and upper limit of each section of a broken frequency axis can be provided,
        given that the frequency arrays (s2_f, s3_f, s4_f) contain disconnected sections.
        Default is None.

    """

    def __init__(self, plot_orders=(2, 3, 4), plot_f_max=None, f_min=None, sigma=1, green_alpha=0.3,
                 arcsinh_plot=False, arcsinh_const=0.02, contours=False, s3_filter=0, s4_filter=0,
                 s2_data=None, s2_err=None, s3_data=None, s3_err=None, s4_data=None, s4_err=None,
                 s2_f=None, s3_f=None, s4_f=None, imag_plot=False, plot_error=True, broken_lims=None,
                 tick_fontsize=10, label_fontsize=14):

        if not (isinstance(plot_orders, (tuple, list)) and all(
                isinstance(order, int) and 1 <= order <= 4 for order in plot_orders)):
            raise ValueError("plot_orders must be a tuple or list of integers between 1 and 4.")

        self.plot_orders = plot_orders
        self.tick_fontsize = tick_fontsize
        self.label_fontsize = label_fontsize
        self.plot_f_max = self._validate_positive_float(plot_f_max, "plot_f_max")
        self.f_min = self._validate_positive_float(f_min, "f_min")
        self.sigma = self._validate_positive_float(sigma, "sigma")
        self.green_alpha = self._validate_range(green_alpha, 0, 1, "green_alpha")
        self.arcsinh_plot = self._validate_bool(arcsinh_plot, "arcsinh_plot")
        self.arcsinh_const = self._validate_positive_float(arcsinh_const, "arcsinh_const")
        self.contours = self._validate_bool(contours, "contours")
        self.s3_filter = self._validate_non_negative_integer(s3_filter, "s3_filter")
        self.s4_filter = self._validate_non_negative_integer(s4_filter, "s4_filter")
        self.imag_plot = self._validate_bool(imag_plot, "imag_plot")
        self.plot_error = self._validate_bool(plot_error, "plot_error")

        # Validate numpy arrays
        self.s2_data = self._validate_numpy_array(s2_data, "s2_data")
        self.s2_err = self._validate_numpy_array(s2_err, "s2_err")
        self.s3_data = self._validate_numpy_array(s3_data, "s3_data")
        self.s3_err = self._validate_numpy_array(s3_err, "s3_err")
        self.s4_data = self._validate_numpy_array(s4_data, "s4_data")
        self.s4_err = self._validate_numpy_array(s4_err, "s4_err")
        self.s2_f = self._validate_numpy_array(s2_f, "s2_f")
        self.s3_f = self._validate_numpy_array(s3_f, "s3_f")
        self.s4_f = self._validate_numpy_array(s4_f, "s4_f")

        # Validate broken_limits
        if broken_lims is not None and not all(isinstance(item, list) for item in broken_lims):
            raise ValueError("broken_lims must be a list of lists.")
        self.broken_lims = broken_lims

    @staticmethod
    def _validate_positive_float(value, name):
        if value is not None and (not isinstance(value, (float, int)) or value <= 0):
            raise ValueError(f"{name} must be a positive float or integer.")
        return value

    @staticmethod
    def _validate_range(value, min_value, max_value, name):
        if value is not None and (not isinstance(value, (float, int)) or not min_value <= value <= max_value):
            raise ValueError(f"{name} must be a float or integer between {min_value} and {max_value}.")
        return value

    @staticmethod
    def _validate_bool(value, name):
        if not isinstance(value, bool):
            raise ValueError(f"{name} must be a boolean.")
        return value

    @staticmethod
    def _validate_non_negative_integer(value, name):
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer.")
        return value

    @staticmethod
    def _validate_numpy_array(value, name):
        if value is not None and not isinstance(value, np.ndarray):
            raise ValueError(f"{name} must be a numpy array.")
        return value
