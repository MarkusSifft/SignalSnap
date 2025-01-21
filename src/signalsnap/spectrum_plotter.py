# This file is part of signalsnap: Signal Analysis In Python Made Easy
# Copyright (c) 2020 and later, Markus Sifft and Daniel Hägele.
#
# This software is provided under the terms of the 3-Clause BSD License.
# For details, see the LICENSE file in the root of this repository or
# https://opensource.org/licenses/BSD-3-Clause

from .plot_config import PlotConfig
from .spectrum_calculator import SpectrumCalculator
from .spectrum_config import SpectrumConfig
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as colors
import plotly.graph_objects as go


class SpectrumPlotter:
    def __init__(self, spectrum_calculator: SpectrumCalculator, plot_config: PlotConfig):
        self.spectrum_calculator = spectrum_calculator
        self.plot_config = plot_config

    def __import_spec_data_for_plotting(self, s_data, s_err, order):

        """
        Helper function for importing spectral data during plotting. Depending on the parameters,
        it may retrieve either the real or imaginary part of the spectral data and its corresponding error.

        Parameters
        ----------
        s_data : array_like or None
            Spectral data array. If provided, it is used in the function; otherwise, it is retrieved from `self.S[order]`.
        s_err : array_like or None
            Spectral error array. If provided, it is used in the function; otherwise, it is retrieved from `self.S_err[order]`.
        order : {2, 3, 4}
            Order of the spectral data to be loaded, indicating which data in `self.S` and `self.S_err` to use.

        Returns
        -------
        tuple of array_like
            A tuple containing two arrays for spectral data and error, processed based on the input parameters.

        Notes
        -----
        This function is intended for internal use during plotting, manipulating the data as needed
        based on the provided parameters.
        """

        if self.plot_config.imag_plot:
            s_data = np.imag(self.spectrum_calculator.S[order]).copy() if s_data is None else np.imag(s_data).copy()
            if s_err is not None:
                s_err = np.imag(s_err).copy()
            elif self.spectrum_calculator.S_err[order] is not None:
                s_err = np.imag(self.spectrum_calculator.S_err[order]).copy()

        else:
            s_data = np.real(self.spectrum_calculator.S[order]).copy() if s_data is None else np.real(s_data).copy()
            if s_err is not None:
                s_err = np.real(s_err).copy()
            elif self.spectrum_calculator.S_err[order] is not None:
                s_err = np.real(self.spectrum_calculator.S_err[order]).copy()

        return s_data, s_err

    def arcsinh_scaling(self, s_data, order, s_err=None, s_err_p=None, s_err_m=None):
        """
        Helper function to improve visibility in plotting (similar to a log scale but also works for negative values)

        Parameters
        ----------
        s_data : array
            spectral values of any order
        order : int
            important since the error arrays are called differently in the second-order case
        s_err : array
            spectral errors of order 3 or 4
        s_err_p : array
            spectral values + error of order 2
        s_err_m : array
            spectral values - error of order 2

        Returns
        -------

        """
        x_max = np.max(np.abs(s_data))
        alpha = 1 / (x_max * self.plot_config.arcsinh_const)
        s_data = np.arcsinh(alpha * s_data) / alpha

        if order == 2:
            if s_err_p is not None:
                for i in range(0, 5):
                    s_err_p[i] = np.arcsinh(alpha * s_err_p[i]) / alpha
                    s_err_m[i] = np.arcsinh(alpha * s_err_m[i]) / alpha
            return s_data, s_err_p, s_err_m
        else:
            if s_err is not None:
                s_err = np.arcsinh(alpha * s_err) / alpha
            return s_data, s_err

    def connect_broken_axis(self, s_f):
        """
        Helper function to enable broken axis during plotting

        Parameters
        ----------
        s_f : array
            frequencies at with the spectra had been calculated

        Returns
        -------

        """
        broken_lims_scaled = [(i, j) for i, j in self.plot_config.broken_lims]
        diffs = []
        for i in range(len(broken_lims_scaled) - 1):
            diff = broken_lims_scaled[i + 1][0] - broken_lims_scaled[i][1]
            diffs.append(diff)
            s_f[s_f > broken_lims_scaled[i][1]] -= diff
        return s_f, diffs, broken_lims_scaled

    def calculate_errors(self, s_data, s_err):
        if s_err is None and self.spectrum_calculator.S_err[2] is None:
            return [], []

        return [
            [s_data + (i + 1) * s_err for i in range(5)],
            [s_data - (i + 1) * s_err for i in range(5)]
        ]

    def plot_s2(self, ax, order, s_f_plot, s_data_plot, s_err_plot):

        s_data_plot[order], s_err_plot[order] = self.__import_spec_data_for_plotting(self.plot_config.s2_data,
                                                                                     self.plot_config.s2_err,
                                                                                     order)

        s2_err_p, s2_err_m = self.calculate_errors(s_data_plot[order], s_err_plot[order])

        if self.plot_config.arcsinh_plot:
            s_data_plot[order], s2_err_p, s2_err_m = self.arcsinh_scaling(s_data_plot[order], order,
                                                                          s_err_p=s2_err_p, s_err_m=s2_err_m)

        if self.plot_config.s2_f is None:
            s_f_plot[order] = self.spectrum_calculator.freq[2].copy()
        else:
            s_f_plot[order] = self.plot_config.s2_f

        if self.plot_config.broken_lims is not None:
            s_f_plot[order], diffs, broken_lims_scaled = self.connect_broken_axis(s_f_plot[order])
        else:
            diffs = None
            broken_lims_scaled = None

        if self.plot_config.plot_f_max is None:
            plot_f_max = s_f_plot[order].max()
        else:
            plot_f_max = self.plot_config.plot_f_max
        if self.plot_config.f_min is None:
            f_min = s_f_plot[order].min()
        else:
            f_min = self.plot_config.f_min

        ax[0].set_xlim([f_min, plot_f_max])

        if self.plot_config.plot_error and (
                s_err_plot[order] is not None or self.spectrum_calculator.S_err[2] is not None):
            for i in range(0, 5):
                ax[0].plot(s_f_plot[order], s2_err_p[i], color=[0.1 * i + 0.3, 0.1 * i + 0.3, 0.1 * i + 0.3],
                           linewidth=2, label=r"$%i\sigma$" % (i + 1), alpha=0.5)
                ax[0].plot(s_f_plot[order], s2_err_m[i], color=[0.1 * i + 0.3, 0.1 * i + 0.3, 0.1 * i + 0.3],
                           linewidth=2, label=r"$%i\sigma$" % (i + 1), alpha=0.5)

        ax[0].plot(s_f_plot[order], s_data_plot[order], color=[0, 0.5, 0.9], linewidth=3)

        ax[0].tick_params(axis='both', direction='in')
        ax[0].set_ylabel(r"$S^{(2)}_z$ (" + self.spectrum_calculator.config.f_unit + r"$^{-1}$)", labelpad=13,
                         fontdict={'fontsize': self.plot_config.label_fontsize})
        ax[0].set_xlabel(r"$\omega / 2\pi$ (" + self.spectrum_calculator.config.f_unit + r")", labelpad=13,
                         fontdict={'fontsize': self.plot_config.label_fontsize})
        ax[0].set_title(r"$S^{(2)}_z$ (" + self.spectrum_calculator.config.f_unit + r"$^{-1}$)", fontdict={'fontsize': 16})

        if self.plot_config.broken_lims is not None:
            ylims = ax[0].get_ylim()
            for i, diff in enumerate(diffs):
                ax[0].vlines(broken_lims_scaled[i][-1] - sum(diffs[:i]), ylims[0], ylims[1],
                             linestyles='dashed')

            ax[0].set_ylim(ylims)
            x_labels = ax[0].get_xticks()
            x_labels = np.array(x_labels)
            for i, diff in enumerate(diffs):
                x_labels[x_labels > broken_lims_scaled[i][-1]] += diff
            x_labels = [str(np.round(i * 1000) / 1000) for i in x_labels]
            ax[0].set_xticklabels(x_labels)

    def plot_s3_s4(self, fig, ax, axis, order, s_data_plot, s_err_plot, s_f_plot):
        cmap = colors.LinearSegmentedColormap.from_list('', [[0.1, 0.1, 0.8], [0.97, 0.97, 0.97],
                                                             [1, 0.1, 0.1]])
        color_array = np.array([[0., 0., 0., 0.], [0., 0.5, 0., self.plot_config.green_alpha]])
        cmap_sigma = LinearSegmentedColormap.from_list(name='green_alpha', colors=color_array)

        if order == 3:
            s_data = self.plot_config.s3_data
            s_err = self.plot_config.s3_err
            s_filter = self.plot_config.s3_filter
            s_f = self.plot_config.s3_f
        else:  # order 4
            s_data = self.plot_config.s4_data
            s_err = self.plot_config.s4_err
            s_filter = self.plot_config.s4_filter
            s_f = self.plot_config.s4_f

        s_data_plot[order], s_err_plot[order] = self.__import_spec_data_for_plotting(s_data, s_err, order)

        if s_err_plot[order] is not None or self.spectrum_calculator.S_err[order] is not None:
            s_err_plot[order] *= self.plot_config.sigma

        if self.plot_config.arcsinh_plot:
            s_data_plot[order], s_err_plot[order] = self.arcsinh_scaling(s_data_plot[order],
                                                                         order,
                                                                         s_err=s_err_plot[order])

        if s_f is None:
            s_f_plot[order] = self.spectrum_calculator.freq[order].copy()
        else:
            s_f_plot[order] = s_f

        if self.plot_config.broken_lims is not None:
            s_f_plot[order], diffs, broken_lims_scaled = self.connect_broken_axis(s_f_plot[order])
        else:
            diffs = None
            broken_lims_scaled = None

        abs_max = max(abs(s_data_plot[order].min()), abs(s_data_plot[order].max()))
        norm = colors.TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
        # norm = MidpointNormalize(midpoint=0, vmin=vmin, vmax=vmax)

        if s_f_plot[order].min() < 0: # True in case of full_bispectrum
            x, y = np.meshgrid(s_f_plot[order], s_f_plot[order][s_f_plot[order].shape[0]//2:])
        else:
            x, y = np.meshgrid(s_f_plot[order], s_f_plot[order])

        z = s_data_plot[order].copy()

        err_matrix = np.zeros_like(z)
        if s_err_plot[order] is not None or self.spectrum_calculator.S_err[order] is not None:
            err_matrix[np.abs(s_data_plot[order]) < s_err_plot[order]] = 1

        c = ax[axis].pcolormesh(x, y, gaussian_filter(z, s_filter), cmap=cmap, norm=norm, zorder=1,
                                shading='auto')
        if s_err_plot[order] is not None or self.spectrum_calculator.S_err[order] is not None:
            ax[axis].pcolormesh(x, y, err_matrix, cmap=cmap_sigma, vmin=0, vmax=1, shading='auto')

        if self.plot_config.contours:
            ax[axis].contour(x, y, gaussian_filter(z, s_filter), colors='k', linewidths=0.7)

        if self.plot_config.plot_f_max is None:
            plot_f_max = s_f_plot[order].max()
        else:
            plot_f_max = self.plot_config.plot_f_max

        if self.plot_config.f_min is None:
            f_min = s_f_plot[order].min()
        else:
            f_min = self.plot_config.f_min

        ax[axis].axis([f_min, plot_f_max, f_min, plot_f_max])

        ax[axis].set_xlabel(r"$\omega_1 / 2 \pi$ (" + self.spectrum_calculator.config.f_unit + r")",
                            fontdict={'fontsize': self.plot_config.label_fontsize})
        ax[axis].set_ylabel(r"$\omega_2 / 2 \pi$ (" + self.spectrum_calculator.config.f_unit + r")",
                            fontdict={'fontsize': self.plot_config.label_fontsize})
        ax[axis].tick_params(axis='both', direction='in')

        if self.plot_config.green_alpha == 0:
            ax[axis].set_title(
                r'$S^{(' + f'{order}' + r')}_z $ (' + self.spectrum_calculator.config.f_unit + r'$^{-' + f'{order - 1}' + r'}$)',
                fontdict={'fontsize': 16})
        else:
            ax[axis].set_title(
                r'$S^{(' + f'{order}' + r')}_z $ (' + self.spectrum_calculator.config.f_unit + r'$^{-' + f'{order - 1}' + r'}$) (%i$\sigma$ confidence)' % (
                    self.plot_config.sigma),
                fontdict={'fontsize': 16})
        fig.colorbar(c, ax=(ax[axis]))

        if self.plot_config.broken_lims is not None:
            ylims = ax[axis].get_ylim()
            for i, diff in enumerate(diffs):
                ax[axis].vlines(broken_lims_scaled[i][-1] - sum(diffs[:i]), ylims[0], ylims[1],
                                linestyles='dashed')
                ax[axis].hlines(broken_lims_scaled[i][-1] - sum(diffs[:i]), ylims[0], ylims[1],
                                linestyles='dashed')

            ax[axis].set_ylim(ylims)
            ax[axis].set_xlim(ylims)

            x_labels = ax[axis].get_xticks()
            x_labels = np.array(x_labels)
            for i, diff in enumerate(diffs):
                x_labels[x_labels > broken_lims_scaled[i][-1]] += diff
            x_labels = [str(np.round(i * 1000) / 1000) for i in x_labels]
            ax[axis].set_xticklabels(x_labels)
            ax[axis].set_yticklabels(x_labels)

    def setup_plot(self):
        fig_x = 8 * len(self.plot_config.plot_orders)
        width_ratios = [1.2 if order > 2 else 1.0 for order in self.plot_config.plot_orders]
        fig, ax = plt.subplots(nrows=1, ncols=len(self.plot_config.plot_orders), figsize=(fig_x, 7),
                               gridspec_kw={"width_ratios": width_ratios})
        if len(self.plot_config.plot_orders) == 1:
            ax = [ax]
        plt.rc('text', usetex=False)
        plt.rc('font', size=self.plot_config.tick_fontsize)
        plt.rcParams["axes.axisbelow"] = False

        return fig, ax

    def plot(self):

        fig, ax = self.setup_plot()

        if self.spectrum_calculator.f_lists[2] is not None:
            broken_lims = []
            for part in self.spectrum_calculator.f_lists[2]:
                broken_lims.append((part[0], part[-1]))

            self.plot_config.broken_lims = broken_lims

        s_data_plot = {2: None, 3: None, 4: None}
        s_err_plot = {2: None, 3: None, 4: None}
        s_f_plot = {2: None, 3: None, 4: None}

        for axis, order in enumerate(self.plot_config.plot_orders):

            has_data = self.spectrum_calculator.S[order] is not None and \
                       not self.spectrum_calculator.S[order].shape[0] == 0

            # Plot S2
            if order == 2 and (has_data or self.plot_config.s2_data is not None):
                self.plot_s2(ax, order, s_f_plot, s_data_plot, s_err_plot)

            # Plot S3 and S4
            if order > 2 and has_data:
                self.plot_s3_s4(fig, ax, axis, order, s_data_plot, s_err_plot, s_f_plot)

        plt.show()
        return fig

    def plot_interactive(self, order):
        """

        Returns
        -------
        Returns a plotly figure.
        """

        s_data_plot = {2: None, 3: None, 4: None}
        s_err_plot = {2: None, 3: None, 4: None}
        s_f_plot = {2: None, 3: None, 4: None}

        if order == 2 and self.spectrum_calculator.S[order] is not None and not self.spectrum_calculator.S[order].shape[
                                                                                    0] == 0:

            s_data_plot[order], s_err_plot[order] = self.__import_spec_data_for_plotting(self.plot_config.s2_data,
                                                                                         self.plot_config.s2_err, order)

            s2_err_p = []
            s2_err_m = []

            if s_err_plot[order] is not None or self.spectrum_calculator.S_err[2] is not None:
                s2_err_p.append(s_data_plot[order] + self.plot_config.sigma * s_err_plot[order])
                s2_err_m.append(s_data_plot[order] - self.plot_config.sigma * s_err_plot[order])

            if self.plot_config.arcsinh_plot:
                s_data_plot[order], s2_err_p, s2_err_m = self.arcsinh_scaling(s_data_plot[order], order,
                                                                              s_err_p=s2_err_p, s_err_m=s2_err_m)

            if self.plot_config.s2_f is None:
                s_f_plot[order] = self.spectrum_calculator.freq[2].copy()
            else:
                s_f_plot[order] = self.plot_config.s2_f

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=s_f_plot[order], y=s_data_plot[order],
                                     mode='lines',
                                     name='measurement'))
            fig.add_trace(go.Scatter(x=s_f_plot[order], y=s2_err_p[0],
                                     mode='lines',
                                     name=f'error band ({self.plot_config.sigma} sigma)'))
            fig.add_trace(go.Scatter(x=s_f_plot[order], y=s2_err_m[0],
                                     mode='lines'))
            title = r"$S^{(2)}_z$ (" + self.spectrum_calculator.config.f_unit + r"$^{-1}$)"
            print(title)
            fig.update_layout(title=title,
                              xaxis_title=r"$\omega / 2\pi$ (" + self.spectrum_calculator.config.f_unit + r")",
                              yaxis_title=r"$S^{(2)}_z$ (" + self.spectrum_calculator.config.f_unit + r"$^{-1}$)")
            fig.show()

        if order > 2:
            if self.spectrum_calculator.S[order] is not None and not self.spectrum_calculator.S[order].shape[0] == 0:
                print(1)
                if order == 3:
                    s_data = self.plot_config.s3_data
                    s_err = self.plot_config.s3_err
                    s_f = self.plot_config.s3_f
                else:  # order 4
                    s_data = self.plot_config.s4_data
                    s_err = self.plot_config.s4_err
                    s_f = self.plot_config.s4_f

                s_data_plot[order], s_err_plot[order] = self.__import_spec_data_for_plotting(s_data, s_err, order)

                if s_err_plot[order] is not None or self.spectrum_calculator.S_err[order] is not None:
                    s_err_plot[order] *= self.plot_config.sigma

                if self.plot_config.arcsinh_plot:
                    s_data_plot[order], s_err_plot[order] = self.arcsinh_scaling(s_data_plot[order],
                                                                                 order,
                                                                                 s_err=s_err_plot[order])

                if s_f is None:
                    s_f_plot[order] = self.spectrum_calculator.freq[order].copy()
                else:
                    s_f_plot[order] = s_f

                fig = go.Figure(data=[
                    go.Surface(x=s_f_plot[order],
                               y=s_f_plot[order],
                               z=s_data_plot[order]),
                    go.Surface(x=s_f_plot[order],
                               y=s_f_plot[order],
                               z=s_data_plot[order] + self.plot_config.sigma * s_err_plot[order], showscale=False,
                               opacity=0.9),
                    go.Surface(x=s_f_plot[order],
                               y=s_f_plot[order],
                               z=s_data_plot[order] - self.plot_config.sigma * s_err_plot[order], showscale=False,
                               opacity=0.9)

                ])

                fig.show()
                print(2)

    def stationarity_plot(self, s2_filter=0, normalize=False):

        """
        Plots the \( S_{\text{stationarity}} \) spectra versus time to visualize changes over time.

        Parameters
        ----------
        s2_filter : float, optional
           Sigma value for a Gaussian filter in the direction of time; useful for noisy data. Default is 0.
        normalize : {'area', 'zero'}, optional
           For better visualization, all spectra can be normalized to set the area under \( S_2 \) to 1 or the value at \( S_2(0) \) to 1. Default is False.

        Returns
        -------
        fig, ax : matplotlib figure and axes
           Figure and axes objects representing the plot.

        Notes
        -----
        This function plots the second-order stationarity spectrum over time, with optional normalization, filtering, or transformation.
        The plot helps in understanding how the spectrum evolves and changes during the course of the time series.
        """

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(24, 7))
        plt.rc('text', usetex=False)
        plt.rc('font', size=10)
        plt.rcParams["axes.axisbelow"] = False

        if self.spectrum_calculator.f_lists[2] is not None:
            broken_lims = []
            for part in self.spectrum_calculator.f_lists[2]:
                broken_lims.append((part[0], part[-1]))
        else:
            broken_lims = None

        s2_array = np.real(self.spectrum_calculator.S_stationarity[2]).T.copy()
        s2_array = gaussian_filter(s2_array, sigma=[0, s2_filter])

        if normalize == 'area':
            s2_array /= np.sum(s2_array, axis=0)
        elif normalize == 'zero':
            s2_array /= np.max(s2_array, axis=0)

        if self.plot_config.arcsinh_plot:
            s2_array, _, _ = self.arcsinh_scaling(s2_array, order=2)

        vmin = np.min(s2_array)
        vmax = np.max(s2_array)

        t_for_one_spec = self.spectrum_calculator.T_window * self.spectrum_calculator.m[2] * \
                         self.spectrum_calculator.m_stationarity[2]

        if self.spectrum_calculator.config.interlaced_calculation:
            t_for_one_spec /= 2

        time_axis = np.arange(0, s2_array.shape[1] * t_for_one_spec, t_for_one_spec)
        print(
            f'One spectrum calculated from a {t_for_one_spec * (s2_filter + 1)} ' + self.spectrum_calculator.t_unit + ' measurement')

        s2_f = self.spectrum_calculator.freq[2].copy()
        if broken_lims is not None:
            s2_f, diffs, broken_lims_scaled = self.connect_broken_axis(s2_f)
        else:
            diffs = None
            broken_lims_scaled = None

        x, y = np.meshgrid(time_axis, s2_f)

        c = ax.pcolormesh(x, y, s2_array, cmap='rainbow', vmin=vmin, vmax=vmax, shading='auto')  # norm=norm)
        if self.plot_config.contours:
            ax.contour(x, y, s2_array, 7, colors='k', linewidths=0.7)

        if self.plot_config.plot_f_max:
            ax.axis([0, np.max(time_axis), 0, self.plot_config.plot_f_max])
        ax.set_xlabel(r"$t$ (" + self.spectrum_calculator.t_unit + r")", fontdict={'fontsize': 14})
        ax.set_ylabel(r"$\omega / 2 \pi$ (" + self.spectrum_calculator.config.f_unit + r")", fontdict={'fontsize': 14})
        ax.tick_params(axis='both', direction='in')
        ax.set_title(r'$S^{(2)}_z $ (' + self.spectrum_calculator.config.f_unit + r'$^{-1}$) vs $' + self.spectrum_calculator.t_unit + r'$',
                     fontdict={'fontsize': 16})
        _ = fig.colorbar(c, ax=ax)

        if broken_lims is not None:
            xlims = ax.get_xlim()
            for i, diff in enumerate(diffs):
                ax.hlines(broken_lims_scaled[i][-1] - sum(diffs[:i]), xlims[0], xlims[1], linestyles='dashed')

            ax.set_xlim(xlims)

            y_labels = ax.get_yticks()
            y_labels = np.array(y_labels)
            for i, diff in enumerate(diffs):
                y_labels[y_labels > broken_lims_scaled[i][-1]] += diff
            y_labels = [str(np.round(i * 1000) / 1000) for i in y_labels]
            ax.set_yticklabels(y_labels)
