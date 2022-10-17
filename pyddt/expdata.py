#!/usr/bin/python3

"""This class is used for analyzing Renninger scans."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.subplots as sp

from .funcs import delete_multiple
from .scatter import *
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from tqdm import tqdm

from .crystal import Crystal
from .expdata_funcs import dataframe_BClines, fwhm, peak_fit, bootstrap_resampling

__author__ = "Rafaela Felix"
__credits__ = {"Rafaela Felix", "Sergio Morelhão"}
__version__ = "1.0"
__maintainer__ = "Rafaela Felix"
__email__ = "rafaelafelixp@usp.br"
__status__ = "Production"


class ExpData:

    def __init__(self, E: float, G: list, fname: str, colx: int = 0, coly: int = 1, name: str = ''):

        """Given the experimental data, returns a new ``exp`` object.

        Args:
            E (float): X-ray energy (eV).
            G (list): Primary reflection (eg `[1, 1, 1]`).
            fname (str): Filename containing the data.
            colx (int): Column number of angle values. **Default**: 0 (1st column).
            coly (int): Column number of intensity values. **Default**: 1 (2nd column).
            name (str): Label for the current work. **Default**: None.
        """

        self.asy = None
        self.slope_data = None
        self.index = None
        self.BC = None
        self.region = None
        self.peaks = None
        self.phi = np.loadtxt(fname, usecols=colx)
        self.counts = np.loadtxt(fname, usecols=coly)
        self.energy = E
        self.primary = np.array(G)
        self.name = name

    def plot(self) -> matplotlib.figure.Figure:

        """Plots the phi-scan.

        Returns:
            matplotlib.figure.Figure: Interactive plot.
        """

        fig = plt.figure()
        plt.plot(self.phi, self.counts, c='k')
        if self.peaks is not None:
            plt.plot(self.phi[self.peaks], self.counts[self.peaks], "*b")
        plt.xlabel(r'$\phi\; (deg)$')
        plt.ylabel(r'$counts$')
        plt.xlim(self.phi.min(), self.phi.max())
        return fig

    def BC_plot(self, fname: str, M: list, Fmin: float, dw: float = 0.1, npoints: int = 20):

        """Plots the phi-scan and BC lines for visual indexing.

        Args:
            fname (str): Structure filename (`.in`).
            M (list): Reference direction (eg `[1, 0, 0]`).
            Fmin (float): Cutoff for W (minimum absolute value).
            dw (float): Maximum distance between BC lines and primary reflection.
            npoints (int): Number of points of the 2D-cone representation.

        Notes:
            * By default, the ``plotly.graph_objects`` will be displayed in an installed browser or notebook.
            * BC lines ranging from -180 to 180 deg.
        """

        M = np.array(M)
        struc = Crystal(fname)
        struc.klines(self.energy, self.primary, M, Fmin, dw, npoints)

        fname = "G_" + "".join((self.primary.astype(int)).astype(str))
        fname += "_M_" + "".join((M.astype(int)).astype(str))
        fname += "_E_" + str(self.energy) + ".dat"
        thG, FHFGH, df_in = dataframe_BClines('IN_' + fname)
        thG, FHFGH, df_out = dataframe_BClines('OUT_' + fname)

        subfigure = sp.make_subplots(rows=2, cols=1,
                                     shared_xaxes=True,
                                     vertical_spacing=0.02)

        # Experimental data
        df_exp = pd.DataFrame(np.column_stack((self.phi, self.counts)),
                              columns=['phi', 'I'])

        fig = px.scatter(df_exp, x='phi', y='I',
                         labels={'phi': '𝜙 (deg)', 'I': 'Counts'})
        fig.data[0].update(mode='markers+lines')
        fig.update_traces(marker=dict(color='black', size=5),
                          line=dict(color='black', width=2))

        fig_traces = []
        for trace in range(len(fig['data'])):
            fig_traces.append(fig['data'][trace])
        for traces in fig_traces:
            subfigure.append_trace(traces, row=1, col=1)

        subfigure.update_yaxes(title_text='Counts',
                               range=[0, 1.1 * self.counts.max()],
                               row=1, col=1)
        subfigure.update_xaxes(range=[self.phi.min(), self.phi.max()],
                               row=1, col=1)
        subfigure.update_layout(legend_title_text='', template='plotly_white')

        # BC lines
        fig = px.scatter()
        for i in range(1, len(FHFGH)):
            hover_in = '<i>𝜙</i>: %{x:.3f} <br><b>F</b>:' + str(FHFGH[i - 1])
            hover_in += '<br><b>hkl</b>:' + df_in.columns[i]
            
            xin = df_in[df_in.columns[i]].to_numpy()
            yin = df_in['ω'].to_numpy()
            
            xin[np.where(np.abs(np.ediff1d(xin)) > 10)] = None
            
            fig.add_scatter(x=xin, y=yin, name='',
                            hovertemplate=hover_in, mode='lines',
                            line=dict(color='Red', 
                            width=10*FHFGH[i - 1] / np.max(FHFGH)))
                                      
            fig.update_traces(connectgaps=False)

            hover_out = '<i>𝜙</i>: %{x:.3f} <br><b>F</b>:' + str(FHFGH[i - 1])
            hover_out += '<br><b>hkl</b>:' + df_out.columns[i]
            
            xout = df_out[df_out.columns[i]].to_numpy()
            yout = df_out['ω'].to_numpy()
            
            xout[np.where(np.abs(np.ediff1d(xout)) > 10)] = None

            fig.add_scatter(x=xout, y=yout, name='',
                            hovertemplate=hover_out, mode='lines',
                            line=dict(color='Blue', 
                            width=10*FHFGH[i - 1] / np.max(FHFGH)))
                                      
            fig.update_traces(connectgaps=False)

        fig_traces = []
        for trace in range(len(fig["data"])):
            fig_traces.append(fig["data"][trace])
        for traces in fig_traces:
            subfigure.append_trace(traces, row=2, col=1)

        subfigure.add_shape(type='line', x0=self.phi.min(), y0=thG,
                            x1=self.phi.max(), y1=thG,
                            line=dict(color='black', width=3),
                            row=2, col=1)
        subfigure.update_xaxes(title_text='𝜙 (deg)',
                               range=[self.phi.min(), self.phi.max()],
                               row=2, col=1,
                               showgrid=False)
        subfigure.update_yaxes(title_text='ω (deg)',
                               range=[thG - dw, thG + dw],
                               row=2, col=1,
                               showgrid=False)
        subfigure.update_layout(showlegend=False, template='plotly_white',
                                font=dict(size=16))

        subfigure.show()

    def peak_picker(self):

        """Manual selection of MD peaks.

        Notes:
                * To select a peak, stop the mouse over the maximum and press **SPACE**.
                * To unselect, stop the mouse over the maximum and press **DEL**.
                * Feel free to zoom in or out.
                * After finishing the selection, press **ENTER**. Then, close the figure (mouse or press **q**).

        Attention:
            Press **ENTER** before closing the plot, even if no peak was selected.
        """

        self.plot()
        plt.scatter(self.phi, self.counts, c='k', s=1)
        mk = plt.ginput(n=0, timeout=0, mouse_add=None,
                        mouse_pop=None, mouse_stop=None)
        plt.draw()
        while True:
            if plt.waitforbuttonpress(0):
                plt.close()
                break

        mk = np.array(mk)
        if len(mk) != 0:

            phi_mk = mk[:, 0]

            idx = []
            for j in range(len(phi_mk)):
                diff = np.abs(self.phi - phi_mk[j])
                idx.append(int(diff.argmin()))

            if self.peaks.all() == None:
                self.peaks = idx
            else:
                self.peaks = np.append(self.peaks, idx)

            self.plot()
            plt.show()

    def peak_finder(self, minimum_intensity_p: float = 0.03):

        """Automatic definition of the MD peak list.

        Args:
            minimum_intensity_p (float): Minimum intensity for considering a peak (% of the maximum intensity ranging
                between 0 and 1). **Default**: 0.03.
        """

        yf = gaussian_filter1d(self.counts, 2)
        peaks, _ = find_peaks(yf, prominence=100,
                              height=self.counts.max() * minimum_intensity_p)

        self.peaks = peaks

    def review(self):

        """Plots the experimental data highlighting the selected MD peaks for user review.

        Notes:
                * To select a peak, stop the mouse over the maximum and press **SPACE**.
                * To unselect, stop the mouse over the maximum and press **DEL**.
                * Feel free to zoom in or out.
                * After finishing the selection, press **ENTER**. Then, close the figure (mouse or press **q**).

        Attention:
            Press **ENTER** before closing the plot, even if no peak was selected.
        """

        self.plot()
        mk = plt.ginput(n=0, timeout=0, mouse_add=None,
                        mouse_pop=None, mouse_stop=None)
        plt.show()

        mk = np.array(mk)
        if len(mk) != 0:

            phi_mk = mk[:, 0]

            idx = []
            for j in range(len(phi_mk)):
                diff = np.abs(self.phi[self.peaks] - phi_mk[j])
                idx.append(int(diff.argmin()))

            self.peaks = np.delete(self.peaks, idx)
            if self.region is not None:
                self.region = np.delete(self.region, idx, axis=0)

    def region_of_fit(self, interval: int = 15, points: int = 60, flag: int = 0):

        """Defines the region of fit for each selected MD peak.

        Args:
            interval (int): Number of fwhm defining the region of fit. **Default**: 15.
            points (int): Max number of points defining a peak.
            flag (int): If ``flag != 0``, plots the data highlighting the calculated regions. **Default**: 0.

        Notes:
                * If `flag != 0`, it's possible to delete peaks.
                * To select a peak, stop the mouse over the maximum and press **SPACE**.
                * To unselect, stop the mouse over the maximum and press **DEL**.
                * Feel free to zoom in or out.
                * After finishing the selection, press **ENTER**. Then, close the figure (mouse or press **q**).

        Attention:
            Press **ENTER** before closing the plot, even if no peak was selected.
        """

        region, i = np.zeros((len(self.peaks), 4)), 0

        for j in self.peaks:

            m, n = self._peak_definer(j, points)
            x = self.phi[m:n]
            y = self.counts[m:n]

            d = self.phi[j]
            f = fwhm(x, y)

            N = np.abs(x - (d - interval * f)).argmin()
            M = np.abs(x - (d + interval * f)).argmin()

            region[i, :] = x[N], x[M], f, d
            i += 1

            self.region = region

        if flag != 0:
            self.region_plot()

    def region_plot(self):

        """Plots the calculated regions of fit for user review.

        Notes:
                * To select a peak, stop the mouse over the maximum and press **SPACE**.
                * To unselect, stop the mouse over the maximum and press **DEL**.
                * Feel free to zoom in or out.
                * After finishing the selection, press **ENTER**. Then, close the figure (mouse or press **q**).

        Attention:
            Press **ENTER** before closing the plot, even if no peak was selected.
        """

        self.plot()
        for i in range(0, len(self.region[:, -1])-1, 2):

            N = np.abs(self.phi - self.region[i, 0]).argmin()
            M = np.abs(self.phi - self.region[i, 1]).argmin()

            x = self.phi[N:M]
            y = self.counts[N:M]

            plt.plot(x, y, 'orange')

            N = np.abs(self.phi - self.region[i + 1, 0]).argmin()
            M = np.abs(self.phi - self.region[i + 1, 1]).argmin()

            x = self.phi[N:M]
            y = self.counts[N:M]

            plt.plot(x, y, '--', c='magenta')

        N = np.abs(self.phi - self.region[-1, 0]).argmin()
        M = np.abs(self.phi - self.region[-1, 1]).argmin()

        x = self.phi[N:M]
        y = self.counts[N:M]

        plt.plot(x, y, 'orange')

        mk = plt.ginput(n=0, timeout=0, mouse_add=None,
                        mouse_pop=None, mouse_stop=None)

        mk = np.array(mk)
        if len(mk) != 0:

            phi_mk = mk[:, 0]

            idx = []
            for j in range(len(phi_mk)):
                diff = np.abs(self.phi[self.peaks] - phi_mk[j])
                idx.append(int(diff.argmin()))

            self.peaks = np.delete(self.peaks, idx)
            self.region = np.delete(self.region, idx, axis=0)

        plt.show()

    def indexer(self, fname: str, M: list, Fmin: float, dmin: float = 0.1, flag: int = 0):

        """Finds the secondary reflection related to each selected MD peak.

        Args:
            fname (str): Structure filename (the same used by ``Crystal`` class).
            M (list): Reference direction (eg `[1, 0, 0]`).
            Fmin (float): Cutoff for W (minimum absolute value).
            dmin (float): Maximum difference between MD peak and BC line crossing the primary. **Default**: 0.1 deg
            flag (int): If `flag != 0`, plots the secondary reflection indices. **Default**: 0.

        Notes:
            * Case two or more lattice planes are excited at the same angle with the same geometry:
              all will be displayed in the output file.
            * If the geometries are opposite, the peak is not **indexable** (so it's excluded from the peak list).
            * It's also not **indexable** the case with two excited reflections that distance themselves by less than
              0.05º, whose smallest W value is at least half of the largest value. Except for these cases,
              the peak is indexed by the stronger reflection that is up to ``dmin`` of distance from the MD
              peak position.
        """

        M = np.array(M)

        fthg = "THG_G_" + "".join((self.primary.astype(int)).astype(str))
        fthg += "_M_" + "".join((M.astype(int)).astype(str))
        fthg += "_E_" + str(self.energy) + ".dat"

        try:
            self.BC = np.loadtxt(fthg)
        except (OSError, IOError, ValueError):
            struc = Crystal(fname)
            struc.klines(self.energy, self.primary, M, Fmin)
            self.BC = np.loadtxt(fthg)

        hkl = (self.BC[:, 0:3].astype(int)).astype(str)
        for i in range(len(hkl)):
            hkl[i] = " ".join(hkl[i])
        hkl = hkl[:, 0]

        idx = []
        s = []
        HKL = []
        for i in range(len(self.peaks)):

            ph = np.round(self.phi[self.peaks[i]], 3)
            if ph < 0:
                ph += 360
            if ph > 360:
                ph -= 360

            phi_in, fin, hkl_in = self._index_find(dmin, ph, 4, hkl)
            phi_out, fout, hkl_out = self._index_find(dmin, ph, 3, hkl)

            if fin == 0 and fout == 0:
                HKL.append('')
                s.append(0)
                idx.append(i)
            elif np.abs(phi_in - phi_out) < 0.05:
                f = np.array([fin, fout])
                miller = np.array([hkl_in, hkl_out])
                diff = np.array([1, -1])
                ratio = f.min() / f.max()
                if not ratio >= 0.5:
                    j = f.argmax()
                    HKL.append(miller[j])
                    s.append(diff[j])
                else:
                    HKL.append('')
                    s.append(0)
                    idx.append(i)
            elif fin > fout:
                HKL.append(hkl_in)
                s.append(1)
            elif fout > fin:
                HKL.append(hkl_out)
                s.append(-1)

        self.index = np.column_stack((s, HKL))
        if flag != 0:
            self.indexation_plot(HKL, s, idx)

        self.peaks, self.region, self.index = delete_multiple(self.peaks, self.region, self.index, idx=idx)

    def indexation_plot(self, hkl: list, s: list, idx: list):

        """Plots the phi-scan and corresponding MD indexing  for user review.

        Args:
            hkl (list): Secondary reflections.
            s (list): Diffraction geometry of excitation.
            idx (list): Index of **unindexable** cases in peak list.

        Notes:
                * To select a peak, stop the mouse over the maximum and press **SPACE**.
                * To unselect, stop the mouse over the maximum and press **DEL**.
                * Feel free to zoom in or out.
                * After finishing the selection, press **ENTER**. Then, close the figure (mouse or press **q**).

        Attention:
            Press **ENTER** before closing the plot, even if no peak was selected.
        """

        self.plot()
        for i in range(len(s)):

            N = np.abs(self.phi - self.region[i, 0]).argmin()
            M = np.abs(self.phi - self.region[i, 1]).argmin()

            x = self.phi[N:M]
            y = self.counts[N:M]

            if i in idx:
                plt.plot(x, y, "g")
            else:
                if s[i] == 1:
                    c = "r"
                else:
                    c = "b"
                plt.annotate(hkl[i], (self.phi[self.peaks[i]], self.counts[self.peaks[i]]), color=c)

        plt.show()

    def fitter(self, nsamples: int = 1000):

        """Calculates the slope and slope error for each MD peak.

        Args:
            nsamples (int): Number of samples for bootstrap resampling.

        Notes:
            If ``nsamples = 0``, the slope error isn't calculated.
        """

        slope_err = np.zeros((len(self.peaks), 2))
        for i in tqdm(range(len(self.peaks))):
            N = np.abs(self.phi - self.region[i, 0]).argmin()
            M = np.abs(self.phi - self.region[i, 1]).argmin()

            x = self.phi[N:M]
            y = np.log10(self.counts[N:M])

            try:
                slope_err[i, 0] = peak_fit(x, y)

                if nsamples != 0:
                    sample = pd.DataFrame(np.column_stack((x, y)))
                    slope_err[i, 1] = bootstrap_resampling(sample, nsamples)
                else:
                    slope_err[i, 1] = 0
            except ValueError:
                slope_err[i, 0] = 0

        self.slope_data = slope_err

    def asymmetry_assigner(self, sbar: float = 0.1, tau: float = 0.4, flag: int = 0):

        """Assigns the asymmetry type.

        Args:
            sbar (float): Minimum value of the relative slope. **Default**: 0.1 (10%)
            tau (float): Maximum value for the ratio slope_error/slope. **Default**: 0.4 (40%)
            flag (int): If ``flag != 0``, plots the data with read asymmetry types. **Default**: 0.

        Notes:
                * If `flag != 0`, it's possible to delete peaks.
                * To select a peak, stop the mouse over the maximum and press **SPACE**.
                * To unselect, stop the mouse over the maximum and press **DEL**.
                * Feel free to zoom in or out.
                * After finishing the selection, press **ENTER**. Then, close the figure (mouse or press **q**).

        Attention:
            Press **ENTER** before closing the plot, even if no peak was selected.

        """

        ratio = self.slope_data[:, 0] / np.abs(self.slope_data[:, 0].max())
        idx1 = np.where(np.abs(ratio) < sbar)

        if np.array(idx1).size == 0:
            pass
        else:
            self.index, self.slope_data, self.region, self.peaks = delete_multiple(self.index,
                                                                                   self.slope_data,
                                                                                   self.region, self.peaks,
                                                                                   idx=idx1)

        ratio = self.slope_data[:, 1] / np.abs(self.slope_data[:, 0])
        idx2 = np.where(np.abs(ratio) > tau)
        if np.array(idx2).size == 0:
            pass
        else:
            self.index, self.slope_data, self.region, self.peaks = delete_multiple(self.index,
                                                                                   self.slope_data,
                                                                                   self.region, self.peaks,
                                                                                   idx=idx2)

        asy = np.zeros((len(self.slope_data[:, 0]), 2), dtype=str)
        asy[np.where(self.slope_data[:, 0] < 0), :] = 'H', 'L'
        asy[np.where(self.slope_data[:, 0] > 0), :] = 'L', 'H'
        asy = [''.join(row) for row in asy]

        self.asy = asy
        if flag != 0:
            self.asymmetry_plot()

    def asymmetry_plot(self):

        """Plots the phi-scan and corresponding asymmetry type assigned for each selected MD peak.

        Notes:
                * To invert the assigned asymmetry of an MD peak,, stop the mouse over the maximum and press **SPACE**.
                * To unselect, stop the mouse over the maximum and press **DEL**.
                * Feel free to zoom in or out.
                * After finishing the selection, press **ENTER**. Then, close the figure (mouse or press **q**).

        Attention:
            Press **ENTER** before closing the plot, even if no peak was selected.

        """

        self.plot()
        for i in range(len(self.asy)):
            plt.annotate(self.asy[i], (self.phi[self.peaks[i]] - 0.02, 1.02 * self.counts[self.peaks[i]]),
                         weight='bold', size=12)

        mk = plt.ginput(n=0, timeout=0, mouse_add=None,
                        mouse_pop=None, mouse_stop=None)

        mk = np.array(mk)
        if len(mk) != 0:

            phi_mk = mk[:, 0]

            idx = []
            for j in range(len(phi_mk)):
                diff = np.abs(self.phi[self.peaks] - phi_mk[j])
                idx.append(int(diff.argmin()))

            for ids in idx:

                if self.asy[ids] == 'HL':
                    self.asy[ids] = 'LH'
                else:
                    self.asy[ids] = 'HL'

    def save(self, fout: str = ''):

        """Saves the current work.

        Args:
            fout (str): Filename of the output data. **Default**: *NAME_E_value_G_indexes.ext/.red*

        Notes:
            * The *.ext* file presents the azimuth position, *hkl*, slope, slope error, statistical
              properties, asymmetry type and excitation geometry.
            * The *.red* file only presents *hkl*, asymmetry and diffraction geometry, besides primary indices
              and energy in 1st line.
        """

        sbar = self.slope_data[:, 0] / np.abs(self.slope_data[:, 0].max())
        tau = np.abs(self.slope_data[:, 1] / self.slope_data[:, 0])

        hkl = []
        for i in range(len(self.index[:, 1:])):
            hkl.append(self.index[i, -1].split('/ ')[0])

        M = np.column_stack((np.round(self.phi[self.peaks], 4),
                             self.asy, np.round(self.slope_data, 4),
                             np.round(sbar, 2), np.round(tau, 2),
                             self.index[:, 0], hkl, self.index[:, 1:]))

        H = np.zeros((len(hkl), 3))
        for i in range(len(hkl)):
            H[i, :] = np.rint(np.int64((hkl[i].split(' '))))

        N = np.column_stack((H, self.asy, self.index[:, 0]))

        header = '    phi    asy     slope   slope error   sbar (%)    tau (%)   s       hkl'

        if fout == '':
            fout = self.name.upper() + '_E' + str(self.energy)
            fout += '_G' + "".join((self.primary.astype(int)).astype(str))

        with open(fout + '.ext', 'wb') as f:
            np.savetxt(f, [], header=header)
            for line in np.matrix(M):
                np.savetxt(f, line, fmt='%10s %5s %10s %10s %10s %10s %5s %10s %32s')

        header = '      h          k         l           asy         s '
        with open(fout + '.red', 'wb') as f:
            np.savetxt(f, [], header=header)
            np.savetxt(f, np.column_stack((self.primary[0], self.primary[1], self.primary[2], self.energy)),
                       fmt='%10.1f')
            for line in np.matrix(N):
                np.savetxt(f, line, fmt='%10s %10s %10s %10s %10s')

        return fout

    def _peak_definer(self, peak_number: int, points: int) -> tuple:

        """Finds the indices of first and last points of an MD peak.

        Args:
            peak_number (int): Index of the peak center in the angles array.
            points (int): Max number of points defining a peak.

        Returns:
            tuple: First and last points of the MD peak.
        """

        p = int(points / 2)

        if peak_number - p < 0:
            m = 0
        else:
            m = peak_number - p
        if peak_number + p > len(self.counts):
            n = len(self.counts) - 1
        else:
            n = peak_number + p

        return m, n

    def _index_find(self, dmin: float, ph: float, col: int, hkl: np.ndarray) -> tuple:

        """Find the closer MD case of a defined azimuth angle.

        Args:
            dmin (float): Maximum difference between peak position and BCs crossing the primary reflection.
            ph (float): Angle on the Renninger scan.
            col (int): 3 or 4 (in-out or out-in BC lines).
            hkl (np.ndarray): Miller indices.

        Returns:
            tuple: Azimuth position, W amplitude, secondary reflection indices.
        """

        k = np.where(np.abs(np.round(self.BC[:, col], 3) - ph) < dmin)

        if not np.array(k).size == 0:
            m = np.argsort(np.abs(np.round(self.BC[k, col], 3) - ph))[::-1]
            phi_values = np.round(self.BC[k[0], col], 3)
            w_values = self.BC[k[0], -1]

            equal = np.all(phi_values == phi_values[m[0]][0])
            if equal:  # all BCs crossing in the same phi
                PHI = phi_values[m[0]][0]
                fst = np.argsort(w_values)[::-1]
                W = w_values[m[0]][fst[0]]
                miller = '/ '.join(hkl[k][m][0][fst])

            elif phi_values[m[0]][0] - phi_values[m[0]][1] < 0.01:  # does not cross in the same phi, but close of that
                phi1 = phi_values[m[0]][0]
                phi2 = phi_values[m[0]][1]
                W1 = w_values[m[0]][0]
                W2 = w_values[m[0]][1]

                if np.abs(np.round(phi1, 4) - np.round(phi2, 4)) < 0.005 and W1 == W2:
                    miller = '/ '.join(hkl[k][m][0][:2])
                    PHI = phi1
                    W = W1
                elif np.abs(phi1 - phi2) < 0.05:
                    Wf = np.array([W1, W2])
                    if not Wf.min() / Wf.max() >= 0.5:
                        miller = hkl[k][m][0][Wf.argmax()]
                        PHI = phi_values[m[0]][Wf.argmax()]
                        W = w_values[m[0]][Wf.argmax()]
                    else:
                        miller = ''
                        PHI = 0
                        W = 0
                else:
                    miller = hkl[k][w_values.argmax()]
                    PHI = phi_values[w_values.argmax()]
                    W = w_values[w_values.argmax()]

            else:  # does not cross so close
                miller = hkl[k][w_values.argmax()]
                PHI = phi_values[w_values.argmax()]
                W = w_values[w_values.argmax()]
        else:  # does not cross
            miller = ''
            PHI = 0
            W = 0

        return PHI, W, miller
