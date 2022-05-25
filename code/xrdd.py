#!/usr/bin/python3

"""XRDD is a tool for help to plan X-ray dynamical diffraction experiments,
generating structural models, analyze experimental data and constructing
compatibility diagrams."""

import re
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.subplots as sp
from halo import Halo
from pyfiglet import Figlet
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from termcolor import colored
from tqdm import tqdm

from funcs import *
from scatter import *

__author__ = "Rafaela Felix"
__credits__ = ["Rafaela Felix", "Sergio Morelhão"]
__version__ = "1.0"
__maintainer__ = "Rafaela Felix"
__email__ = "rafaelafelixp@usp.br"
__status__ = "Production"

np.set_printoptions(precision=4, threshold=np.inf, suppress=True, formatter={'float': '{: 0.4f}'.format})


class Crystal:

    def __init__(self, fname: str):

        """Given a structure file, returns a new `crystal` object.

        Args:
            fname (str): Structure file name.

        Notes:
            This file should have six columns: atom or ion label, fractional
            coordinates (x, y and z), occupancy number and B-factor. The first
            line contains the lattice parameters. For example,

                5.5930 9.8270    11.8080    90.0000    90.0000    90.0000
                C      0.8754    0.9424     0.0357     1.0000     1.1844
                O1-    0.8812    0.8253     0.9960     1.0000     1.1844
        """

        self.atoms = np.loadtxt(fname, skiprows=1, usecols=0, dtype=str)
        self.positions = np.loadtxt(fname, skiprows=1, usecols=(1, 2, 3))
        self.occupancy = np.loadtxt(fname, skiprows=1, usecols=4)
        self.bfactors = np.loadtxt(fname, skiprows=1, usecols=5)

        a, b, c, alpha, beta, gamma = np.loadtxt(fname,
                                                 max_rows=1,
                                                 usecols=(0, 1, 2, 3, 4, 5),
                                                 unpack=True)
        A, B, C, Ar, Br, Cr = lattice(a, b, c, alpha, beta, gamma)

        self.A = A
        self.B = B
        self.C = C
        self.Ar = Ar
        self.Br = Br
        self.Cr = Cr

    def show_lattice(self):

        """Display the direct and reciprocal lattices."""

        print("\nDirect lattice: \n")
        disp = str(np.row_stack((self.A, self.B, self.C)))
        print(disp.replace(' [', '').replace('[', '').replace(']', ''))

        print("\nReciprocal lattice: \n")
        disp = str(np.row_stack((self.Ar, self.Br, self.Cr)))
        print(disp.replace(' [', '').replace('[', '').replace(']', ''))
        print("\n")

    def bragg(self, E: float, hkl: list) -> np.ndarray:

        """Display inter planar distance and Bragg angle.

        Args:
            E (float): Beam energy (eV).
            hkl (list): Miller indices (e.g [1, 0, 0]).
        Returns:
            np.ndarray: Inter planar distance in index 0 and Bragg angle in 1.
        """

        wl = 12398.5 / E
        d = 2 * np.pi
        d = d / np.linalg.norm(hkl[0] * self.Ar + hkl[1] * self.Br + hkl[2] * self.Cr)
        thB = np.arcsin(wl / (2 * d)) * 180 / np.pi

        print("Energy = " + str(E) + " eV")
        print("d(" + str(hkl[0]) + "," + str(hkl[1]) + "," + str(hkl[2]) + ") = ",
              str(np.round(d, 3)) + "A  - thB = " + str(np.round(thB, 3)) + "º")

        return np.array([d, thB])

    def hkl2Q(self, hkl: ndarray) -> np.ndarray:

        """Returns the reciprocal vector from Miller indices.

        Args:
            hkl (ndarray): Miller indices (e.g [[1, 0, 0], [0, 1, 1]]).
        Returns:
            np.ndarray: Q values (1/angstrom).
        """

        hkl = np.array(hkl)
        Q = []
        if np.shape(hkl) == (3,):
            q = hkl[0] * self.Ar + hkl[1] * self.Br + hkl[2] * self.Cr
            Q.append(np.linalg.norm(q))
        else:
            for i in range(len(hkl)):
                q = hkl[i, 0] * self.Ar + hkl[i, 1] * self.Br + hkl[i, 2] * self.Cr
                Q.append(np.linalg.norm(q))

        return np.array(Q)

    def Fhkl(self, E: float or ndarray, H: ndarray) -> np.ndarray:

        """Returns the structure factor.

        Args:
            E (float or ndarray): Beam energy (eV).
            H (ndarray): Miller indices (e.g [[1, 0, 0], [0, 1, 1]]).
        Returns:
            np.ndarray: Complex structure factor list.
        Notes:
            Provide one energy value and an array of reflections or an array
            of energies and just one reflection.
        """

        E = np.array(E)
        H = np.array(H)

        sym = np.unique(self.atoms, return_index=False, return_inverse=False,
                        return_counts=True)[0]  # ions or atoms

        at = np.copy(sym)  # just atoms (ion don't contribute to resonance)
        idx = np.where(np.char.isalpha(sym) is False)
        if not np.shape(idx):
            pass
        else:
            ion = np.copy(sym[idx])
            regex = re.compile('[^a-zA-Z]')

            for i in range(len(ion)):
                at[idx[0][i]] = regex.sub('', ion[i])

        if np.shape(H) == (3,):
            F = 0
        else:
            F = np.zeros(len(H), dtype=np.clongdouble)

        Q = self.hkl2Q(H)
        for j in range(len(sym)):

            fn = asfQ(sym[j], Q / (4 * np.pi)) + aresE(at[j], E)

            idx = np.where(self.atoms == sym[j])

            if np.shape(H) != (3,):

                bf = np.reshape(np.tile(self.bfactors[idx], len(H)),
                                (len(H), len(self.bfactors[idx])))
                x = np.reshape(np.tile(self.positions[idx, 0][0], len(H)),
                               (len(H), len(self.positions[idx, 0][0])))
                y = np.reshape(np.tile(self.positions[idx, 1][0], len(H)),
                               (len(H), len(self.positions[idx, 1][0])))
                z = np.reshape(np.tile(self.positions[idx, 2][0], len(H)),
                               (len(H), len(self.positions[idx, 2][0])))
                occ = np.reshape(np.tile(self.occupancy[idx], len(H)), (len(H),
                                                                        len(self.occupancy[idx])))

                Ma = np.transpose([(Q / (4 * np.pi)) ** 2]) * bf
                frac = np.transpose([H[:, 0]]) * x + np.transpose([H[:, 1]]) * y
                frac = frac + np.transpose([H[:, 2]]) * z

                F = F + np.sum(np.transpose([fn]) * occ * np.exp(-Ma) * np.exp(2 * np.pi * 1j * frac),
                               axis=1)

            elif np.shape(E) != ():

                bf = np.reshape(np.tile(self.bfactors[idx], len(E)), (len(E),
                                                                      len(self.bfactors[idx])))
                x = np.reshape(np.tile(self.positions[idx, 0][0], len(E)),
                               (len(E), len(self.positions[idx, 0][0])))
                y = np.reshape(np.tile(self.positions[idx, 1][0], len(E)),
                               (len(E), len(self.positions[idx, 1][0])))
                z = np.reshape(np.tile(self.positions[idx, 2][0], len(E)),
                               (len(E), len(self.positions[idx, 2][0])))
                occ = np.reshape(np.tile(self.occupancy[idx], len(E)), (len(E),
                                                                        len(self.occupancy[idx])))

                Ma = (Q / (4 * np.pi)) ** 2 * bf
                frac = H[0] * x + H[1] * y + H[2] * z

                F = F + np.sum(np.transpose([fn]) * occ * np.exp(-Ma) * np.exp(2 * np.pi * 1j * frac),
                               axis=1)

            else:

                Ma = self.bfactors[idx] * (Q / (4 * np.pi)) ** 2
                frac = H[0] * self.positions[idx, 0] + H[1] * self.positions[idx, 1]
                frac = frac + H[2] * self.positions[idx, 2]

                F = F + np.sum(fn * self.occupancy[idx] * np.exp(-Ma) * np.exp(2 * np.pi * 1j * frac))

        return F

    def hkl_generate(self, E: float) -> tuple[ndarray, ndarray, ndarray]:

        """Returns the Miller indices of all allowed lattice planes.

        Args:
            E (float): Beam energy (eV).
        Returns:
            tuple[ndarray, ndarray, ndarray]: Miller indices, Bragg angles and inter planar distances.
        """

        wl = 12398.5 / E
        hmax = np.floor((4 * np.pi) / (wl * np.linalg.norm(self.Ar)))
        kmax = np.floor((4 * np.pi) / (wl * np.linalg.norm(self.Br)))
        lmax = np.floor((4 * np.pi) / (wl * np.linalg.norm(self.Cr)))

        z = np.array([0])
        H = np.arange(-hmax, hmax + 1)
        K = np.arange(-kmax, kmax + 1)
        L = np.arange(-lmax, lmax + 1)
        H = np.delete(H, np.argwhere(H == 0))
        K = np.delete(K, np.argwhere(K == 0))
        L = np.delete(L, np.argwhere(L == 0))
        H, K, L = np.append(H, z), np.append(K, z), np.append(L, z)

        Nmax = int((2 * hmax + 1) * (2 * kmax + 1) * (2 * lmax + 1))
        Qmax = (2 * np.pi) / wl
        HKL = np.zeros((Nmax, 3))
        th_hkl = np.zeros(Nmax)
        Dhkl = np.zeros(Nmax)

        m = 0
        for nh in range(0, int(2 * hmax)):

            h = H[nh]

            for nk in range(0, int(2 * kmax + 1)):

                k = K[nk]

                for nl in range(0, int(2 * lmax) + 1):

                    l = L[nl]
                    Q = np.linalg.norm(h * self.Ar + k * self.Br + l * self.Cr)

                    if Q / 2 < Qmax and Q > 0:
                        HKL[m, 0] = h
                        HKL[m, 1] = k
                        HKL[m, 2] = l
                        th_hkl[m] = 180 * np.arcsin(wl * Q / (4 * np.pi)) / np.pi
                        Dhkl[m] = 2 * np.pi / Q
                        m = m + 1

        h = 0
        for nk in range(0, int(2 * kmax)):

            k = K[nk]

            for nl in range(0, int(2 * lmax) + 1):

                l = L[nl]
                Q = np.linalg.norm(k * self.Br + l * self.Cr)

                if Q / 2 < Qmax and Q > 0:
                    HKL[m, 0] = h
                    HKL[m, 1] = k
                    HKL[m, 2] = l
                    th_hkl[m] = 180 * np.arcsin(wl * Q / (4 * np.pi)) / np.pi
                    Dhkl[m] = 2 * np.pi / Q
                    m = m + 1

        k = 0
        for nl in range(0, int(2 * lmax)):

            l = L[nl]
            Q = np.linalg.norm(l * self.Cr)

            if Q / 2 < Qmax and Q > 0:
                HKL[m, 0] = h
                HKL[m, 1] = k
                HKL[m, 2] = l
                th_hkl[m] = 180 * np.arcsin(wl * Q / (4 * np.pi)) / np.pi
                Dhkl[m] = 2 * np.pi / Q
                m = m + 1

        idx = []
        for j in range(Nmax):

            if HKL[j, 0] == 0 and HKL[j, 1] == 0 and HKL[j, 2] == 0:
                idx.append(j)

        HKL = np.delete(HKL, idx, 0)
        th_hkl = np.delete(th_hkl, idx)
        Dhkl = np.delete(Dhkl, idx)

        return HKL, th_hkl, Dhkl

    def diffraction(self, E: float, fout: str = '') -> tuple[ndarray, ndarray, ndarray]:

        """Returns the structure factor and inter planar distance of all lattice
        planes.

        Args:
            E (float): Beam energy (eV).
            fout (str): Name for save output. Does not save by default.
        Returns:
            tuple[ndarray, ndarray, ndarray]: Arrays of reflections, complex structure factor and inter planar distance.
        """

        HKL, th, d = self.hkl_generate(E)

        F = self.Fhkl(E, HKL)

        # Sorting by intensity
        idx = np.argsort(np.absolute(F) / np.max(np.absolute(F)))[::-1]
        F = F[idx]
        HKL = HKL[idx, :]
        th = th[idx]
        d = d[idx]

        if fout != '':
            save_diffraction(HKL, F, th, d, fout)

        return HKL, F, d

    def triplet_relation(self, E: float, G_H: list) -> tuple[float, float]:

        """Calculates the phase triplet relation.

        Args:
            E (float): Beam energy (eV).
            G_H (list): Indices of primary and secondary reflection. Should be in this order.
        Returns:
            tuple[float, float]: Cosine of phase triplet and interference amplitude W.
        """

        G_H = np.array(G_H)
        G_H = np.row_stack((G_H, G_H[0] - G_H[1]))
        W = self.Fhkl(E, G_H)
        W = (W[1] * W[2]) / W[0]

        return np.real(W) / np.absolute(W), np.absolute(W)

    def klines(self, E: float, G: list or ndarray, M: list or ndarray, Fmin: float, dw: float = 0.1):

        """Saves the Bragg Cones crossing the primary reflection.

        Args:
            E (float): Beam energy (eV).
            G (list or ndarray): Indices of primary reflection (eg, [1, 1, 1]).
            M (list or ndarray): Reference direction (eg, [1, 0, 0]).
            Fmin (float): Minimum cutoff value for W.
            dw (float): Maximum Bragg angle distance between secondary and primary.
        Notes:
            The BC lines are saved on `IN/OUT/THG_G_array_M_array_E_value.dat` files.
        """

        M, G = np.array(M), np.array(G)
        HKL, FH, d = self.diffraction(E)

        wl = 12398.5 / E
        th = np.arcsin(wl / (2 * d)) * 180 / np.pi
        FH = np.absolute(FH)

        idx = search_reflection(HKL, G)
        FG = FH[idx]
        thG = np.round(np.arcsin(wl / (2 * d[idx])) * 180 / np.pi, 4)

        HKL, FH, th = delete_multiple(HKL, FH, th, idx=idx)  # Exclude primary
        HKL, FH, GH, FGH, idx = coupling_reflection(FH, HKL, G)

        th = np.delete(th, idx)

        F = FGH * FH / FG
        HKL, F, th = delete_multiple(HKL, F, th, idx=np.argwhere(F <= Fmin))

        Z = np.matmul(G, np.array([self.Ar, self.Br, self.Cr]))  # Rotation axis
        Y = np.matmul(M, np.array([self.Ar, self.Br, self.Cr]))  # Φ-reference

        # xyz coordinate system
        z = Z / (np.sqrt(np.dot(Z, Z)))
        y = np.cross(z, Y) / (np.sqrt(np.dot(np.cross(z, Y), np.cross(z, Y))))
        x = np.cross(y, z)

        # secondary reflection on xyz system
        H = np.matmul(HKL, np.array([self.Ar, self.Br, self.Cr]))
        modH = np.sqrt(np.sum(np.multiply(H, H), axis=1))
        sinthH = np.sin(th * np.pi / 180)

        cosGH = np.sum(np.multiply(H, z), axis=1) / modH  # cos(γ)
        sinGH = np.sqrt(1 - cosGH ** 2)  # sin(γ)

        cosGH, sinthH, F, H, sinGH = delete_multiple(cosGH, sinthH, F, H, sinGH,
                                                     idx=np.argwhere(sinGH == 0))

        alphaH = np.arctan2(np.sum(np.multiply(H, y), axis=1),
                            np.sum(np.multiply(H, x), axis=1))  # α
        alphaH = alphaH * 180 / np.pi

        N = np.argwhere(alphaH < 0)
        kappa = alphaH[N] + 360
        alphaH[N] = kappa[:]

        dom = dw / 20
        omega = np.arange(thG - dw, thG + dw + dom, dom)

        h = np.matmul(H, np.linalg.inv(np.array([self.Ar, self.Br, self.Cr])))

        PHI_IN = np.zeros((len(alphaH) + 1, 4 + len(omega)))
        PHI_IN[0, 3] = thG
        PHI_IN[0, 4:] = omega[:]
        PHI_IN[1:, 0] = h[:, 0]
        PHI_IN[1:, 1] = h[:, 1]
        PHI_IN[1:, 2] = h[:, 2]
        PHI_IN[1:, 3] = F[:]
        PHI_OUT = np.copy(PHI_IN)

        fileout = "G_" + "".join((G.astype(int)).astype(str))
        fileout += "_M_" + "".join((M.astype(int)).astype(str))
        fileout += "_E_" + str(E) + ".dat"

        for i in range(len(omega)):

            Om = omega[i]
            omg = np.round(Om, 4)

            cosBH = (sinthH - np.sin(Om * np.pi / 180) * cosGH) / (np.cos(Om * np.pi / 180) * sinGH)
            N = np.argwhere(np.abs(cosBH) > 1)
            cosBH[N] = 1

            phi1 = alphaH - (np.arccos(cosBH)) * 180 / np.pi  # out-in

            N = np.argwhere(phi1 < 0)
            phi = phi1[N] + 360
            phi1[N] = phi[:]
            N = np.argwhere(phi1 > 360)
            phi = phi1[N] - 360
            phi1[N] = phi[:]

            phi2 = alphaH + (np.arccos(cosBH)) * 180 / np.pi  # in-out

            N = np.argwhere(phi2 > 360)
            phi = phi2[N] - 360
            phi2[N] = phi[:]
            N = np.argwhere(phi2 < 0)
            phi = phi2[N] + 360
            phi2[N] = phi[:]

            PHI_OUT[1:, 4 + i] = phi1[:]
            PHI_IN[1:, 4 + i] = phi2[:]

            if omg == thG:  # BC lines crossing primary

                M = np.zeros((len(PHI_IN) - 1, 6))
                M[:, :3] = np.copy(PHI_IN[1:, :3])
                M[:, 3] = np.copy(phi1[:])
                M[:, 4] = np.copy(phi2[:])
                M[:, 5] = np.copy(F[:])

                header = "h   k   l     out-in     in-out     FH*FGH"
                fmt = "%3.f %3.f %3.f %10.4f %10.4f %10.f"

                np.savetxt('THG_' + fileout, M, fmt=fmt, header=header)

        line1 = "      0          0         0      G-thBragg            "
        line1 += "omega (deg)         -       1st line"
        line2 = "      h          k         l      FH*FGH               "
        line2 += "phi (in-out)        -       following lines"
        header = '\n'.join([line1, line2])
        np.savetxt('IN_' + fileout, PHI_IN, fmt="%10.4f", header=header)

        line2 = "      h          k         l      FH*FGH               "
        line2 += "phi (out-in)        -       following lines"
        header = '\n'.join([line1, line2])
        np.savetxt('OUT_' + fileout, PHI_OUT, fmt="%10.4f", header=header)


class ExpData:

    def __init__(self, E: float, G: list, fname: str, colx: int = 0,
                 coly: int = 1, name: str = ''):

        """Given the experimental data, returns a new `exp` object.

        Args:
            E (float): Beam energy (eV).
            G (list): Indices of primary reflection (eg, [1, 1, 1]).
            fname (str): Experimental data file name.
            colx (int): Column number with angle values on data file. 1st column, by default.
            coly (int): Column number with intensity values. 2nd column, by default.
            name: Label for analysis (e.g. 'pure', 'doped'). Default: None.
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

    def plot(self) -> plotly.graph_objects:

        """Plot the experimental data.

        Returns:
            plotly.graph_objects: Interactive plot.
        """

        df_exp = pd.DataFrame(np.column_stack((self.phi, self.counts)),
                              columns=['phi', 'I'])

        fig = px.scatter(df_exp, x='phi', y='I',
                         labels={'phi': '𝜙 (deg)', 'I': 'Counts'})
        fig.data[0].update(mode='markers+lines')
        fig.update_traces(marker=dict(color='black', size=6),
                          line=dict(color='black', width=2))

        fig.update_yaxes(title_text='Counts',
                         range=[0, 1.1 * self.counts.max()])
        fig.update_xaxes(range=[self.phi.min(), self.phi.max()])
        fig.update_layout(legend_title_text='', template='plotly_white',
                          font=dict(size=16))
        fig.show()

    def BC_plot(self, fname: str, M: list, Fmin: float) -> plotly.graph_objects:

        """Plot the experimental data and BC lines.

        Args:
            fname (str): Structure file name (the same used by Crystal class).
            M (list): Reference direction (eg, [1, 0, 0]).
            Fmin (float): Minimum cutoff value for W (absolute).
        Returns:
            plotly.graph_objects: Interactive plot.
        """

        M = np.array(M)
        struc = Crystal(fname)
        struc.klines(self.energy, self.primary, M, Fmin)

        fname = "G_" + "".join((self.primary.astype(int)).astype(str))
        fname += "_M_" + "".join((M.astype(int)).astype(str))
        fname += "_E_" + str(self.energy) + ".dat"
        thG, FHFGH, df_in = dataframe_BClines('IN_' + fname)
        thG, FHFGH, df_out = dataframe_BClines('OUT_' + fname)

        subfigure = sp.make_subplots(rows=2, cols=1,
                                     shared_xaxes=True,
                                     vertical_spacing=0.02)

        # Experimental data subplot
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

        # BC lines subplot
        fig = px.line()
        for i in range(1, len(FHFGH)):
            hover_in = '<i>𝜙</i>: %{x:.3f} <br><b>F</b>:' + str(FHFGH[i - 1])
            hover_in += '<br><b>hkl</b>:' + df_in.columns[i]

            fig.add_scatter(x=df_in[df_in.columns[i]], y=df_in['ω'], name='',
                            hovertemplate=hover_in,
                            line=dict(color='Red',
                                      width=10 * FHFGH[i - 1] / np.max(FHFGH)))

            hover_out = '<i>𝜙</i>: %{x:.3f} <br><b>F</b>:' + str(FHFGH[i - 1])
            hover_out += '<br><b>hkl</b>:' + df_out.columns[i]

            fig.add_scatter(x=df_out[df_out.columns[i]], y=df_out['ω'], name='',
                            hovertemplate=hover_out,
                            line=dict(color='Blue',
                                      width=10 * FHFGH[i - 1] / np.max(FHFGH)))

        fig_traces = []
        for trace in range(len(fig["data"])):
            fig_traces.append(fig["data"][trace])
        for traces in fig_traces:
            subfigure.append_trace(traces, row=2, col=1)

        subfigure.add_shape(type='line', x0=self.phi.min(), y0=thG,
                            x1=self.phi.max(), y1=thG,
                            line=dict(color='black'),
                            row=2, col=1)
        subfigure.update_xaxes(title_text='𝜙 (deg)',
                               range=[self.phi.min(), self.phi.max()],
                               row=2, col=1,
                               showgrid=False)
        subfigure.update_yaxes(title_text='ω (deg)',
                               range=[thG - 0.1, thG + 0.1],
                               row=2, col=1,
                               showgrid=False)
        subfigure.update_layout(showlegend=False, template='plotly_white',
                                font=dict(size=16))

        subfigure.show()

    def peak_finder(self, minimum_intensity_p: float = 0.03):

        """Defines the peak list.

        Args:
            minimum_intensity_p (float): Minimum value of intensity to define a peak given as percentage of maximum
                value of spectrum. 3% by default.
        """

        yf = gaussian_filter1d(self.counts, 2)
        peaks, _ = find_peaks(yf, prominence=100,
                              height=self.counts.max() * minimum_intensity_p)

        self.peaks = peaks

    def data_cleaner(self, minimum_intensity_p: float = 0.2, points: int = 60):

        """Update the peak list after automatic clean the data.

        Args:
            minimum_intensity_p (float): Minimum value of intensity to define a peak given as percentage of maximum
                value of region. 20% by default.
            points (int): Max number of points defining a peak.
        """

        idx, i = [], 0
        for j in self.peaks:

            m, n = self.peak_definer(j, points)
            y = self.counts[m:n]

            height = (y.max() + (y[0] + y[-1]) / 2) * minimum_intensity_p
            pk, _ = find_peaks(gaussian_filter1d(y, 2), height=height)

            if len(pk) != 1:
                idx.append(i)
            i += 1

        self.peaks = np.delete(self.peaks, idx)

    def data_review(self):

        """Plot the experimental data highlighting peaks. Delete the peaks
        selected by external user.

        Note:
                To select a peak, stop the mouse over the maximum and press `SPACE`.
                To unselect, stop the mouse over the maximum and press `DEL`.
                Feel free to zoom in or out.
                After finishing the selection, press `ENTER`. Then, press `ENTER: int
                to close the figure.
        """

        plt.plot(self.phi, self.counts, c='k')
        plt.plot(self.phi[self.peaks], self.counts[self.peaks], "*b")
        plt.xlabel('ɸ (deg)')
        plt.ylabel('counts')
        plt.xlim(self.phi.min(), self.phi.max())
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
                diff = np.abs(self.phi[self.peaks] - phi_mk[j])
                idx.append(int(diff.argmin()))

            self.peaks = np.delete(self.peaks, idx)

    def region_of_fit(self, interval: int = 6, points: int = 60):

        """Defines the region of fit list.

        Args:
            interval (int): Number of standard deviations defining the region of fit. By default, 6-sigma is used.
            points (int): Max number of points defining a peak.
        """

        region, i = np.zeros((len(self.peaks), 4)), 0
        for j in self.peaks:
            m, n = self.peak_definer(j, points)
            x = self.phi[m:n]
            y = self.counts[m:n]

            s, d, f = fwhm(x, np.log10(y))

            N = np.abs(x - (d - interval * s)).argmin()
            M = np.abs(x - (d + interval * s)).argmin()

            region[i, :] = x[N], x[M], f, d
            i += 1

            self.region = region

    def MD_indexer(self, fname: str, M: list, Fmin: float):

        """Finds the secondary reflection list corresponding to each founded
        peak.

        Args:
            fname (str): Structure file name (the same used by Crystal class).
            M (list): Reference direction (eg, [1, 0, 0]).
            Fmin (float): Minimum cutoff value for W (absolute).

        Notes:
            Case two or more lattice planes are excited at the same angle with the same geometry:
            all will be displayed in the output file. If the geometries are opposite, the peak is
            not indexable (so it's excluded from the peak list). It's also not indexable the case
            with two excited reflections that distance themselves by less than 0.05º, whose smallest
            W value is at least half of the largest value. Except for these cases, the peak is indexed
            by the stronger reflection that is up to 0.1º of distance from the peak position.
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

            phi_in, fin, hkl_in = self.index_find(ph, 4, hkl)
            phi_out, fout, hkl_out = self.index_find(ph, 3, hkl)

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

        self.plot_indexation(HKL, s, idx)
        self.peaks = np.delete(self.peaks, idx)
        self.region = np.delete(self.region, idx, axis=0)
        self.index = np.column_stack((s, HKL))

    def plot_indexation(self, hkl: list, s: list, idx: list):

        """Plot the peaks and corresponding secondary reflection indexes.

        Args:
            hkl (list): Secondary reflections ('' are the unindexed cases).
            s (list): Diffraction geometry (in-out or out-in).
            idx (list): Unindexed cases index in peak list.
        """

        plt.plot(self.phi, self.counts, c='k')

        for i in range(len(s)):

            N = np.abs(self.phi - self.region[i, 0]).argmin()
            M = np.abs(self.phi - self.region[i, 1]).argmin()

            x = self.phi[N:M]
            y = self.counts[N:M]

            if i in idx:
                plt.plot(x, y, "r")
            else:
                if s[i] == 1:
                    c = "r"
                else:
                    c = "b"
                plt.annotate(hkl[i], (self.phi[self.peaks[i]], self.counts[self.peaks[i]]), color=c)

        plt.xlabel('ɸ (deg)')
        plt.ylabel('counts')
        plt.xlim(self.phi.min(), self.phi.max())
        plt.ylim(0, 1.1 * self.counts.max())
        plt.draw()
        while True:
            if plt.waitforbuttonpress(0):
                plt.close()
                break

    def plot_peaks(self):

        """Plot the selected peaks and corresponding region of fit."""

        plt.plot(self.phi, self.counts, c='k')

        for i in range(len(self.region[:, -1])):
            N = np.abs(self.phi - self.region[i, 0]).argmin()
            M = np.abs(self.phi - self.region[i, 1]).argmin()

            x = self.phi[N:M]
            y = self.counts[N:M]

            plt.plot(x, y, "r")

        plt.xlabel('ɸ (deg)')
        plt.ylabel('counts')
        plt.xlim(self.phi.min(), self.phi.max())
        plt.ylim(0, 1.1 * self.counts.max())
        plt.draw()
        while True:
            if plt.waitforbuttonpress(0):
                plt.close()
                break

    def data_fitter(self, nsamples: int = 1000):

        """Determine the slope and slope error list.

        Args:
            nsamples (int): Number of samples to calculate the slope error.
        """

        slope_err = np.zeros((len(self.peaks), 2))
        for i in tqdm(range(len(self.peaks))):
            N = np.abs(self.phi - self.region[i, 0]).argmin()
            M = np.abs(self.phi - self.region[i, 1]).argmin()

            x = self.phi[N:M]
            y = np.log10(self.counts[N:M])

            slope_err[i, 0] = peak_fit(x, y)

            sample = pd.DataFrame(np.column_stack((x, y)))
            slope_err[i, 1] = bootstrap_resampling(sample, nsamples)

        self.slope_data = slope_err

    def asymmetry_assigner(self, sbar: float = 0.1, tau: float = 0.4):

        """Determines the asymmetry list.

        Args:
            sbar (float): Minimum value of relative slope. Default = 10%.
            tau (float): Maximum value for ratio slope_error/slope. Default = 40%.
        """

        ratio = self.slope_data[:, 0] / np.abs(self.slope_data[:, 0].max())
        idx1 = np.where(np.abs(ratio) < sbar)

        if not np.array(idx1).size == 0:
            self.index, self.slope_data, self.region = delete_multiple(self.index,
                                                                       self.slope_data,
                                                                       self.region,
                                                                       idx=idx1)

        ratio = self.slope_data[:, 1] / self.slope_data[:, 0]
        idx2 = np.where(np.abs(ratio) > tau)
        if not np.array(idx2).size == 0:
            self.index, self.slope_data, self.region = delete_multiple(self.index,
                                                                       self.slope_data,
                                                                       self.region,
                                                                       idx=idx2)

        asy = np.zeros((len(self.slope_data[:, 0]), 2), dtype=str)
        asy[np.where(self.slope_data[:, 0] < 0), :] = 'H', 'L'
        asy[np.where(self.slope_data[:, 0] > 0), :] = 'L', 'H'
        asy = [''.join(row) for row in asy]

        self.asy = asy

    def save_output(self, fout: str = ''):

        """Save file with output of experimental data analysis.

        Args:
            fout (str): File name to save. Default:`asymmetries_NAME_E_value_G_indexes.dat`
        """

        sbar = self.slope_data[:, 0] / np.abs(self.slope_data[:, 0].max())
        tau = np.abs(self.slope_data[:, 1] / self.slope_data[:, 0])

        hkl = []
        for i in range(len(self.index[:, 1:])):
            hkl.append(self.index[i, -1].split('/ ')[0])

        M = np.column_stack((np.round(self.region[:, -1], 4),
                             self.asy, np.round(self.slope_data, 4),
                             np.round(sbar, 2), np.round(tau, 2),
                             self.index[:, 0], hkl, self.index[:, 1:]))

        header = '    phi    asy     slope   slope error   sbar (%)    tau (%)   s       hkl'

        if fout == '':
            fout = 'data' + self.name.upper() + '_E' + str(self.energy)
            fout += '_G' + "".join((self.primary.astype(int)).astype(str)) + '.dat'

        with open(fout, 'wb') as f:
            np.savetxt(f, [], header=header)
            for line in np.matrix(M):
                np.savetxt(f, line, fmt='%10s %5s %10s %10s %10s %10s %5s %10s %32s')

        return fout

    def peak_definer(self, peak_number: int, points: int) -> tuple:

        """Return the indexes of first and last points of peak.

        Args:
            peak_number (int): Index of peak point on angles array.
            points (int): Max number of points defining a peak.
        Returns:
            tuple: First and last points of peak.
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

    def index_find(self, ph: float, col: int, hkl: np.ndarray) -> tuple:

        """Returns the closer multiple-diffraction case of certain phi.

        Args:
            ph (float): Angle on Renninger scan.
            col (int): 3 or 4 (in-out or out-in BC lines).
            hkl (np.ndarray): Miller indices.
        Returns:
            tuple: Angle position, W amplitude, reflection indices.
        """

        k = np.where(np.abs(np.round(self.BC[:, col], 3) - ph) < 0.1)

        if not np.array(k).size == 0:
            m = np.argsort(np.abs(np.round(self.BC[k, col], 3) - ph))[::-1]
            phi_values = np.round(self.BC[k[0], col], 3)
            w_values = self.BC[k[0], -1]

            equal = np.all(phi_values == phi_values[m[0]][0])
            if equal:  # all BCs crossing in the same phi
                PHI = phi_values[m[0]][0]
                W = w_values[m[0]][0]
                miller = '/ '.join(hkl[k][m][0])

            elif phi_values[m[0]][0] - phi_values[m[0]][1] < 0.01:  # does not cross in the same phi, but close of that
                phi1 = phi_values[m[0]][0]
                phi2 = phi_values[m[0]][1]
                W1 = w_values[m[0]][0]
                W2 = w_values[m[0]][1]

                if phi1 == phi2 and W1 == W2:
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
        else: # does not cross
            miller = ''
            PHI = 0
            W = 0

        return PHI, W, miller

    def analyze(self, fname: str, M: list, Fmin: float, p_find: float = 0.03,
                p_clean: float = 0.2, points: int = 60, sbar: float = 0.1,
                tau: float = 0.4, fout: str = ''):

        """Executes the entire peak analysis.

        Args:
            fname (str): Structure file name (the same used by Crystal class).
            M (list): Reference direction (eg, [1, 0, 0]).
            Fmin (float): Minimum cutoff value for W (absolute).
            p_find (float): Minimum value of intensity to define a peak (percentage of maximum value of spectrum).
            3% by default.
            p_clean (float): Minimum value of intensity to define a peak (percentage of maximum value of region).
            20% by default.
            points (int): Max number of points defining a peak.
            sbar (float): Minimum value of relative slope. Default = 10%.
            tau (float): Maximum value for ratio slope_error/slope. Default = 40%.
            fout (str): File name to save. Default:`asymmetries_NAME_E_value_G_indexes.dat`
        Notes:
            This method does not work properly on notebooks.
        """

        f = Figlet(font='banner3-D', width=50, justify="center")
        print(colored(f.renderText('xrdd_asy'), 'green'))

        spinner = Halo(text='Finding peaks...', spinner='dots')
        spinner.start()
        self.peak_finder(p_find)
        spinner.stop()
        spinner.stop_and_persist(symbol='🗸',
                                 text='Finding peaks:\033[1m\033[92m Success')
        print('\t Minimum of counts = ', int(p_find * self.counts.max()))
        print('\t\033[1m' + ' ' + str(len(self.peaks)) + ' found peaks.' + '\033[0m')

        spinner = Halo(text='Automatic cleaning the data...', spinner='dots')
        spinner.start()
        self.data_cleaner(p_clean, points=points)
        spinner.stop()
        spinner.stop_and_persist(symbol='🗸',
                                 text='Automatic cleaning the data:\033[1m\033[92m Success')
        print('\t\033[1m' + ' ' + str(len(self.peaks)) + ' selected peaks.' + '\033[0m')

        spinner = Halo(text='Manual cleaning the data...', spinner='dots')
        spinner.start()
        self.data_review()
        spinner.stop()
        spinner.stop_and_persist(symbol='🗸',
                                 text='Manual cleaning the data:\033[1m\033[92m Success')
        print('\t\033[1m' + ' ' + str(len(self.peaks)) + ' selected peaks.' + '\033[0m')

        spinner = Halo(text='Defining region of fit...', spinner='dots')
        spinner.start()
        self.region_of_fit(points=points)
        spinner.stop()
        spinner.stop_and_persist(symbol='🗸',
                                 text='Defining region of fit:\033[1m\033[92m Success')

        spinner = Halo(text='Indexing peaks...', spinner='dots')
        spinner.start()
        self.MD_indexer(fname, M, Fmin)
        spinner.stop()
        spinner.stop_and_persist(symbol='🗸',
                                 text='Indexing peaks:\033[1m\033[92m Success')
        print('\t\033[1m' + 'Data set with ' + str(len(self.peaks)) + ' peaks.' + '\033[0m')
        print('\tMinimum value of FHFGH = ', Fmin)
        self.plot_peaks()

        print(':: Fitting peaks...')
        self.data_fitter()
        sys.stdout.write("\033[2F")
        print('🗸 Fitting peaks:\033[1m\033[92m Success')

        spinner = Halo(text='Assigning asymmetry type...', spinner='dots')
        spinner.start()
        self.asymmetry_assigner(sbar, tau)
        spinner.stop()
        spinner.stop_and_persist(symbol='🗸',
                                 text='Assigning asymmetry type:\033[1m\033[92m Success')
        print('\tCUTOFF: slope > ', sbar)
        print('\t            τ < ', tau)

        spinner = Halo(text='Saving file...', spinner='dots')
        spinner.start()
        fn = self.save_output(fout)
        spinner.stop()
        spinner.stop_and_persist(symbol='🗸',
                                 text='Saving file:\033[1m\033[92m Success')
        print('\t\033[1m' + fn + '\033[0m')


def phase_comparison(data1: np.ndarray, data2: np.ndarray) -> plotly.graph_objects:
    """Display and plot the structure factor phase variation between two models.

    Args:
        data1 (np.ndarray): Complex structure factor list for structure 1 (obtained from `Crystal().diffraction()`)
        data2 (np.ndarray): Complex structure factor list for structure 2.
    Returns:
            plotly.graph_objects: Interactive plot.
    """

    HKL1, F1, d1 = data1[0], data1[1], data1[2]
    HKL2, F2, d2 = data2[0], data2[1], data2[2]

    Z, Q, F, HKL = [], [], [], []

    for i in range(len(HKL1)):

        x = np.where(HKL2[:, 0] == HKL1[i, 0])
        y = np.where(HKL2[:, 1] == HKL1[i, 1])
        z = np.where(HKL2[:, 2] == HKL1[i, 2])

        m = reduce(np.intersect1d, (x, y, z))

        if len(m) != 0:
            m = int(m[0])
            Z.append(np.abs(np.angle(F1[i]) - np.angle(F2[m])) * 180 / np.pi)
            HKL.append(HKL1[i, :])
            Q.append(2 * np.pi / d1[i])
            F.append(np.absolute(F1[i]))

    Z = np.array(Z, dtype=float)
    Q = np.array(Q, dtype=float)
    F = np.array(F, dtype=float)
    HKL = np.array(HKL, dtype=float)

    Z, Q, F, HKL = delete_multiple(Z, Q, F, HKL, idx=np.argwhere(Z < 15))
    Z, Q, F, HKL = delete_multiple(Z, Q, F, HKL, idx=np.argwhere(Z > 180))

    HKL = (HKL.astype(int)).astype(str)
    for i in range(len(HKL)):
        HKL[i] = " ".join(HKL[i])
    HKL = HKL[:, 0]

    df = pd.DataFrame(np.column_stack((HKL, np.round(F, 2), np.round(Q, 2),
                                       np.round(Z, 2))),
                      columns=['hkl', 'F', 'Q', 'Z'])
    date = datetime.now().strftime("%Y%m%d_%X")
    name = 'PHASE' + date + '.txt'

    np.savetxt(name, df.values, fmt='% 10s',
               header='     hkl         |F|²      Q (Å)    Δδ (deg)')

    print('Saved in', name)

    comparison_plot(df, 'Δδ (deg)', '|F|²')


def phase_triplet_comparison(data1: np.ndarray, data2: np.ndarray, G: list, wmin: float = 5) -> plotly.graph_objects:
    """Display and plot the phase triplet variation between two models for a given primary reflection.

    Args:
        data1 (np.ndarray): Complex structure factor list for structure 1 (obtained from `Crystal().diffraction()`)
        data2 (np.ndarray): Complex structure factor list for structure 2.
        G (list): Indices of primary reflection (eg, [1, 1, 1]).
        wmin (float): Cutoff for interference amplitude as percentage. 5% by default.
    Returns:
            plotly.graph_objects: Interactive plot.
    """

    G = np.array(G)
    HKL1, F1, d1 = data1[0], data1[1], data1[2]
    HKL2, F2, d2 = data2[0], data2[1], data2[2]

    idx1 = search_reflection(HKL1, G)
    idx2 = search_reflection(HKL2, G)

    if not idx1:
        print('Primary reflection not found for 1st structure.')
        return
    elif not idx2:
        print('Primary reflection not found for 2nd structure.')
        return
    elif np.absolute(F1[idx1]) == 0:
        print('Null primary reflection in 1st structure.')
        return
    elif np.absolute(F2[idx2]) == 0:
        print('Null primary reflection in 2nd structure.')
        return

    FG1 = F1[idx1]
    FG2 = F2[idx2]

    HKL1, F1, d1 = delete_multiple(HKL1, F1, d1, idx=idx1)
    HKL2, F2, d2 = delete_multiple(HKL2, F2, d2, idx=idx2)

    HKL1, F1, GH1, FGH1, i = coupling_reflection(F1, HKL1, G)
    HKL2, F2, GH2, FGH2, i = coupling_reflection(F2, HKL2, G)

    Z, Q, W, HKL = [], [], [], []

    for i in range(len(HKL1)):

        x = np.where(HKL2[:, 0] == HKL1[i, 0])
        y = np.where(HKL2[:, 1] == HKL1[i, 1])
        z = np.where(HKL2[:, 2] == HKL1[i, 2])

        m = reduce(np.intersect1d, (x, y, z))

        if len(m) != 0:

            m = int(m[0])

            cpsi1, psi1, w1 = triplet_calculation(FG1, F1[i], FGH1[i])
            cpsi2, psi2, w2 = triplet_calculation(FG2, F2[m], FGH2[m])

            if cpsi1 * cpsi2 < 0:  # inverted asymmetries
                HKL.append(HKL1[i, :])
                Z.append(np.abs(psi1 - psi2) * 180 / np.pi)
                Q.append(2 * np.pi / d1[i])
                W.append(w1)

    Z = np.array(Z, dtype=float)
    Q = np.array(Q, dtype=float)
    W = np.array(W, dtype=float)
    HKL = np.array(HKL, dtype=float)

    W = 100 * W / W.max()
    HKL, Z, Q, W = delete_multiple(HKL, Z, Q, W, idx=np.argwhere(Z < 1))
    HKL, Z, Q, W = delete_multiple(HKL, Z, Q, W, idx=np.argwhere(Z > 180))
    HKL, Z, Q, W = delete_multiple(HKL, Z, Q, W, idx=np.argwhere(W < wmin))

    HKL = (HKL.astype(int)).astype(str)
    for i in range(len(HKL)):
        HKL[i] = " ".join(HKL[i])
    HKL = HKL[:, 0]

    df = pd.DataFrame(np.column_stack((HKL, np.round(W, 2), np.round(Q, 2),
                                       np.round(Z, 2))),
                      columns=['hkl', 'F', 'Q', 'Z'])

    date = datetime.now().strftime("%Y%m%d_%X")
    name = 'TRIPLET_G_' + "".join((G.astype(int)).astype(str)) + '_' + date + '.txt'

    np.savetxt(name, df.values, fmt='% 10s',
               header='     hkl         W (%)      Q (Å)    ΔΨ (deg)')

    print('Saved in', name)

    comparison_plot(df, 'ΔΨ (deg)', 'W (%)', 'blue')
