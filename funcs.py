#!/usr/bin/env python

"""These tools are used by the main script XRDD. They perform simple
manipulation, output or plot operations."""

import numpy as np
import plotly
import plotly.graph_objects as go
from functools import reduce
from lmfit.models import LinearModel, GaussianModel
from numpy import ndarray
from sklearn.utils import resample
import scipy.stats as st
import pandas as pd


# --- Functions used by Crystal class ---

def lattice(a: float, b: float, c: float, alpha: float, beta: float, gamma: float) -> tuple[np.ndarray, np.ndarray,
                                                                                            np.ndarray, np.ndarray,
                                                                                            np.ndarray, np.ndarray]:
    """Calculate the direct and reciprocal vectors.

    Args:
        a (float): 1st lattice parameter amplitude (angstrom).
        b (float): 2nd lattice parameter amplitude (angstrom).
        c (float): 3rd lattice parameter amplitude (angstrom).
        alpha (float): 1st lattice angle (degree).
        beta (float): 2nd lattice angle (degree).
        gamma (float): 3rd lattice angle (degree).
    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Direct and reciprocal vectors.
    """

    alpha, beta, gamma = np.radians(alpha), np.radians(beta), np.radians(gamma)

    cphi = (np.cos(gamma) - np.cos(beta) * np.cos(alpha)) / (np.sin(alpha) * np.sin(beta))
    sphi = np.sqrt(1 - cphi ** 2)

    A = a * np.array([np.sin(beta), 0, np.cos(beta)])
    B = b * np.array([np.sin(alpha) * cphi, np.sin(alpha) * sphi, np.cos(alpha)])
    C = c * np.array([0, 0, 1])

    Vcel = np.dot(A, np.cross(B, C))

    Ar = 2 * np.pi * np.cross(B, C) / Vcel
    Br = 2 * np.pi * np.cross(C, A) / Vcel
    Cr = 2 * np.pi * np.cross(A, B) / Vcel

    return A, B, C, Ar, Br, Cr


def save_diffraction(HKL: list, F: list, th: list, d: list, fout: str):

    """Format and save the output of diffraction method.

    Args:
        HKL (list): Miller indices.
        F (list): Structure factors.
        th (list): Bragg angles.
        d (list): Inter planar distances.
        fout (str): File name for saving.
    """

    M = np.zeros((len(F), 10))
    M[:, 0] = HKL[:, 0]
    M[:, 1] = HKL[:, 1]
    M[:, 2] = HKL[:, 2]
    M[:, 3] = np.real(F)
    M[:, 4] = np.imag(F)
    M[:, 5] = np.absolute(F)
    M[:, 6] = np.angle(F) * 180 / np.pi
    M[:, 7] = th
    M[:, 8] = d
    M[:, 9] = 100 * np.absolute(F) ** 2 / np.max(np.absolute(F) ** 2)

    form = " %3.f %3.f %3.f %10.3f  %10.3f %10.3f %10.1f %10.4f %10.4f %10.1f "

    header = "h   k   l       Re(F)        Im(F)       |F|          ph(º)     "
    header = header + "th(º)         d           I(%)"

    np.savetxt(fout, M, fmt=form, header=header)


def delete_multiple(*args: ndarray or list[ndarray], idx: list or np.ndarray or int) -> list[ndarray]:

    """Delete the same positions of multiple arrays after.

    Args:
        args (list[ndarray]): Arrays to delete positions (eg 'arr1, arr2, arr3...').
        idx (list or ndarray or int): Index of positions to delete. Please use 'idx=' on function call.
    Returns:
         list[ndarray]: Arrays after deleting positions.
    """

    arr = []
    for arg in args:
        arr.append(np.delete(arg, idx, axis=0))
    return [comp for comp in arr]


def comparison_plot(dataframe: pd.DataFrame, zlabel: str, xlabel: str,
                    color: str = 'darkred') -> plotly.graph_objects:
    """Plot phase comparison.

    Args:
        dataframe (pd.DataFrame): Dataframe with x, y and z coordinates labeled as F, Q and Z.
        zlabel (str): Title for Z axes.
        xlabel (str): Title for X axes.
        color (str): Color of markers. Darkred by default.
    Returns:
        plotly.graph_objects: Interactive plot.
    """

    x = np.array(dataframe['F'], dtype=float)
    y = np.array(dataframe['Q'], dtype=float)
    z = np.array(dataframe['Z'], dtype=float)

    layout = go.Layout(scene=dict(xaxis=dict(title=xlabel),
                                  yaxis=dict(title='Q (Å)'),
                                  zaxis=dict(title=zlabel, range=[0, 200]),
                                  ))

    fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z,
                                       name='', mode='markers',
                                       hovertemplate='<br><b>hkl</b>:' + dataframe['hkl'])],
                    layout=layout)

    fig.update_traces(marker=dict(color=color, size=3, line=dict(color='black',
                                                                 width=2)))

    for i in range(len(z)):
        fig.add_trace(
            go.Scatter3d(x=[x[i], x[i]], y=[y[i], y[i]], z=[0, z[i]], name='',
                         mode='lines', line=dict(color='Black', width=1.5),
                         hovertemplate='<br><b>hkl</b>:' + dataframe['hkl']))

    fig.update_layout(template='seaborn', showlegend=False)

    fig.show()


def search_reflection(HKL: np.ndarray, G: np.ndarray) -> int:
    """Finds the index of primary reflection in HKL array.

    Args:
        HKL (np.ndarray): Array with Miller indices (eg [[arr1], [arr2], [...]]).
        G (np.ndarray): Miller indices of primary reflection (eg [1, 0, 0]).
    Returns:
        int: Index of primary reflection in HKL array.
    """

    x = np.where(HKL[:, 0] == G[0])
    y = np.where(HKL[:, 1] == G[1])
    z = np.where(HKL[:, 2] == G[2])

    m = reduce(np.intersect1d, (x, y, z))

    if len(m) == 0:
        return []
    else:
        return int(m[0])


def coupling_reflection(F: np.ndarray, H: np.ndarray, G: np.ndarray) -> tuple[ndarray, ndarray, ndarray, ndarray, int]:
    """Find the coupling reflection indexes and respective structure factors.

    Args:
        F (np.ndarray): Complex structure factors.
        H (np.ndarray): Array with Miller indices.
        G (np.ndarray): Miller indices of primary reflection.
    Returns:
        tuple[ndarray, ndarray, ndarray, ndarray, int]: Secondary reflections, structure factors of secondary
             reflections, coupling reflection indices, structure factors of coupling reflections, index of not found
             coupling reflections.
    """

    GH = G - H
    FGH = []
    idx = []

    for i in range(len(GH)):

        x = np.where(H[:, 0] == GH[i, 0])
        y = np.where(H[:, 1] == GH[i, 1])
        z = np.where(H[:, 2] == GH[i, 2])

        m = reduce(np.intersect1d, (x, y, z))

        if len(m) != 0:

            m = int(m[0])
            FGH.append(F[m])

        else:
            idx.append(i)

    return np.delete(H, idx, axis=0), np.delete(F, idx), GH, np.array(FGH), idx


def triplet_calculation(FG: np.ndarray, FH: np.ndarray, FGH: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculates the phase triplet relation.

    Args:
        FG (np.ndarray): Complex structure factor of primary reflection.
        FH (np.ndarray): Complex structure factor of secondary reflection.
        FGH (np.ndarray): Complex structure factor of coupling reflection.
    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: Cosine of phase triplet, phase triplet (radians) and
            interference amplitude.
    Notes:
         In case the structure factors are unknown, use the triplet_relation method for a Crystal object.
    """

    W = (FH * FGH) / FG

    return np.real(W) / np.absolute(W), np.angle(W), np.absolute(W)


# --- Functions used by ExpData class ---

def dataframe_BClines(fname: str) -> tuple:

    """Construct the dataframe for BCs plot.

    Args:
        fname (str): File name with BC lines (eg 'IN/OUT_G_indexes_M_array_E_value.dat).
    Returns:
        tuple: Bragg angle of primary reflection, FHFGH values, pd.DataFrame with BC lines.
    """

    data = np.loadtxt(fname)
    thG, FHFGH, omega, phi = data[0, 3], data[1:, 3], data[0, 4:], data[1:, 4:]

    s = ['ω']
    hkl = (data[1:, 0:3].astype(int)).astype(str)
    for i in range(len(hkl)):
        s.append("".join(hkl[i]))

    M = np.concatenate((np.array([omega]).T, phi.T), axis=1)
    df = pd.DataFrame(M, columns=s)

    return thG, FHFGH, df


def fwhm(x: np.ndarray, y: np.ndarray) -> tuple:

    """Calculate peak properties manually.

    Args:
        x (np.ndarray): Data points (angle positions).
        y (np.ndarray): Data points (intensity values)
    Returns:
        tuple: standard deviation, center position and full width at half maximum  (fwhm).
    """

    xbar = x[5:-5]
    ybar = y[5:-5]
    idxbar = np.argsort(np.abs(ybar - (ybar.max() + (ybar[-1] + ybar[0]) / 2) / 2))

    if np.abs(xbar[idxbar[0]] - xbar[idxbar[1]]) > 0.01:
        f = np.abs(xbar[idxbar[0]] - xbar[idxbar[1]])
    else:
        if np.abs(xbar[idxbar[0]] - xbar[idxbar[2]]) > 0.01:
            f = np.abs(xbar[idxbar[0]] - xbar[idxbar[2]])
        else:
            f = np.abs(xbar[idxbar[0]] - xbar[idxbar[3]])

    s = f/2.355
    d = xbar[ybar.argmax()]

    return s, d, f


def peak_fit(x: np.ndarray, y: np.ndarray) -> float:

    """Fits Gaussian + linear model to peak.

    Args:
        x (np.ndarray): Data points (angle positions).
        y (np.ndarray): Data points (intensity values)
    Returns:
        float: Slope value.
    """

    gauss = GaussianModel(prefix='g_')
    linear = LinearModel(prefix='l_')
    model = gauss + linear

    pars = gauss.guess(y, x=x)
    pars += linear.guess(y, x=x)

    out = model.fit(y, pars, x=x)

    return out.params['l_slope'].value


def bootstrap_resampling(sample: pd.DataFrame, n: int) -> float:

    """Calculates the slope error by bootstrap resampling.

    Args:
        sample (pd.DataFrame): Dataframe with x and y values.
        n (int): Number of samples.
    Returns:
        float: Width of slope distribution (slope error).
    """

    S = np.zeros(n)

    for i in range(0, n):
        boot = resample(sample, replace=True, n_samples=len(sample))
        S[i] = peak_fit(boot[0].to_numpy(), boot[1].to_numpy())

    return st.tstd(S)
