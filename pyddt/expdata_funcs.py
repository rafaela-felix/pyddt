#!/usr/bin/python3

"""
**Functions required by the Expdata class.**
"""

import numpy as np
import pandas as pd
import scipy.stats as st
from lmfit.models import LinearModel, GaussianModel
from sklearn.utils import resample

__author__ = "Rafaela Felix"
__credits__ = {"Rafaela Felix", "Sergio Morelhão"}
__version__ = "1.0"
__maintainer__ = "Rafaela Felix"
__email__ = "rafaelafelixp@usp.br"
__status__ = "Production"


def dataframe_BClines(fname: str) -> tuple:

    """Constructs the required dataframe for BC lines plot.

    Args:
        fname (str): Filename with BC lines.

    Returns:
        tuple: Bragg angle of primary reflection, FHFGH values, `pd.DataFrame` with BC lines.
    """

    data = np.loadtxt(fname)
    thG, FHFGH, omega, phi = data[0, 3], data[1:, 3], data[0, 4:], data[1:, 4:]

    phi[np.where(phi > 180)] -= 360

    s = ['ω']
    hkl = (data[1:, 0:3].astype(int)).astype(str)
    for i in range(len(hkl)):
        s.append("".join(hkl[i]))

    M = np.concatenate((np.array([omega]).T, phi.T), axis=1)
    df = pd.DataFrame(M, columns=s)

    return thG, FHFGH, df


def fwhm(x: np.ndarray, y: np.ndarray) -> tuple:

    """Calculates the full width at half maximum of a peak.

    Args:
        x (np.ndarray): Data points - azimuth angle.
        y (np.ndarray): Data points - intensity.

    Returns:
        tuple: *full width at half maximum* (fwhm).
    """

    xbar = x[5:-5]
    ybar = y[5:-5]

    yave = np.mean(ybar)
    ymed = (ybar.max() - yave) / 2 + yave
    idxbar = np.argsort(np.abs(ybar - ymed))

    if np.abs(xbar[idxbar[0]] - xbar[idxbar[1]]) > 0.01:
        f = np.abs(xbar[idxbar[0]] - xbar[idxbar[1]])
    else:
        if np.abs(xbar[idxbar[0]] - xbar[idxbar[2]]) > 0.01:
            f = np.abs(xbar[idxbar[0]] - xbar[idxbar[2]])
        else:
            f = np.abs(xbar[idxbar[0]] - xbar[idxbar[3]])

    return f


def peak_fit(x: np.ndarray, y: np.ndarray) -> float:

    """Fits the Gaussian + linear model.

    Args:
        x (np.ndarray): Data points (angle values).
        y (np.ndarray): Data points (intensity values).

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

    """Calculates the slope error using the bootstrap resampling.

    Args:
        sample (pd.DataFrame): Dataframe with x and y values.
        n (int): Number of samples.

    Returns:
        float: Width of slope distribution.
    """

    S = np.zeros(n)

    for i in range(0, n):
        boot = resample(sample, replace=True, n_samples=len(sample))
        S[i] = peak_fit(boot[0].to_numpy(), boot[1].to_numpy())

    return st.tstd(S)
