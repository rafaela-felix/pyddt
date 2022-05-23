#!/usr/bin/env python

"""These tools are used by the main script XRDD. They perform the calculation of atomic scattering factor and
resonance amplitude."""

import numpy as np
from scipy.interpolate import interp1d


def asfQ(atom: str, Q: list or float) -> np.ndarray or float:

    """Calculates the atomic scattering factor.

    Args:
        atom (str): Atom or ion symbol (eg 'Na3+'. 'H')
        Q (list or float): Reciprocal vector.
    Returns:
        np.ndarray or float: Atomic scattering factor values.
    """

    el = np.loadtxt('f0_CromerMann.txt', dtype='str', usecols=0)
    coef = np.loadtxt('f0_CromerMann.txt',
                      usecols=(1, 2, 3, 4, 5, 6, 7, 8, 9))

    idx = np.where(el == atom)

    an = coef[idx, 0:4]
    bn = coef[idx, 4:8]
    c = coef[idx, 8]

    if not np.shape(Q):

        f0 = np.sum(an*np.exp(-bn*Q**2)) + c

    else:
        Q = np.array(Q)
        f0 = np.zeros(len(Q))

        i = 0
        while i < len(Q):

            f0[i] = np.sum(an*np.exp(-bn*(Q[i]**2))) + c
            i = i + 1

    return f0


def aresE(atom: str, E: np.ndarray or float) -> np.ndarray or float:

    """Calculates the resonance amplitude.

    Args:
        atom (str: Element symbol (eg 'Se','Fe')
        E (np.ndarray or float): Beam energy (eV).
    Returns:
        np.ndarray or float: Resonance amplitude values.
    """

    L = np.loadtxt('atomnm.txt', dtype='int', usecols=(1, 2))
    el = np.loadtxt('atomnm.txt', dtype='str', usecols=0)
    Z = np.loadtxt('dispersion.txt')

    idx = np.where(el == atom)

    if not np.shape(idx[0])[0]:

        return 0

    else:

        E = np.array(E)
        W = Z[int(L[idx, 0]-1):int(L[idx, 1]-1), :]

        f1 = interp1d(W[:, 0], W[:, 1], kind='linear')
        f2 = interp1d(W[:, 0], W[:, 2], kind='linear')

        return f1(E) + 1j*f2(E)
