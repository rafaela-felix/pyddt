#!/usr/bin/python3

"""
This module provides the calculation of atomic scattering amplitudes (see `Computer Simulation Tools for X-ray Analysis: Scattering and Diffraction Methods <https://link.springer.com/book/10.1007/978-3-319-19554-4>`_
for reference).
"""

from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d

__author__ = "Rafaela Felix"
__credits__ = {"Rafaela Felix", "Sergio Morelhão"}
__version__ = "1.0"
__maintainer__ = "Rafaela Felix"
__email__ = "rafaelafelixp@usp.br"
__status__ = "Production"


def asfQ(atom: str, Q: list or float or np.ndarray) -> np.ndarray or float:

    """Calculates the atomic scattering factor using the Cromer–Mann coefficients (highly inaccurate for
    Q > 30 1/angstrom).

    Args:
        atom (str): Atom or ion symbol.
        Q (list or float or np.ndarray): Reciprocal vector amplitude (1/angstrom) divided by 4pi.

    Returns:
        np.ndarray or float: Atomic scattering factor values.

    Usage:
        * ``asfQ('Na1+', np.linspace(0, 10, 1000))``
        * ``asfQ('H', 3)``
        * ``asfQ('Se', [0, 10, 20])``
    """

    crom = Path(__file__).parent / "f0_CromerMann.txt"

    el = np.loadtxt(crom, dtype='str', usecols=0)
    coef = np.loadtxt(crom,
                      usecols=(1, 2, 3, 4, 5, 6, 7, 8, 9))

    idx = np.where(el == atom)

    if idx[0].shape[0] == 0:
        rank = np.array([SequenceMatcher(None, atom, el[i]).ratio() for i in range(len(el))])
        print(atom, 'is not included in Cromermann factors. Replaced by', el[rank.argmax()])  # logging
        idx = np.where(el == el[rank.argmax()])

    an = coef[idx, 0:4]
    bn = coef[idx, 4:8]
    c = coef[idx, 8]

    if not np.shape(Q):

        f0 = np.sum(an * np.exp(-bn * Q ** 2)) + c

    else:
        Q = np.array(Q)
        f0 = np.zeros(len(Q))

        i = 0
        while i < len(Q):
            f0[i] = np.sum(an * np.exp(-bn * (Q[i] ** 2))) + c
            i = i + 1

    return f0


def aresE(atom: str, E: np.ndarray or float) -> np.ndarray or float:

    """Calculates the atomic resonance amplitude by using linear interpolation of tabulated values.

    Args:
        atom (str): Element symbol.
        E (np.ndarray or float): X-ray energy (from 1004.16 to 70000 eV).

    Returns:
        np.ndarray or float: Complex resonance amplitude.

    Usage:
        * ``aresE('Na', np.linspace(3000, 10000, 1000))``
        * ``aresE('O', 8048)``

    """

    at = Path(__file__).parent / "atomnm.txt"
    val = Path(__file__).parent / "dispersion.txt"

    L = np.loadtxt(at, dtype='int', usecols=(1, 2))
    el = np.loadtxt(at, dtype='str', usecols=0)
    Z = np.loadtxt(val)

    idx = np.where(el == atom)

    if not np.shape(idx[0])[0]:

        return 0

    else:

        E = np.array(E)
        W = Z[int(L[idx, 0] - 1):int(L[idx, 1] - 1), :]

        f1 = interp1d(W[:, 0], W[:, 1], kind='linear')
        f2 = interp1d(W[:, 0], W[:, 2], kind='linear')

        return f1(E) + 1j * f2(E)
