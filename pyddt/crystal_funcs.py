#!/usr/bin/python3

"""
**Functions required by the crystal class.**
"""

import numpy as np

__author__ = "Rafaela Felix"
__credits__ = {"Rafaela Felix", "Sergio Morelhão"}
__version__ = "1.0"
__maintainer__ = "Rafaela Felix"
__email__ = "rafaelafelixp@usp.br"
__status__ = "Production"


def lattice(a: float, b: float, c: float, alpha: float, beta: float, gamma: float) -> tuple:

    """Calculates the direct and reciprocal vectors.

    Args:
        a (float): 1st lattice parameter (amplitude).
        b (float): 2nd lattice parameter (amplitude).
        c (float): 3rd lattice parameter (amplitude).
        alpha (float): 1st lattice angle.
        beta (float): 2nd lattice angle.
        gamma (float): 3rd lattice angle.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Direct and reciprocal vectors.

    Notes:
        **Expected units**: angstrom and degrees.
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

    """Setting the output for the diffraction method.

    Args:
        HKL (list): Miller indices.
        F (list): Structure factors.
        th (list): Bragg angles.
        d (list): Interplanar distances.
        fout (str): File name for output.
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


def phase_triplet(FG: np.ndarray, FH: np.ndarray, FGH: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    """Calculates the phase triplet.

    Args:
        FG (np.ndarray): Structure factor (complex) of primary reflection.
        FH (np.ndarray): List of structure factors (complex) for secondary reflection(s).
        FGH (np.ndarray): List of structure factors (complex) of coupling reflections(s).

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: Cosine of phase triplet, phase triplet (radians) and W.

    Notes:
         For unknown structure factors, use the `triplet_relation` method from `crystal` module.
    """

    W = (FH * FGH) / FG

    return np.real(W) / np.absolute(W), np.angle(W), np.absolute(W)
