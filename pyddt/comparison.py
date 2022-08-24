#!/usr/bin/python3

"""
**Phase comparison between structures.**
"""

from datetime import datetime
from functools import reduce

import numpy as np
import pandas as pd

from .crystal_funcs import phase_triplet
from .funcs import comparison_plot, delete_multiple, search_reflection, coupling_reflection

__author__ = "Rafaela Felix"
__credits__ = {"Rafaela Felix", "Sergio Morelhão"}
__version__ = "1.0"
__maintainer__ = "Rafaela Felix"
__email__ = "rafaelafelixp@usp.br"
__status__ = "Production"


def phase(data1: np.ndarray, data2: np.ndarray, min: float = 15):

    """Comparison of the structure factor phase with respect to two structural models.

    Args:
        data1 (np.ndarray): List of structure factors for 1st structure.
        data2 (np.ndarray): List of structure factors for 2nd structure.
        min (float): Minimum phase difference (degrees). **Default**: 15.

    Notes:
        * List of structure factors can be obtained by `crystal.Crystal.diffraction()` method.
        * By default, the `plotly.graph_objects` will be displayed in an installed browser.
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

    Z, Q, F, HKL = delete_multiple(Z, Q, F, HKL, idx=np.argwhere(Z < min))
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

    print('Saved in', name)  # logging

    fig = comparison_plot(df, 'Δδ (deg)', '|F|²')
    fig.show()

    return name


def triplet(data1: np.ndarray, data2: np.ndarray, G: list, wmin: float = 5):

    """Comparison of the phase triplet with respect to two structural models for a given primary reflection.

    Args:
        data1 (np.ndarray): List of structure factors for 1st structure.
        data2 (np.ndarray): List of structure factors for 2nd structure.
        G (list): Indices of primary reflection (eg  `[1, 0, 0]`).
        wmin (float): Cutoff for triplet amplitude (% ranging from 0 to 100). **Default**: 5.

    Notes:
        * List of structure factors can be obtained by `crystal.Crystal.diffraction()` method.
        * By default, the `plotly.graph_objects` will be displayed in an installed browser.
        * The method returns `None` for null or absent primary reflection.
    """

    G = np.array(G)
    HKL1, F1, d1 = data1[0], data1[1], data1[2]
    HKL2, F2, d2 = data2[0], data2[1], data2[2]

    idx1 = search_reflection(HKL1, G)
    idx2 = search_reflection(HKL2, G)

    if not idx1:  # logging
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

            cpsi1, psi1, w1 = phase_triplet(FG1, F1[i], FGH1[i])
            cpsi2, psi2, w2 = phase_triplet(FG2, F2[m], FGH2[m])

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

    fig = comparison_plot(df, 'ΔΨ (deg)', 'W (%)', 'blue')
    fig.show()

    return name
