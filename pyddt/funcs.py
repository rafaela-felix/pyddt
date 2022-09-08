#!/usr/bin/python3

"""
Functions required by the main classes. Unuseful for external users.
"""

from functools import reduce

import numpy as np
import pandas as pd
import plotly.graph_objects as go

__author__ = "Rafaela Felix"
__credits__ = {"Rafaela Felix", "Sergio Morelhão"}
__version__ = "1.0"
__maintainer__ = "Rafaela Felix"
__email__ = "rafaelafelixp@usp.br"
__status__ = "Production"


def delete_multiple(*args: np.ndarray or list[np.ndarray], idx: list or np.ndarray or int) -> list[np.ndarray]:

    """Deletes the same indices of multiple arrays.

    Args:
        args (list[np.ndarray]): Arrays.
        idx (list or np.ndarray or int): Indices.

    Returns:
         ndarray: New array after deleting the indicated indices.

    Usage:
        * ``delete_multiple(arr1, arr2, arr3, arr4, ...,  idx=0)``
        * ``delete_multiple(arr1, arr2, arr3, arr4, ...,  idx=[0, 1, 2])``
        * ``delete_multiple(arr1, arr2, ...,  idx=np.arange(0, 100))``

    Notes:
        Call this function explicitly using ``idx=``.
    """

    arr = []
    for arg in args:
        arr.append(np.delete(arg, idx, axis=0))
    return [comp for comp in arr]


def search_reflection(HKL: np.ndarray, G: np.ndarray) -> int or list:

    """Finds the primary reflection.

    Args:
        HKL (np.ndarray): Array of Miller indices.
        G (np.ndarray): Miller indices of the primary reflection (eg `[1, 0, 0]`).

    Returns:
        int or list: Position of the primary reflection in the `HKL` array.
    """

    x = np.where(HKL[:, 0] == G[0])
    y = np.where(HKL[:, 1] == G[1])
    z = np.where(HKL[:, 2] == G[2])

    m = reduce(np.intersect1d, (x, y, z))

    if len(m) == 0:
        return []
    else:
        return int(m[0])


def coupling_reflection(F: np.ndarray, H: np.ndarray, G: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                                                              np.ndarray, list[int]]:

    """Defines the coupling reflections and finds the corresponding structure factors.

    Args:
        F (np.ndarray): Complex structure factors of the secondary reflections.
        H (np.ndarray): Secondary reflections.
        G (np.ndarray): Primary reflection.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int]]: Secondary reflections, structure factors of
        secondary reflections, coupling reflections, structure factors of coupling reflections, index of not found
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


def comparison_plot(dataframe: pd.DataFrame, zlabel: str, xlabel: str,
                    color: str = 'darkred'):

    """Comparison of phase (three-dimensional plot).

    Args:
        dataframe (pd.DataFrame): Dataframe with x, y and z coordinates labeled as F, Q and Z.
        zlabel (str): Title for Z axes (phase difference).
        xlabel (str): Title for X axes (scattering amplitude - F or W).
        color (str): Color of markers. **Default**: darkred.
    """

    x = np.array(dataframe['F'], dtype=float)
    y = np.array(dataframe['Q'], dtype=float)
    z = np.array(dataframe['Z'], dtype=float)

    layout = go.Layout(scene=dict(xaxis=dict(title=xlabel),
                                  yaxis=dict(title='Q (Å)'),
                                  zaxis=dict(title=zlabel),
                                  ), template='simple_white')

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
    return fig


def str_append(s: str, text: str) -> str:

    """Appends new text to an existent string.

    Args:
        s (str): Existing string.
        text (str): New text.

    Returns:
        str: Concatenation of `s` and `text`.
    """

    return s + text
