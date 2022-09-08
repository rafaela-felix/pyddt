#!/usr/bin/python3

"""
This module converts a Crystallographic Information File (CIF) into a structure file (*.in*).
"""

import nglview as nv
import numpy as np
import pymatgen.symmetry.analyzer as sy
from pymatgen.io.cif import CifParser
from .funcs import str_append

__author__ = "Rafaela Felix"
__credits__ = {"Rafaela Felix", "Sergio Morelhão"}
__version__ = "1.0"
__maintainer__ = "Rafaela Felix"
__email__ = "rafaelafelixp@usp.br"
__status__ = "Production"


def to_in(fcif: str):

    """Generates *.in* file from CIF.

    Args:
        fcif (str): CIF filename.
    """

    fin = fcif[:-4] + '.in'
    parser = CifParser(fcif)
    structure = parser.get_structures(primitive=False)[0]

    fst_row = f'{structure.lattice.abc[0]:10.4f} {structure.lattice.abc[1]:10.4f} {structure.lattice.abc[2]:10.4f} ' \
              f'{structure.lattice.angles[0]:10.4f} {structure.lattice.angles[1]:10.4f} ' \
              f'{structure.lattice.angles[2]:10.4f}\n'

    with open(fin, "w") as f:
        f.write(fst_row)
        for i in range(len(structure.species)):
            text = f"{structure.species[i].symbol:>10s} {structure.frac_coords[i][0]:10.4f} " \
                   f"{structure.frac_coords[i][1]:10.4f} {structure.frac_coords[i][2]:10.4f} " \
                   f"{1.000:10.4f} {1.000:10.4f}\n"
            f.write(text)

    to_struc(fcif)


def to_struc(fcif: str):

    """Saves the *.struc* file.

    Args:
        fcif (str): CIF filename.

    Notes:

        * The *.struc* file presents some structural and electronic properties which might be useful for structural
          modelling.
        * Guesses of ionic charge (based on `pymatgen.core.composition_oxi_state_guesses`_) are presented for
          inorganic crystals. If organic, the indices of the equivalent atoms are shown.

    Attention:
        The number of asymmetric units probably isn't correct for hydrated crystals.

    .. _pymatgen.core.composition_oxi_state_guesses:
        https://pymatgen.org/pymatgen.core.composition.html

    """

    parser = CifParser(fcif)
    structure = parser.get_structures(primitive=False)[0]

    fname = fcif[:-4] + '.struc'

    text = ''
    text = str_append(text, fcif + ' to ' + fcif[:-4] + '.in\n\n')
    text = str_append(text, str(structure.composition.reduced_formula))
    text = str_append(text, ' (conventional cell = ' + str(structure.composition.get_reduced_formula_and_factor()[1]))
    text = str_append(text, ' molecules, ' + str(int(structure.composition.num_atoms)) + ' atoms)\n\n')
    text = str_append(text, 'Space group: ' + structure.get_space_group_info()[0] + ' (' + str(
        structure.get_space_group_info()[1]) + ')\n')
    text = str_append(text, 'Total charge = ' + str(structure.charge))
    text = str_append(text, '\nElectronic structure:\n')

    spec = structure.species
    lst = []
    nam = []
    for sp in spec:
        lst.append(sp.electronic_structure)
        nam.append(sp.symbol)

    arr, nam = np.array(lst), np.array(nam)

    _, idx = np.unique(nam, return_index=True)

    for i in range(len(idx)):
        text = str_append(text, '\t' + nam[idx][i] + ' - ' + arr[idx][i] + '\n')

    text = str_append(text, '\nIndices:\n')
    s = structure.symbol_set
    for a in s:
        text = str_append(text, str(a) + ' - ' + str(structure.indices_from_symbol(a))[1:-1] + '\n')

    sym = sy.SpacegroupAnalyzer(structure).get_symmetrized_structure()
    equil = np.array(sym.equivalent_indices, dtype=object)
    try:  # molecular crystal
        x = [np.column_stack((sym.species[equil[i, 0]].symbol, *equil[i, :])) for i in range(len(equil))]
        text = str_append(text, '\nEquivalent indices:\n')
        molc = True
    except IndexError:  # inorganic crystal
        x = []
        molc = False

    for i in range(len(x)):
        text = str_append(text, ' '.join(x[i][0]) + '\n')

    oxi = structure.composition.oxi_state_guesses()[:15]
    lst = []
    for i in range(len(oxi)):
        lst.append([k + ':' + str(np.round(v, 1)) + '\t' for k, v in oxi[i].items()])

    if not molc:
        text = str_append(text, '\nPossible oxidation states:\n')
        lst = [''.join(lst[i]) for i in range(len(lst))]
        for i in range(len(lst)):
            text = str_append(text, lst[i] + '\n')

        structure.add_oxidation_state_by_guess()

        text = str_append(text, '\nStructure of .in file with charge (by oxidation state guesses):\n')

        text = str_append(text,
                          f"{structure.lattice.abc[0]:2.4f} {structure.lattice.abc[1]:11.4f} "
                          f"{structure.lattice.abc[2]:10.4f} {structure.lattice.angles[0]:10.4f} "
                          f"{structure.lattice.angles[1]:10.4f} {structure.lattice.angles[2]:10.4f}\n")

        for i in range(len(structure.species)):
            text = str_append(text,
                              f"{structure.species[i].to_pretty_string():>7s} {structure.frac_coords[i][0]:10.4f} "
                              f"{structure.frac_coords[i][1]:10.4f} {structure.frac_coords[i][2]:10.4f} {1.000:10.4f} "
                              f"{1.000:10.4f}\n")

    with open(fname, "w") as f:
        for line in text:
            f.write(line)


def visualizer_cif(fcif: str) -> nv.widget.NGLWidget:

    """Visualizes the CIF structure. This method is available exclusively in Jupyter Notebooks.

    Args:
        fcif (str): CIF filename.

    Returns:
        nv.widget.NGLWidget: Interactive visualization of the conventional unit cell.

    Attention:
        To avoid bugs, it's highly recommended to assign this method to a variable, then close the figure after
        the visualization using ``variable_name.close()``.

    """

    parser = CifParser(fcif)
    structure = parser.get_structures(primitive=False)[0]

    view = nv.show_pymatgen(structure)
    view.add_representation('point', selection='all', radius='0.6')
    view.add_label(color='black', scale=0.8, labelType='text',
                   labelText=[structure.species[i].symbol for i in range(len(structure.species))],
                   zOffset=2.0, attachment='middle_center')
    view.add_unitcell()
    return view
