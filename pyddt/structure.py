#!/usr/bin/python3

"""**Structural modelling from *.in* files.**"""

import re

import time
from difflib import SequenceMatcher
from pathlib import Path

import nglview as nv
import numpy as np
from ase import Atoms

__author__ = "Rafaela Felix"
__credits__ = {"Rafaela Felix", "Sergio Morelhão"}
__version__ = "1.0"
__maintainer__ = "Rafaela Felix"
__email__ = "rafaelafelixp@usp.br"
__status__ = "Production"


class Structure:

    def __init__(self, fname: str):

        """Given a *.in* file, returns a new `struc` object.

        Args:
            fname (str): Structure file name.
        """

        self.name = fname
        self.atoms = np.loadtxt(fname, usecols=0, dtype='<U32', skiprows=1)
        self.positions = np.loadtxt(fname, usecols=(1, 2, 3), skiprows=1)
        self.occupancy = np.loadtxt(fname, usecols=4, skiprows=1)
        self.bfactors = np.loadtxt(fname, usecols=5, skiprows=1)
        self.lattice = np.loadtxt(fname, max_rows=1)

    def visualizer_in(self) -> nv.widget.NGLWidget:

        """Visualizes the structure.

        Returns:
            nv.widget.NGLWidget: Interactive visualization of the  conventional unit cell.

        Notes:
            * Available exclusively in Jupyter Notebooks.
            * ATTENTION: We found bugs in this method, so it's highly recommended to assign it to a variable
                (x = Structure.visualize_in()) and close the figure after the visualization (x.close()).
        """

        at = np.copy(self.atoms)  # just atoms (ions don't contribute to resonance)
        idx = np.where(np.char.isalpha(self.atoms) == False)
        if not np.shape(idx):
            pass
        else:
            ion = np.copy(self.atoms[idx])
            regex = re.compile('[^a-zA-Z]')

            for i in range(len(ion)):
                at[idx[0][i]] = regex.sub('', ion[i])

        system = Atoms(positions=self.positions * self.lattice[:3], symbols=at, cell=self.lattice)

        view = nv.show_ase(system)
        view.add_representation('point', selection='all', radius='0.6')
        view.add_label(color='black', scale=0.8, labelType='text',
                       labelText=[str(i) for i in range(len(system))],
                       zOffset=2.0, attachment='middle_center')
        view.add_unitcell()
        return view

    def replace_ion(self, old: str, new: str):

        """Replaces atoms or ions.

        Args:
            old (str): Array indices or element symbol to be replaced.
            new (str): New atom/ion symbol.

        Notes:
            * There are many ways to pass indices as arguments: a number, numbers separated by commas, an interval
            of indices (first and last indices separated by ":" ) and intervals separated by commas. A string is
            expected.
            * If the new symbol isn't in the available CromerMann list, the symbol is replaced by the most similar and a
            warning will be displayed.

        Usage:
            * `replace_ion('Ce', 'Ce3+')`
            * `replace_ion('1', 'O1-')`
            * `replace_ion('1, 2, 3', 'O1-')`
            * `replace_ion('1:3', 'Fe2+')`
            * `replace_ion('1:3, 5:6, 7:8', 'N3-')`
        """
        crom = Path(__file__).parent / "f0_CromerMann.txt"

        el = np.loadtxt(crom, dtype='str', usecols=0)
        rank = np.array([SequenceMatcher(None, new, el[i]).ratio() for i in range(len(el))])

        if rank.max() != 1:
            print(new, 'is not included in Cromermann factors. Replaced by', el[rank.argmax()])  # future log warning
            new = el[rank.argmax()]

        if old.count(':') == 0:
            if len(old.split(',')) > 1:
                idx = [int(i) for i in old.split(',')]
                self.atoms[idx] = new
            else:
                if old.isnumeric():
                    self.atoms[int(old)] = new
                else:
                    idx = np.where(self.atoms == old)[0]
                    self.atoms[idx] = np.char.replace(self.atoms[idx], old, new)
        else:
            index = old.split(',')
            idx = [(int(i.split(':')[0]), int(i.split(':')[1]) + 1) for i in index]
            for i in range(len(idx)):
                m, n = idx[i]
                self.atoms[m:n] = new

    def replace_occupancy(self, index: str, value: str):

        """Replaces a specific set of occupancy numbers.

        Args:
            index (str): Array indices of the occupancy numbers to be replaced.
            value (str): New values.

        Notes:
            There are many ways to pass indices as arguments: a number, numbers separated by commas, an interval
            of indices (first and last indices separated by ":" ) and intervals separated by commas. A string is
            expected.

        Usage:
            * `replace_occupancy('1', 0.78)`
            * `replace_occupancy('1, 2, 3', 0.92)`
            * `replace_occupancy('1:3', 0)`
            * `replace_occupancy('1:3, 5:6, 7:8', 0.5)`
        """

        if index.count(':') == 0:
            if len(index.split(',')) > 1:
                idx = [int(i) for i in index.split(',')]
                self.occupancy[idx] = value
            else:
                self.occupancy[int(index)] = value
        else:
            ind = index.split(',')
            idx = [(int(i.split(':')[0]), int(i.split(':')[1]) + 1) for i in ind]
            for i in range(len(idx)):
                m, n = idx[i]
                self.occupancy[m:n] = value

    def replace_bfactor(self, index: str, value: str):

        """Replaces a specific set of B-factors.

        Args:
            index (str): Array indices of B-factors to be replaced.
            value (str): New values.

        Notes:
            There are many ways to pass indices as arguments: only a number, numbers separated by commas, an interval
            of indices (first and last indices separated by ":" ) and intervals separated by commas. A string is
            expected.

        Usage:
            * `replace_bfactor('1', 3.14)`
            * `replace_bfactor('1, 2, 3', 0.92)`
            * `replace_bfactor('1:3', 10.015)`
            * `replace_bfactor('1:3, 5:6, 7:8', 0.5)`
        """

        if index == ':':
            self.bfactors[:] = value
        elif index.count(':') == 0:
            if len(index.split(',')) > 1:
                idx = [int(i) for i in index.split(',')]
                self.bfactors[idx] = value
            else:
                self.bfactors[int(index)] = value
        else:
            ind = index.split(',')
            idx = [(int(i.split(':')[0]), int(i.split(':')[1]) + 1) for i in ind]
            for i in range(len(idx)):
                m, n = idx[i]
                self.bfactors[m:n] = value

    def append_atom(self, info: list):

        """Appends a new atom to the structure.

        Args:
            info (list): Atom or ion symbol, fractional coordinates (x, y and z), occupancy number and B-factor.

        Usage:
            * `append_atom(['Fe2+', 0.5, 0.5, 0.5, 1, 3.14])`

        Notes:
            * The new atom will be allocated in the last index of the atoms array.
            * For the current version, append atoms one-by-one.
        """
        crom = Path(__file__).parent / "f0_CromerMann.txt"

        el = np.loadtxt(crom,  dtype='str', usecols=0)
        rank = np.array([SequenceMatcher(None, info[0], el[i]).ratio() for i in range(len(el))])

        if rank.max() != 1:
            print(info[0], 'is not included in Cromermann factors. Replaced by', el[rank.argmax()])  # log warning
            info[0] = el[rank.argmax()]

        self.atoms = np.append(self.atoms, info[0])
        self.positions = np.row_stack((self.positions, info[1:4]))
        self.occupancy = np.append(self.occupancy, info[4])
        self.bfactors = np.append(self.bfactors, info[5])

    def delete_atom(self, index: str):

        """Deletes a set of atoms.

        Args:
            index (str): Array indices of the atoms to be deleted.

        Notes:
            There are many ways to pass indices as arguments: only a number, numbers separated by commas, an interval
            of indices (first and last indices separated by ":" ) and intervals separated by commas.  A string is
            expected.

        Usage:
            * `delete_atom('1')`
            * `delete_atom('1, 2, 3')`
            * `delete_atom('1:3')`
            * `delete_atom('1:3, 5:6, 7:8')`
        """

        if index.count(':') == 0:
            if len(index.split(',')) > 1:
                idx = [int(i) for i in index.split(',')]
            else:
                idx = int(index)
            self.atoms = np.delete(self.atoms, idx, axis=0)
            self.positions = np.delete(self.positions, idx, axis=0)
            self.occupancy = np.delete(self.occupancy, idx, axis=0)
            self.bfactors = np.delete(self.bfactors, idx, axis=0)
        else:
            ind = index.split(',')
            idx = [(int(i.split(':')[0]), int(i.split(':')[1]) + 1) for i in ind]
            for i in range(len(idx)):
                m, n = idx[i]
                ind = np.arange(m, n)
                self.atoms = np.delete(self.atoms, ind, axis=0)
                self.positions = np.delete(self.positions, ind, axis=0)
                self.occupancy = np.delete(self.occupancy, ind, axis=0)
                self.bfactors = np.delete(self.bfactors, ind, axis=0)

    def save_infile(self, fout=''):

        """Saves *.in* file.

        Args:
            fout (str): Filename. **Default**: `datetime_fname`
        """

        fst_row = f"{self.lattice[0]:10.4f} {self.lattice[1]:10.4f} {self.lattice[2]:10.4f} " \
                  f"{self.lattice[3]:10.4f} {self.lattice[4]:10.4f} {self.lattice[5]:10.4f}\n"

        if fout == '':
            fout = time.strftime("%Y%m-%H_%M_%S") + '_' + self.name

        with open(fout, "w") as f:
            f.write(fst_row)
            for i in range(len(self.atoms)):
                text = f"{self.atoms[i]:>10s} {self.positions[i][0]:10.4f} " \
                       f"{self.positions[i][1]:10.4f} {self.positions[i][2]:10.4f} " \
                       f"{self.occupancy[i]:10.4f} {self.bfactors[i]:10.4f}\n"
                f.write(text)
