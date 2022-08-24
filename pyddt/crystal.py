#!/usr/bin/python3

"""**X-ray diffraction calculation for crystalline structures.**"""

import re

import numpy as np

from .funcs import delete_multiple, search_reflection, coupling_reflection
from .scatter import asfQ, aresE
from .crystal_funcs import lattice, save_diffraction
from .structure import Structure

__author__ = "Rafaela Felix"
__credits__ = {"Rafaela Felix", "Sergio Morelhão"}
__version__ = "1.0"
__maintainer__ = "Rafaela Felix"
__email__ = "rafaelafelixp@usp.br"
__status__ = "Production"


class Crystal:

    def __init__(self, struc_obj: str or Structure):

        """Given a structure file, returns a new `crystal` object.

        Args:
            struc_obj (str or Structure): File name or Structure object.

        Notes:
            * **Expected *.in* file**: atom or ion symbols, fractional coordinates, occupancy numbers
            and B-factors in columns. The first line must present the lattice parameters. Use the `cif`
             module to generate this file from CIF.
        """

        if type(struc_obj) == str:
            self.structure = Structure(struc_obj)
        else:
            self.structure = struc_obj

        A, B, C, Ar, Br, Cr = lattice(self.structure.lattice[0], self.structure.lattice[1], self.structure.lattice[2],
                                      self.structure.lattice[3], self.structure.lattice[4], self.structure.lattice[5])

        self.A = A
        self.B = B
        self.C = C
        self.Ar = Ar
        self.Br = Br
        self.Cr = Cr

    def bragg(self, E: float, hkl: list) -> np.ndarray:

        """Determines the interplanar distance and Bragg angle from Miller indices and X-ray energy.

        Args:
            E (float): X-ray energy (eV).
            hkl (list): Miller indices (e.g [1, 0, 0]).

        Returns:
            np.ndarray: Interplanar distance and Bragg angle.
        """

        wl = 12398.5 / E
        d = 2 * np.pi
        d = d / np.linalg.norm(hkl[0] * self.Ar + hkl[1] * self.Br + hkl[2] * self.Cr)
        thB = np.arcsin(wl / (2 * d)) * 180 / np.pi

        return np.array([d, thB])

    def hkl2Q(self, hkl: np.ndarray) -> np.ndarray:

        """Determines the reciprocal vector from Miller indices.

        Args:
            hkl (np.ndarray): Miller indices

        Returns:
            np.ndarray: Q values (1/angstrom).

        Usage:
            * `hkl2Q([[1, 0, 0], [0, 1, 1]])`
            * `hkl2Q([1, 0, 0])`
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

    def Fhkl(self, E: float or np.ndarray, H: np.ndarray) -> np.ndarray:

        """Calculates the complex structure factor.

        Args:
            E (float or ndarray): X-ray energy (eV).
            H (ndarray): Miller indices.

        Returns:
            np.ndarray: List of structure factors (complex).

        Notes:
            This method accepts an array of energy for a specific reflection or
            a fixed energy value jointly an array of reflections.

        Usage:
            * `Fhkl(8048, [[1, 0, 0], [0, 1, 1]])`
            * `Fhkl([8048, 10004], [1, 0, 0])`
        """

        E = np.array(E)
        H = np.array(H)

        sym = np.unique(self.structure.atoms, return_index=False, return_inverse=False,
                        return_counts=True)[0]  # ions or atoms

        at = np.copy(sym)  # just atoms (ions don't contribute to resonance)
        idx = np.where(np.char.isalpha(sym) == False)
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

            idx = np.where(self.structure.atoms == sym[j])

            if np.shape(H) != (3,):

                bf = np.reshape(np.tile(self.structure.bfactors[idx], len(H)),
                                (len(H), len(self.structure.bfactors[idx])))
                x = np.reshape(np.tile(self.structure.positions[idx, 0][0], len(H)),
                               (len(H), len(self.structure.positions[idx, 0][0])))
                y = np.reshape(np.tile(self.structure.positions[idx, 1][0], len(H)),
                               (len(H), len(self.structure.positions[idx, 1][0])))
                z = np.reshape(np.tile(self.structure.positions[idx, 2][0], len(H)),
                               (len(H), len(self.structure.positions[idx, 2][0])))
                occ = np.reshape(np.tile(self.structure.occupancy[idx], len(H)), (len(H),
                                                                                  len(self.structure.occupancy[idx])))

                Ma = np.transpose([(Q / (4 * np.pi)) ** 2]) * bf
                frac = np.transpose([H[:, 0]]) * x + np.transpose([H[:, 1]]) * y
                frac = frac + np.transpose([H[:, 2]]) * z

                F = F + np.sum(np.transpose([fn]) * occ * np.exp(-Ma) * np.exp(2 * np.pi * 1j * frac),
                               axis=1)

            elif np.shape(E) != ():

                bf = np.reshape(np.tile(self.structure.bfactors[idx], len(E)), (len(E),
                                                                                len(self.structure.bfactors[idx])))
                x = np.reshape(np.tile(self.structure.positions[idx, 0][0], len(E)),
                               (len(E), len(self.structure.positions[idx, 0][0])))
                y = np.reshape(np.tile(self.structure.positions[idx, 1][0], len(E)),
                               (len(E), len(self.structure.positions[idx, 1][0])))
                z = np.reshape(np.tile(self.structure.positions[idx, 2][0], len(E)),
                               (len(E), len(self.structure.positions[idx, 2][0])))
                occ = np.reshape(np.tile(self.structure.occupancy[idx], len(E)), (len(E),
                                                                                  len(self.structure.occupancy[idx])))

                Ma = (Q / (4 * np.pi)) ** 2 * bf
                frac = H[0] * x + H[1] * y + H[2] * z

                F = F + np.sum(np.transpose([fn]) * occ * np.exp(-Ma) * np.exp(2 * np.pi * 1j * frac),
                               axis=1)

            else:

                Ma = self.structure.bfactors[idx] * (Q / (4 * np.pi)) ** 2
                frac = H[0] * self.structure.positions[idx, 0] + H[1] * self.structure.positions[idx, 1]
                frac = frac + H[2] * self.structure.positions[idx, 2]

                F = F + np.sum(fn * self.structure.occupancy[idx] * np.exp(-Ma) * np.exp(2 * np.pi * 1j * frac))

        return F

    def hkl_generate(self, E: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        """Determines the Miller indices of all structure-allowed lattice planes.

        Args:
            E (float): X-ray energy (eV).

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: Miller indices, Bragg angles and interplanar distances.
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

    def diffraction(self, E: float, fout: str = '') -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        """Calculates the complex structure factor and interplanar distance of all structure-allowed lattice planes.

        Args:
            E (float): X-ray energy (eV).
            fout (str): Filename for output. **Default**: None - don't save.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: Reflections, structure factors and interplanar distances.

        """

        HKL, th, d = self.hkl_generate(E)

        F = self.Fhkl(E, HKL)

        idx = np.argsort(np.absolute(F) / np.max(np.absolute(F)))[::-1]  # Sorting by intensity
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
            G_H (list): Indices of primary and secondary reflection (in this order).

        Returns:
            tuple[float, float]: Cosine of phase triplet and interference amplitude W.
        """

        G_H = np.array(G_H)
        G_H = np.row_stack((G_H, G_H[0] - G_H[1]))
        W = self.Fhkl(E, G_H)
        W = (W[1] * W[2]) / W[0]

        return np.real(W) / np.absolute(W), np.absolute(W)

    def klines(self, E: float, G: list or np.ndarray, M: list or np.ndarray, Fmin: float, dw: float = 0.1,
               npoints: int = 20):

        """Calculates the Bragg Cones (BC lines) of all reflections.

        Args:
            E (float): X-ray energy (eV).
            G (list or ndarray): Primary reflection indices.
            M (list or ndarray): Reference direction.
            Fmin (float): Cutoff value for W (minimum).
            dw (float): Maximum distance between BC lines and primary reflection.
            npoints (int): Number of points of 2D-cone representation.

        Notes:
            The BC lines are saved in *IN/OUT/THG_G_array_M_array_E_value.dat* files.
            - IN_G_array_M_array_E_value.dat: BC lines entering the Ewald sphere.
            - OUT_G_array_M_array_E_value.dat: BC lines exiting the Ewald sphere.
            - THG_G_array_M_array_E_value.dat: MD positions (in-out and out-in geometries).

        Usage:
            * `klines(8048, [1, 1, 1], [1, 0, 0], 10)`
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

        dom = dw / npoints
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
            NN = np.argwhere(np.abs(cosBH) > 1)
            cosBH[NN] = 1
                        
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
            
            phi1[NN] = None
            phi2[NN] = None

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
