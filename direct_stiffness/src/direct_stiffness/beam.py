"""Module with a class for beam elements."""
import warnings

import numpy as np

from direct_stiffness import utils
from direct_stiffness.cross_section import CrossSection

__all__ = ["BeamElement"]


class BeamElement():
    """A flexible beam element.

    Parameters
    ----------
    cross_section : CrossSection
        The beam cross section.
    E : int | float | np.number
        Young's modulus [N/m^2 = Pa].
    G : int | float | np.number, optional
        Shear modulus [N/m^2 = Pa]. The default is None.
    nu : int | float | np.number, optional
        Poisson's ratio. The default is None.
    idx : Sequence, optional
        Global indices of the start and end node of the beam.\n
        The default is (1, 2).
    coords : ArrayLike, optional
        Cartesian coordinates of the start and end node of the beam,
        specified in the global coordinate system.\n
        The default is ((0, 0, 0), (1, 0, 0)).

    Raises
    ------
    TypeError
        - If the cross_section is not an instance of the CrossSection
          class.
        - If idx is not a Sequence.
        - If the elements of idx are not integer values.
    ValueError
        - If the Young's modulus is not a positive, non-zero scalar numeric
          value.
        - If the Poisson's ratio and the shear ratio are both not a
          positive, non-zero scalar numeric value.
        - If the elements of idx are not positive & non-zero.
        - If the coords are not of shape (2, 3) or (3, 2). I.e. if they
          don't represent 3D cartesion coordinates.
        - If the start and end node have the same coordinates.
    """

    def __init__(self, cross_section, E, G=None, nu=None, idx=(1, 2),
                 coords=((0, 0, 0), (1, 0, 0))):
        """Create a flexible beam element.

        Parameters
        ----------
        cross_section : CrossSection
            The beam cross section.
        E : int | float | np.number
            Young's modulus [N/m^2 = Pa].
        G : int | float | np.number, optional
            Shear modulus [N/m^2 = Pa]. The default is None.
        nu : int | float | np.number, optional
            Poisson's ratio. The default is None.
        idx : Sequence, optional
            Global indices of the start and end node of the beam.\n
            The default is (1, 2).
        coords : ArrayLike, optional
            Cartesian coordinates of the start and end node of the beam,
            specified in the global coordinate system.\n
            The default is ((0, 0, 0), (1, 0, 0)).

        Raises
        ------
        TypeError
            - If the cross_section is not an instance of the CrossSection
              class.
            - If idx is not a Sequence.
            - If the elements of idx are not integer values.
        ValueError
            - If the Young's modulus is not a positive, non-zero scalar numeric
              value.
            - If the Poisson's ratio and the shear ratio are both not a
              positive, non-zero scalar numeric value.
            - If the elements of idx are not positive & non-zero.
            - If the coords are not of shape (2, 3) or (3, 2). I.e. if they
              don't represent 3D cartesion coordinates.
            - If the start and end node have the same coordinates.
        """
        if not isinstance(cross_section, CrossSection):
            raise TypeError("Cross section must be a CrossSection instance "
                            "from the direct_stiffness.cross_section module")

        if not utils._validate_numeric(E, allow_neg=False, allow_zero=False):
            raise ValueError("Young's modulus must be positive, non-zero "
                             "scalar numeric value.")

        if G is None:
            if not utils._validate_numeric(nu, allow_neg=False,
                                           allow_zero=False):
                raise ValueError("Poisson's ratio must be positive, non-zero "
                                 "scalar numeric value.")

            # Calculate shear modulus assuming isotropic material
            warnings.warn("Shear modulus calculated from young's modulus and "
                          "poisson's ratio assuming ISOTROPIC material.")
            G = E/2/(1+nu)

        if not isinstance(idx, (tuple, list, np.ndarray)):
            raise TypeError("idx must be a tuple, list or numpy array.")
        if not len(idx) == 2:
            raise ValueError("idx must be of length 2.")
        if not all(isinstance(idx_i, (int, np.integer)) for idx_i in idx):
            raise TypeError("All elements of idx must be integers")
        elif not all(idx_i > 0 for idx_i in idx):
            raise ValueError("All elements of idx must be positive and "
                            "non-zero.")

        coords = utils._validate_arraylike_numeric(coords, ndim=2)
        if not coords.shape == (2, 3):
            if coords.shape == (3, 2):
                coords = coords.T
            else:
                raise ValueError("coords must be of shape (2, 3) or (3, 2).")
        L = np.linalg.norm(np.diff(coords, axis=0))

        if L == 0:
            raise ValueError("Start and end node are identical.")

        self._coords = coords
        self._L = L
        self._cross_section = cross_section
        self._E = E
        self._G = G
        self._idx = tuple(idx)

    @property
    def cross_section(self):
        """CrossSection : The beam cross section."""
        return self._cross_section

    @property
    def coords(self):
        """np.ndarray : The beam node coordinates."""
        return self._coords

    @property
    def L(self):
        """int | float | np.number : The beam length."""
        return self._L

    @property
    def E(self):
        """int | float | np.number : The beam's young's modulus."""
        return self._E

    @property
    def G(self):
        """int | float | np.number : The beam's shear modulus."""
        return self._G

    @property
    def idx(self):
        """tuple : The beam node indices."""
        return self._idx

    def local_stiffness_matrix(self, is_3d=True):
        """Compute the local stiffness matrix.

        Compute the stiffness matrix of the beam element in the local
        coordinate system (x-axis defined in the direction from node 1 to
        node 2).

        Parameters
        ----------
        is_3d : bool, optional
            If True, return the 12x12 3D beam stiffness matrix.
            If False, return the 6x6 2D beam stiffness matrix.

        Raises
        ------
        TypeError
            If is_3d is not a boolean.

        Returns
        -------
        K : np.ndarray
            Local stiffness matrix for the beam element.
        """
        if not isinstance(is_3d, bool):
            raise TypeError("is_3d must be boolean.")

        A = self.cross_section.area
        Iy = self.cross_section.iy
        Iz = self.cross_section.iz
        J = self.cross_section.j
        L = self.L
        E = self.E
        G = self.G

        if is_3d:
            # 12x12 stiffness matrix for 3D beam (axial + torsion + bending)
            K = np.zeros((12, 12))

            # Axial terms
            k_axial = E * A / L
            K[0, 0] = K[6, 6] = k_axial
            K[0, 6] = K[6, 0] = -k_axial

            # Torsion terms
            k_torsion = G * J / L
            K[3, 3] = K[9, 9] = k_torsion
            K[3, 9] = K[9, 3] = -k_torsion

            # Bending about z-axis (Iz)
            k_flex_z1 = E * Iz / L
            k_flex_z2 = E * Iz / (L**2)
            k_flex_z3 = E * Iz / (L**3)
            K[1, 1] = K[7, 7] = 12 * k_flex_z3
            K[1, 7] = K[7, 1] = -12 * k_flex_z3
            K[1, 5] = K[5, 1] = 6 * k_flex_z2
            K[1, 11] = K[11, 1] = 6 * k_flex_z2
            K[5, 7] = K[7, 5] = -6 * k_flex_z2
            K[7, 11] = K[11, 7] = -6 * k_flex_z2
            K[5, 5] = 4 * k_flex_z1
            K[5, 11] = K[11, 5] = 2 * k_flex_z1
            K[11, 11] = 4 * k_flex_z1

            # Bending about y-axis (Iy)
            k_flex_y1 = E * Iy / L
            k_flex_y2 = E * Iy / (L**2)
            k_flex_y3 = E * Iy / (L**3)
            K[2, 2] = K[8, 8] = 12 * k_flex_y3
            K[2, 8] = K[8, 2] = -12 * k_flex_y3
            K[4, 8] = K[8, 4] = 6 * k_flex_y2
            K[8, 10] = K[10, 8] = 6 * k_flex_y2
            K[2, 4] = K[4, 2] = -6 * k_flex_y2
            K[2, 10] = K[10, 2] = -6 * k_flex_y2
            K[4, 4] = 4 * k_flex_y1
            K[4, 10] = K[10, 4] = 2 * k_flex_y1
            K[10, 10] = 4 * k_flex_y1

        else:
            # 6x6 stiffness matrix for 2D beam (axial + bending)
            K = np.zeros((6, 6))

            k_axial = E * A / L
            k_flex_1 = E * Iz / L
            k_flex_2 = E * Iz / (L**2)
            k_flex_3 = E * Iz / (L**3)

            # Axial
            K[0, 0] = K[3, 3] = k_axial
            K[0, 3] = K[3, 0] = -k_axial

            # Bending
            K[1, 1] = K[4, 4] = 12 * k_flex_3
            K[1, 4] = K[4, 1] = -12 * k_flex_3
            K[1, 2] = K[2, 1] = 6 * k_flex_2
            K[1, 5] = K[5, 1] = 6 * k_flex_2
            K[2, 4] = K[4, 2] = -6 * k_flex_2
            K[4, 5] = K[5, 4] = -6 * k_flex_2
            K[2, 2] = K[5, 5] = 4 * k_flex_1
            K[2, 5] = K[5, 2] = 2 * k_flex_1

        return K

    def global_stiffness_matrix(self, is_3d=True):
        """Compute the global stiffness matrix.

        Compute the stiffness matrix of the beam element in the global
        coordinate system).

        Parameters
        ----------
        is_3d : bool, optional
            If True, return the 12x12 3D beam stiffness matrix.
            If False, return the 6x6 2D beam stiffness matrix.

        Returns
        -------
        K : np.ndarray
            Global stiffness matrix for the beam element.
        """

        K_local = self.local_stiffness_matrix(is_3d=is_3d)
        T = self.rotation_matrix(is_3d=is_3d)

        return T @ K_local @ T.T

    def rotation_matrix(self, is_3d=True):
        """Compute the rotational transformation matrix.

        Parameters
        ----------
        is_3d : bool, optional
            If True, compute 12x12 3D rotation matrix.\n
            If False, compute 6x6 2D rotation matrix.\n
            The default is True.

        Raises
        ------
        TypeError
            If is_3d is not a boolean.

        Returns
        -------
        T : np.ndarray
            Rotational transformation matrix for transforming element stiffness
            to global coordinates.
        """
        if not isinstance(is_3d, bool):
            raise TypeError("is_3d must be boolean.")

        delta = np.diff(self.coords, axis=0).flatten()
        lx, ly, lz = delta / self.L
        local_x = np.array([lx, ly, lz])

        # Find local y and z directions using arbitrary reference vector
        ref = np.array([0, 0, 1])
        if np.isclose(abs(np.dot(local_x, ref)), 1.0):
            # local_x is nearly parallel to ref → choose another reference
            # direction
            ref = np.array([0, 1, 0])
        local_y = np.cross(ref, local_x)
        local_y = local_y / np.linalg.norm(local_y)
        local_z = np.cross(local_x, local_y)

        # Assemble rotation matrix
        # Note that each vector local_<> represents a column in the rotation
        # matrix. Thus when stacking them as rows, the matrix needs to be
        # transposed afterwards
        R = np.vstack([local_x, local_y, local_z]).T

        # Assemble transformation matrix
        # Note that this is the transformation matrix to convert the local
        # coordinates to the global ones. In the lecture, we used the letter
        # T for the matrix to transform from global to local coordinates
        # instead.
        if is_3d:
            T = np.zeros((12, 12))
            for i in range(4):
                T[i*3:(i+1)*3, i*3:(i+1)*3] = R
        else:
            T = np.zeros((6, 6))
            for i in range(2):
                T[i*3:(i+1)*3, i*3:(i+1)*3] = R

        return T
