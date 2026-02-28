import numpy as np

from direct_stiffness.beam import BeamElement
from direct_stiffness import utils

class Frame:
    def __init__(self, beams):
        if not isinstance(beams, (tuple, list, dict)):
            raise TypeError("beams must be a tuple, list or dict.")
        elif isinstance(beams, dict):
            beams = beams.values()

        if not all(isinstance(beam, BeamElement) for beam in beams):
            raise TypeError("Elements of beams must be BeamElement instances.")

        # Get indices of nodes
        node_idx = np.array([beam.idx for beam in beams])
        node_idx.flags.writeable = False  # Read-only

        self._beams = tuple(beams)
        self._node_idx = node_idx
        self._n_nodes = len(np.unique(node_idx))

        self._dof_2d = [[f"u_x{i}", f"u_z{i}", f"theta_y{i}"]
                        for i in range(1, self._n_nodes+1)]
        self._dof_2d = tuple([dof for node_dofs in self._dof_2d
                              for dof in node_dofs])

        self._dof_3d = [[f"u_x{i}", f"u_y{i}", f"u_z{i}",
                         f"theta_x{i}", f"theta_y{i}", f"theta_z{i}"]
                        for i in range(1, self._n_nodes+1)]
        self._dof_3d = tuple([dof for node_dofs in self._dof_3d
                              for dof in node_dofs])

        self._idx_dof_2d = {key: value for key, value
                            in zip(self._dof_2d, range(len(self._dof_2d)))}
        self._idx_dof_3d = {key: value for key, value
                            in zip(self._dof_3d, range(len(self._dof_3d)))}

    @property
    def beams(self):
        """tuple : The beams from which the frame is constructed."""
        return self._beams

    @property
    def node_idx(self):
        """np.ndarray : Indices of the nodes of the beams of the frame."""
        return self._node_idx

    @property
    def n_nodes(self):
        """int : Number of nodes of the frame."""
        return self._n_nodes

    @property
    def n_dof_2d(self):
        """int : Number of degrees of freedom of the frame in 2d."""
        return self._n_nodes * 3

    @property
    def n_dof_3d(self):
        """int : Number of degrees of freedom of the frame in 2d."""
        return self._n_nodes * 6

    @property
    def dof_2d(self):
        """tuple : list of degrees of freedom of the frame in 2d."""
        return self._dof_2d

    @property
    def dof_3d(self):
        """tuple : list of degrees of freedom of the frame in 3d."""
        return self._dof_3d

    @property
    def idx_dof_2d(self):
        """dict : Indices of the degrees of freedom of the frame in the
        stifness matrix."""
        return self._idx_dof_2d

    @property
    def idx_dof_3d(self):
        """dict : Indices of the degrees of freedom of the frame in the
        stifness matrix."""
        return self._idx_dof_3d

    def stiffness_matrix(self, is_3d=True):
        """Compute the stiffness matrix for a frame system.

        Parameters
        ----------
        is_3d : bool, optional
            If True, return the 12x12 3D frame stiffness matrix.
            If False, return the 6x6 2D frame stiffness matrix.

        Returns
        -------
        K : np.ndarray
            Global stiffness matrix for the frame system.
        """
        n_dof_beam = 12 if is_3d else 6  # Number of dof per beam
        n_dof_node = int(n_dof_beam/2)  # Number of dof per node
        K = np.zeros([n_dof_node * self.n_nodes] * 2)

        for i in range(len(self.beams)):
            K_beam = self.beams[i].global_stiffness_matrix(is_3d=is_3d)

            idx_1, idx_2 = self.node_idx[i]-1

            # Fill in the frame stiffness matrix from the beam stiffness matrix
            # quadrant by quadrant
            # Upper left quadrant:
            K[idx_1*n_dof_node:(idx_1+1)*n_dof_node,
              idx_1*n_dof_node:(idx_1+1)*n_dof_node] = \
                K[idx_1*n_dof_node:(idx_1+1)*n_dof_node,
                  idx_1*n_dof_node:(idx_1+1)*n_dof_node] \
                + K_beam[0:n_dof_node, 0:n_dof_node]

            # Lower right quadrant:
            K[idx_2*n_dof_node:(idx_2+1)*n_dof_node,
              idx_2*n_dof_node:(idx_2+1)*n_dof_node] = \
                K[idx_2*n_dof_node:(idx_2+1)*n_dof_node,
                  idx_2*n_dof_node:(idx_2+1)*n_dof_node] \
                + K_beam[n_dof_node:n_dof_beam, n_dof_node:n_dof_beam]

            # Upper right quadrant:
            K[idx_1*n_dof_node:(idx_1+1)*n_dof_node,
              idx_2*n_dof_node:(idx_2+1)*n_dof_node] = \
                K[idx_1*n_dof_node:(idx_1+1)*n_dof_node,
                  idx_2*n_dof_node:(idx_2+1)*n_dof_node] \
                + K_beam[0:n_dof_node, n_dof_node:n_dof_beam]

            # Lower left quadrant:
            K[idx_2*n_dof_node:(idx_2+1)*n_dof_node,
              idx_1*n_dof_node:(idx_1+1)*n_dof_node] = \
                K[idx_2*n_dof_node:(idx_2+1)*n_dof_node,
                  idx_1*n_dof_node:(idx_1+1)*n_dof_node] \
                + K_beam[n_dof_node:n_dof_beam, 0:n_dof_node]

        return K

    import numpy as np

    def solve_fea_system(self, F, fixed_dofs, u_presc=None, is_3d=True):
        """Solve KU = F for nodal displacements with given boundary conditions.

        Parameters
        ----------
        F : np.ndarray
            Global force vector (n,).
        fixed_dofs : tuple[int, str] | list[int, str]
            Indices of degrees of freedom (DOFs) that are fixed (zero
            displacement).
        u_presc : tuple | list | dict[int, float], optional
            DOFs with prescribed (nonzero) displacements.

        Returns
        -------
        U : np.ndarray
            Global displacement vector (n,).
        R : np.ndarray
            Global reaction forces (n,).
        """
        n_dof_frame = self.n_dof_3d if is_3d else self.n_dof_2d
        idx_dof = self.idx_dof_3d if is_3d else self.idx_dof_2d

        if not isinstance(is_3d, bool):
            raise TypeError("is_3d must be boolean.")

        # Check fixed dofs
        if not isinstance(fixed_dofs, (tuple, list, np.ndarray)):
            raise TypeError("Fixed dofs must be a tuple, list or numpy array.")

        if isinstance(fixed_dofs, np.ndarray):
            fixed_dofs = fixed_dofs.flatten()
        fixed_dofs = list(fixed_dofs)  # Ensure mutability

        for i in range(len(fixed_dofs)):
            fixed_dof = fixed_dofs[i]
            if isinstance(fixed_dof, (int, np.integer)):
                if fixed_dof < 0 or fixed_dof >= n_dof_frame:
                    raise ValueError(f"Fixed dof '{fixed_dof}' is not a valid "
                                     "index.")
            elif isinstance(fixed_dof, str):
                idx_dof_i = idx_dof.get(fixed_dof)
                if idx_dof_i is None:
                    raise ValueError(f"Fixed dof '{fixed_dof}' unknown.")

                fixed_dofs[i] = idx_dof_i
            else:
                raise TypeError("Fixed dofs must consist of integers or "
                                "strings.")

        if len(set(fixed_dofs)) < len(fixed_dofs):
            raise ValueError("Fixed dofs contains duplicates.")

        # Check prescribed dofs
        U = np.zeros(n_dof_frame)
        if u_presc is not None:
            if isinstance(u_presc, (tuple, list, np.ndarray)):
                if not len(u_presc) == n_dof_frame:
                    raise ValueError("Length of u_presc must match number of "
                                     "dofs of the system or be a dict.")

                if isinstance(u_presc, np.ndarray):
                    u_presc = u_presc.flatten()
                u_presc = utils._validate_arraylike_numeric(
                    u_presc.values(), name="u_presc", ndim=1)

                mask = np.isfinite(u_presc)
                U[mask] = u_presc[mask]
            elif isinstance(u_presc, (dict)):
                if not set(u_presc.keys()).issubset(idx_dof.keys()):
                    raise KeyError("Unknown dofs found in u_prec")

                utils._validate_arraylike_numeric(u_presc.values(),
                                                  name="u_presc")

                for dof, val in u_presc.items():
                    idx_dof_i = idx_dof.get(fixed_dof)
                    # if idx_dof_i is None:
                    #     raise ValueError("Prescribed dof "
                    #                      f"'{u_presc}' unknown.")

                    # if not _validate_numeric(val):
                    #     raise ValueError("Invalid value for prescribed dof "
                    #                      f"'{u_presc}'. Must be "
                    #                      "a scalar, finite numeric value")

                    U[idx_dof_i] = val
            else:
                raise TypeError("Fixed dofs must be None, a tuple, list, "
                                "numpy array or dict.")

        K = self.stiffness_matrix(is_3d=is_3d)

        # Determine free DOFs (those not fixed)
        free_dofs = np.setdiff1d(np.arange(n_dof_frame), fixed_dofs)

        # Partition matrices/vectors
        K_ff = K[np.ix_(free_dofs, free_dofs)]
        F_f = F[free_dofs]

        # Adjust F_f for prescribed displacements (if any)
        if u_presc:
            K_fc = K[np.ix_(free_dofs, fixed_dofs)]
            F_f = F_f - K_fc @ U[fixed_dofs]

        # Solve for free displacements
        U[free_dofs] = np.linalg.solve(K_ff, F_f)

        # Compute reaction forces
        R = K @ U - F

        return U, R
