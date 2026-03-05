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

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

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
        """int : Number of degrees of freedom of the frame in 3d."""
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
        """dict : Indices of the degrees of freedom in the stiffness matrix."""
        return self._idx_dof_2d

    @property
    def idx_dof_3d(self):
        """dict : Indices of the degrees of freedom in the stiffness matrix."""
        return self._idx_dof_3d

    # ------------------------------------------------------------------
    # Stiffness matrix assembly
    # ------------------------------------------------------------------

    def stiffness_matrix(self, is_3d=True):
        """Compute the global stiffness matrix for the frame system.

        Parameters
        ----------
        is_3d : bool, optional
            If True, return the 3D stiffness matrix. If False, return 2D.

        Returns
        -------
        K : np.ndarray
            Global stiffness matrix.
        """
        n_dof_beam = 12 if is_3d else 6
        n_dof_node = n_dof_beam // 2
        K = np.zeros([n_dof_node * self.n_nodes] * 2)

        for i, beam in enumerate(self.beams):
            K_beam = beam.global_stiffness_matrix(is_3d=is_3d)
            idx_1, idx_2 = self.node_idx[i] - 1

            s1 = slice(idx_1 * n_dof_node, (idx_1 + 1) * n_dof_node)
            s2 = slice(idx_2 * n_dof_node, (idx_2 + 1) * n_dof_node)
            sA = slice(0, n_dof_node)
            sB = slice(n_dof_node, n_dof_beam)

            K[s1, s1] += K_beam[sA, sA]  # Upper-left
            K[s2, s2] += K_beam[sB, sB]  # Lower-right
            K[s1, s2] += K_beam[sA, sB]  # Upper-right
            K[s2, s1] += K_beam[sB, sA]  # Lower-left

        return K

    # ------------------------------------------------------------------
    # Solver
    # ------------------------------------------------------------------

    def solve_fea_system(self, F, fixed_dofs, u_presc=None, is_3d=True):
        """Solve KU = F for nodal displacements with given boundary conditions.

        Parameters
        ----------
        F : np.ndarray
            Global force vector (n,).
        fixed_dofs : tuple | list | np.ndarray
            DOFs that are fixed (zero displacement). Entries may be integer
            indices or DOF-name strings (e.g. ``"u_z1"``).
        u_presc : dict[str, float] | list | tuple | np.ndarray, optional
            Prescribed (nonzero) displacements. When a dict, keys are DOF
            name strings and values are the prescribed displacement magnitudes.
        is_3d : bool, optional
            Selects 3-D (True) or 2-D (False) formulation.

        Returns
        -------
        U : np.ndarray
            Global displacement vector (n,).
        R : np.ndarray
            Global reaction force vector (n,).
        """
        if not isinstance(is_3d, bool):
            raise TypeError("is_3d must be boolean.")

        n_dof_frame = self.n_dof_3d if is_3d else self.n_dof_2d
        idx_dof = self.idx_dof_3d if is_3d else self.idx_dof_2d

        # --- Resolve fixed_dofs to integer indices -------------------------
        if not isinstance(fixed_dofs, (tuple, list, np.ndarray)):
            raise TypeError("fixed_dofs must be a tuple, list or numpy array.")

        fixed_dofs = list(np.asarray(fixed_dofs).flatten())

        for i, fd in enumerate(fixed_dofs):
            if isinstance(fd, (int, np.integer)):
                if fd < 0 or fd >= n_dof_frame:
                    raise ValueError(f"Fixed dof '{fd}' is out of range.")
            elif isinstance(fd, str):
                idx = idx_dof.get(fd)
                if idx is None:
                    raise ValueError(f"Fixed dof '{fd}' is unknown.")
                fixed_dofs[i] = idx
            else:
                raise TypeError("Fixed dofs must be integers or strings.")

        if len(set(fixed_dofs)) < len(fixed_dofs):
            raise ValueError("fixed_dofs contains duplicates.")

        # --- Handle prescribed displacements --------------------------------
        U = np.zeros(n_dof_frame)

        if u_presc is not None:
            if isinstance(u_presc, dict):
                unknown_keys = set(u_presc.keys()) - set(idx_dof.keys())
                if unknown_keys:
                    raise KeyError(f"Unknown DOFs in u_presc: {unknown_keys}")
                utils._validate_arraylike_numeric(u_presc.values(),
                                                  name="u_presc")
                for dof, val in u_presc.items():          # BUG FIX: was `fixed_dof`
                    U[idx_dof[dof]] = val
            elif isinstance(u_presc, (tuple, list, np.ndarray)):
                u_presc = np.asarray(u_presc).flatten()
                if len(u_presc) != n_dof_frame:
                    raise ValueError("Length of u_presc must match n_dof.")
                mask = np.isfinite(u_presc)
                U[mask] = u_presc[mask]
            else:
                raise TypeError("u_presc must be None, a dict, list, tuple, "
                                "or numpy array.")

        # --- Assemble and partition -----------------------------------------
        K = self.stiffness_matrix(is_3d=is_3d)
        fixed_dofs = np.array(fixed_dofs, dtype=int)
        free_dofs = np.setdiff1d(np.arange(n_dof_frame), fixed_dofs)

        K_ff = K[np.ix_(free_dofs, free_dofs)]
        F_f = F[free_dofs].copy()

        if u_presc is not None:
            K_fc = K[np.ix_(free_dofs, fixed_dofs)]
            F_f -= K_fc @ U[fixed_dofs]

        U[free_dofs] = np.linalg.solve(K_ff, F_f)
        R = K @ U - F

        return U, R

    # ------------------------------------------------------------------
    # Post-processing: internal fields
    # ------------------------------------------------------------------

    def beam_internal_fields(self, U, beam_index, is_3d=False, n_pts=200):
        """Recover shear force, bending moment, curvature, slope and
        deflection along a single beam element.

        The calculation works in three steps:

        1. **Extract local end displacements** for the beam from the global
           displacement vector ``U``.
        2. **Recover local end forces** via ``f_local = K_local @ u_local``.
           The left-end transverse force is the shear ``V_L`` and the
           left-end moment gives ``M_L``.
        3. **Integrate analytically** along the beam using the
           Euler-Bernoulli relations (no distributed load assumed):

           .. code-block::

               V(x)  = V_L                                    (constant)
               M(x)  = M_L + V_L * x
               κ(x)  = M(x) / EI
               θ(x)  = θ_L + M_L*x/EI + V_L*x²/(2EI)
               v(x)  = v_L + θ_L*x + M_L*x²/(2EI) + V_L*x³/(6EI)

        Sign convention (standard beam theory):
          - ``x``  increases left → right along the beam axis
          - ``w``  positive upward
          - ``θ``  positive counter-clockwise  (= dv/dx)
          - ``M``  positive when sagging (tension on bottom face)
          - ``Q``  positive upward on the left face (= −dM/dx when no
            distributed load)

        Parameters
        ----------
        U : np.ndarray
            Global displacement vector returned by ``solve_fea_system``.
        beam_index : int
            Zero-based index into ``self.beams``.
        is_3d : bool, optional
            Must match the flag used when calling ``solve_fea_system``.
        n_pts : int, optional
            Number of evaluation points along the beam (default 200).

        Returns
        -------
        dict with keys:

        ``'x'``
            Local coordinate along the beam, shape (n_pts,).
        ``'Q'``
            Shear force, shape (n_pts,).
        ``'M'``
            Bending moment, shape (n_pts,).
        ``'kappa'``
            Curvature M/EI, shape (n_pts,).
        ``'theta'``
            Slope dv/dx [rad], shape (n_pts,).
        ``'w'``
            Transverse deflection, shape (n_pts,).
        ``'end_forces'``
            Local end-force vector (6,) for 2-D: ``[F_x1, F_z1, M_y1,
            F_x2, F_z2, M_y2]``.
        """
        beam = self.beams[beam_index]
        idx_1, idx_2 = self.node_idx[beam_index] - 1  # 0-based node indices

        n_dof_node = 6 if is_3d else 3

        # Step 1: gather local end displacements
        # Global DOF slices for each end-node
        s1 = slice(idx_1 * n_dof_node, (idx_1 + 1) * n_dof_node)
        s2 = slice(idx_2 * n_dof_node, (idx_2 + 1) * n_dof_node)
        w_global = np.concatenate([U[s1], U[s2]])

        # Transform to local frame
        T = beam.rotation_matrix(is_3d=is_3d)     # (n_dof_beam × n_dof_beam)
        w_local = T @ w_global

        # Step 2: recover local end forces via local stiffness
        K_local = beam.local_stiffness_matrix(is_3d=is_3d)
        f_local = K_local @ w_local  # [F_x1, F_z1, M_y1,  F_x2, F_z2, M_y2]

        # For 2-D: indices within f_local
        #   node 1:  F_x1=0, F_z1=1, M_y1=2
        #   node 2:  F_x2=3, F_z2=4, M_y2=5
        if is_3d:
            # In 3-D local frame the bending-in-xz-plane DOFs are:
            #   F_z1=2, M_y1=4,  F_z2=8, M_y2=10  (standard 12-DOF ordering)
            Q_L = f_local[2]
            M_L = f_local[4]
            theta_L = w_local[4]
            w_L = w_local[2]
        else:
            # 2-D: [F_x1, F_z1, M_y1, F_x2, F_z2, M_y2]
            Q_L = f_local[1]
            M_L = f_local[2]
            theta_L = w_local[2]
            w_L = w_local[1]

        EI = beam.E * beam.cross_section.iz
        L = beam.L

        # Step 3: integrate
        x = np.linspace(0.0, L, n_pts)

        Q     = np.full_like(x, Q_L)
        M     = M_L + Q_L * x
        kappa = M / EI
        theta = theta_L + M_L * x / EI + Q_L * x**2 / (2.0 * EI)
        w     = w_L + theta_L * x + M_L * x**2 / (2.0 * EI) + Q_L * x**3 / (6.0 * EI)

        return {
            'x': x,
            'w': w,
            'M': M,
            'kappa': kappa,
            'theta': theta,
            'Q': Q,
            'end_forces': f_local,
        }

    def all_beam_fields(self, U, is_3d=False, n_pts=200):
        """Recover internal fields for every beam in the frame and
        assemble them into a single set of arrays ordered along the
        global x-axis.

        Assumes the beams form a collinear series (cantilever topology)
        with monotonically increasing node indices so that they can be
        concatenated in order.

        Parameters
        ----------
        U : np.ndarray
            Global displacement vector from ``solve_fea_system``.
        is_3d : bool, optional
            Must match the flag used when calling ``solve_fea_system``.
        n_pts : int, optional
            Evaluation points per beam segment.

        Returns
        -------
        dict with keys ``'x'``, ``'Q'``, ``'M'``, ``'kappa'``,
        ``'theta'``, ``'v'``, ``'node_x'``, and ``'segment_results'``
        (list of per-beam dicts from :meth:`beam_internal_fields`).
        """
        # Build cumulative node x-coordinates (assumes collinear, left→right)
        node_x = np.zeros(self.n_nodes)
        for i, beam in enumerate(self.beams):
            idx_1, idx_2 = self.node_idx[i] - 1
            node_x[idx_2] = node_x[idx_1] + beam.L

        segment_results = []
        x_all, Q_all, M_all, k_all, th_all, w_all = [], [], [], [], [], []

        for i in range(len(self.beams)):
            res = self.beam_internal_fields(U, i, is_3d=is_3d, n_pts=n_pts)
            idx_1 = self.node_idx[i][0] - 1
            x_offset = node_x[idx_1]

            # Avoid duplicating the shared node between segments
            sl = slice(None) if i == 0 else slice(1, None)

            x_all.append(res['x'][sl] + x_offset)
            Q_all.append(res['Q'][sl])
            M_all.append(res['M'][sl])
            k_all.append(res['kappa'][sl])
            th_all.append(res['theta'][sl])
            w_all.append(res['w'][sl])
            segment_results.append(res)

        return {
            'x':               np.concatenate(x_all),
            'Q':               np.concatenate(Q_all),
            'M':               np.concatenate(M_all),
            'kappa':           np.concatenate(k_all),
            'theta':           np.concatenate(th_all),
            'w':               np.concatenate(w_all),
            'node_x':          node_x,
            'segment_results': segment_results,
        }
