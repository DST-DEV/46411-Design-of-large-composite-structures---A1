from dataclasses import dataclass, field

import numpy as np

from classical_laminate_theory.materials.base import Material2D

@dataclass(frozen=True, slots=True)
class OrthotropicLamina(Material2D):
    """
    Linear elastic orthotropic lamina under plane stress conditions.

    This class represents a homogeneous orthotropic material layer
    defined in its local material coordinate system (1–2), where:
    - 1-direction : fiber direction
    - 2-direction : transverse in-plane direction

    The material follows linear elastic constitutive behavior:
        {σ} = [Q] {ε}
    under the plane stress assumption:
        σ3 = τ13 = τ23 = 0

    The reduced in-plane compliance matrix [S] and stiffness matrix [Q]
    are computed from the engineering constants and stored as immutable
    attributes.

    Parameters
    ----------
    E1 : float
        Young’s modulus in material direction 1.
    E2 : float
        Young’s modulus in material direction 2.
    G12 : float
        In-plane shear modulus.
    nu12 : float
        Major Poisson’s ratio (strain in direction 2 due to stress in
        direction 1).

    Attributes
    ----------
    Q : ndarray of shape (3, 3)
        Reduced in-plane stiffness matrix in the local (1–2) coordinate
        system using engineering shear strain γ_12.

    S : ndarray of shape (3, 3)
        Reduced in-plane compliance matrix in the local (1–2) coordinate
        system.

    Notes
    -----
    - The minor Poisson’s ratio is assumed as: ν_21 = ν_12 * E2 / E1
    - Matrices are defined using engineering shear strain γ_12.
    - The material object is immutable.
    - Valid only for plane stress applications in Classical Laminate Theory.
    - No temperature, damage, or nonlinear effects are included.
    """
    E1: float
    E2: float
    G12: float
    nu12: float
    Q: np.ndarray = field(init=False, repr=False, doc="Stiffness matrix")
    S: np.ndarray = field(init=False, repr=False, doc="Compliance matrix")

    def __post_init__(self):
        S = self._compute_compliance()
        object.__setattr__(self, "S", S)

        Q = np.linalg.inv(S)
        object.__setattr__(self, "Q", Q)

    def _compute_compliance(self) -> np.ndarray:
        """Compute the plane stress compliance matrix (3x3)"""
        S = np.array([
            [1/self.E1, -self.nu12/self.E1, 0],
            [-self.nu12/self.E1, 1/self.E2, 0],
            [0, 0, 1/self.G12]
        ])
        return S

if __name__ == "__main__":
    glass_lamina = OrthotropicLamina(E1=40e9, E2=9.8e9, G12=2.8e9, nu12=.3)
