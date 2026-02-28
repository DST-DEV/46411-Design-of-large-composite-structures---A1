from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Callable, Sequence, Union

import yaml
import numpy as np
import pandas as pd

import classical_laminate_theory as clt

class TurbineBlade:
    def __init__(self, data_dir = Path(__file__).parent / "_data"):
        data_dir = Path(data_dir)

        xj, _ = self._load_spline_params(data_dir / "coeffs_chord.csv")

        self._r_root = min(xj)
        self._L = max(xj)

        self._chord = self._make_cubic_spline(
            *self._load_spline_params(data_dir / "coeffs_chord.csv"))
        self._w_spar = self._make_cubic_spline(
            *self._load_spline_params(data_dir / "coeffs_spar_cap_width.csv"))
        self._thickness = self._make_cubic_spline(
            *self._load_spline_params(data_dir / "coeffs_thickness.csv"))

        with open(data_dir / 'materials.yml', 'r') as file:
            materials = yaml.safe_load(file)

        self._lamina = materials.get("lamina")
        self._multiax_plies = materials.get("multidirectional_ply")

        self._layup = pd.read_csv(data_dir / "layup.csv")
        self._layup.columns = [col.split("[")[0].strip().lower()
                               for col in self._layup.columns]

    @property
    def r_root(self):
        """Root radius [m]."""
        return self._r_root

    @property
    def L(self):
        """Nominal blade length [m]."""
        return self._L

    @property
    def lamina(self):
        """Lamina material properties."""
        return self._lamina

    @property
    def ply_multidir(self):
        """Multiaxial ply material properties."""
        return self._ply_multidir

    @property
    def layup(self):
        """Composite layup of the blade."""
        return self._layup

    def chord(self, x: Union[int, float, np.number, np.ndarray]
              ) -> Union[float, np.ndarray]:
        """
        Calculate the chord length at a spanwise position x.

        Parameters
        ----------
        x : int | float | np.number | np.ndarray
            Spanwise positions along the blade (incl. the root radius) [m].

        Returns
        -------
        np.ndarray
            Chord lengths at the specified spanwise positions [m].
        """
        return self._chord(x)

    def w_spar(self, x: Union[int, float, np.number, np.ndarray]
              ) -> Union[float, np.ndarray]:
        """
        Calculate the spar cap width at a spanwise position x.

        Parameters
        ----------
        x : int | float | np.number | np.ndarray
            Spanwise positions along the blade (incl. the root radius) [m].

        Returns
        -------
        np.ndarray
            Spar cap width at the specified spanwise positions [m].
        """
        return self._w_spar(x)

    def thickness(self, x: Union[int, float, np.number, np.ndarray]
              ) -> Union[float, np.ndarray]:
        """
        Calculate the relative thickness at a spanwise position x.

        Parameters
        ----------
        x : int | float | np.number | np.ndarray
            Spanwise positions along the blade (incl. the root radius) [m].

        Returns
        -------
        np.ndarray
            Relative thickness at the specified spanwise positions.
        """
        return self._thickness(x)

    def thickness_abs(self, x: Union[int, float, np.number, np.ndarray]
              ) -> Union[float, np.ndarray]:
        """
        Calculate the absolute thickness at a spanwise position x.

        Parameters
        ----------
        x : int | float | np.number | np.ndarray
            Spanwise positions along the blade (incl. the root radius) [m].

        Returns
        -------
        np.ndarray
            Relative thickness at the specified spanwise positions [m].
        """
        return self._thickness(x)*self._chord(x)

    @staticmethod
    def _load_spline_params(fpath: PurePath) -> tuple[np.ndarray, np.ndarry]:
        """
        Load spline parameters from a .csv file.

        Parameters
        ----------
        fpath : PurePath
            Path to the .csv file containing the breaks and coefficients.

        Returns
        -------
        xj : np.ndarray of shape (n_breaks,)
            The break points of the spline.
        cj : np.ndarray of shape (n_breaks,n_params)
            The coefficients of the spline
        """
        xj1, xj2, *cj = np.loadtxt(fpath, skiprows=1, unpack=True,
                                   delimiter=",")

        xj = np.zeros(len(xj1) + 1)
        xj[:-1] = xj1
        xj[-1] = xj2[-1]
        cj = np.array(cj).T

        return xj, cj

    @staticmethod
    def _make_cubic_spline(breaks: Sequence[float],
                          coefficients: Sequence[Sequence[float]],
                          ) -> Callable[[Union[float, np.ndarray]],
                                        Union[float, np.ndarray]]:
        """
        Create a vectorized cubic spline evaluator.

        Parameters
        ----------
        breaks : sequence of floats
            Strictly increasing breakpoints [xi_0, ..., xi_n]
        coefficients : sequence of sequences
            Shape (n, 4). Each row:
            [c1, c2, c3, c4] for interval [xi_j, xi_{j+1}]

        Returns
        -------
        spline : callable
            Evaluates spline at scalar or array input.
        """

        breaks = np.asarray(breaks, dtype=float)
        coeffs = np.asarray(coefficients, dtype=float)

        if breaks.ndim != 1:
            raise ValueError("`breaks` must be 1D.")
        if not np.all(np.diff(breaks) > 0):
            raise ValueError("`breaks` must be strictly increasing.")

        n_intervals = len(breaks) - 1

        if not (5 <= n_intervals <= 12):
            raise ValueError("Number of intervals must be between 5 and 12.")

        if coeffs.shape != (n_intervals, 4):
            raise ValueError(
                f"`coefficients` must have shape ({n_intervals}, 4)."
            )

        def spline(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
            x_arr = np.asarray(x, dtype=float)

            if np.any((x_arr < breaks[0]) | (x_arr > breaks[-1])):
                raise ValueError("Some x values are outside the spline domain.")

            # Find interval indices for all x at once
            indices = np.searchsorted(breaks, x_arr, side="right") - 1

            # Fix right boundary case (x == breaks[-1])
            indices = np.clip(indices, 0, n_intervals - 1)

            dx = x_arr - breaks[indices]

            c1 = coeffs[indices, 0]
            c2 = coeffs[indices, 1]
            c3 = coeffs[indices, 2]
            c4 = coeffs[indices, 3]

            result = c1*dx**3 + c2*dx**2 + c3*dx + c4

            # Return scalar if input was scalar
            if np.isscalar(x):
                return float(result)
            return result

        return spline

    @staticmethod
    def create_lamina(material: dict) -> clt.materials.EquivalentLamina:
        """
        Create a lamina from a dict of material properties

        Parameters
        ----------
        material : dict
            Material properties of the lamina.

        Returns
        -------
        mat : clt.materials.EquivalentLamina
            Equivalent lamina with the specified material properties.

        """
        stiffness = material["elastic_properties"]
        strength = material["strength_properties"]
        mat = clt.materials.EquivalentLamina(
            E1=stiffness["E1"], E2=stiffness["E2"],
            G12=stiffness["G12"], nu12=stiffness["nu12"],
            nu21=stiffness["nu21"],
            s_hat_1t=strength["sigma_1t"], s_hat_1c=strength["sigma_1c"],
            s_hat_2t=strength["sigma_2t"], s_hat_2c=strength["sigma_2c"],
            t_hat_12=strength["tau_12"])

        return mat

    def create_laminates(self, layup: None | pd.core.frame.DataFrame = None
                         ) -> list[BladeSection]:
        """
        Create the laminates from a spanwise layup distribution.

        Parameters
        ----------
        layup : None | pd.core.frame.DataFrame, optional
            Layup of the laminates along the blade span.
            If None is given, the layup of the class instance is used.
            The default is None.

        Raises
        ------
        TypeError
            If layup is not a pandas DataFrame.

        Returns
        -------
        list[BladeSection]
            List of blade sections containing the laminate and spanwise
            position for each cross section.

        """
        if layup is None:
            layup = self.layup
        elif not isinstance(layup, pd.core.frame.DataFrame):
            raise TypeError("layup must be a pandas DataFrame.")

        uniax = self.create_lamina(self._multiax_plies["uniax"])
        triax = self.create_lamina(self._multiax_plies["triax"])

        laminates = []
        for i, section in layup.iterrows():
            plies = []
            for ply in ["triax1", "uniax1", "uniax2", "triax2"]:
                if section[ply]>0:
                    if "uniax" in ply:
                        plies.append(clt.Ply(material=uniax, theta=0,
                                             thickness=section[ply]))
                    else:
                        plies.append(clt.Ply(material=triax, theta=0,
                                             thickness=section[ply]))

            if len(plies)>0:
                thickness = sum(p.thickness for p in plies)

                laminates.append(BladeSection(
                    r_start=section.r_start,
                    r_end=section.r_end,
                    length=section.r_end-section.r_start,
                    thickness=thickness,
                    laminate=clt.Laminate(plies)))
            else:
                pass

        return laminates


@dataclass(frozen=True, slots=True)
class BladeSection:
    r_start: float
    r_end: float
    length: float
    thickness: float
    laminate: clt.Laminate

if __name__ == "__main__":
    blade = TurbineBlade()

    import matplotlib.pyplot as plt
    x = np.linspace(blade.r_root, blade.L, 300)
    plt.plot(x, blade.thickness(x))
    plt.plot(x, blade.thickness_abs(x))

    laminates = blade.create_laminates()
