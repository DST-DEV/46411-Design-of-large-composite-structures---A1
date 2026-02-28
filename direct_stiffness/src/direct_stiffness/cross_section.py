from abc import ABC

import numpy as np

__all__ = ["Rectangle", "Circle", "Pipe", "RectangularPipe"]

class CrossSection(ABC):
    """Base class for cross sections.

    Attributes
    ----------
    dimensions : dict
        Dictionary containing geometric parameters (e.g., width, height,
        diameter).
    """

    def __init__(self, **kwargs):
        """Initialize a cross section.

        Parameters
        ----------
        **kwargs : float
            Named dimensions of the cross section.
        """
        self._dimensions = kwargs

    @property
    def type(self):
        """str: Cross section type (class name)."""
        return self.__class__.__name__

    @property
    def dimensions(self):
        """dict: Dimensions of the cross section."""
        return self._dimensions

    @property
    def area(self):
        """float: Cross-sectional area."""

    @property
    def iy(self):
        """float: Moment of inertia about y-axis."""

    @property
    def iz(self):
        """float: Moment of inertia about z-axis."""

    @property
    def j(self):
        """float: Polar moment of inertia."""


class Rectangle(CrossSection):
    """Rectangular cross section.

    Parameters
    ----------
    b : float
        Width of the rectangle.
    h : float
        Height of the rectangle.
    """

    def __init__(self, **kwargs):
        if any(not isinstance(val, (int, float, np.number))
               for val in kwargs.values()):
            raise TypeError("Dimensions must be scalar numeric values.")

        self._dimensions = {}
        # Check if dimension are complete (if only one dimension is given, a
        # square is assumed)
        if kwargs.get("b") is None:
            if kwargs.get("h") is not None:
                self._dimensions["b"] = kwargs["h"]
                self._dimensions["h"] = kwargs["h"]
            else:
                raise ValueError("Incomplete dimensions.")
        else:
            if kwargs.get("h") is None:
                self._dimensions["b"] = kwargs["b"]
                self._dimensions["h"] = kwargs["b"]
            else:
                self._dimensions["b"] = kwargs["b"]
                self._dimensions["h"] = kwargs["h"]

    @property
    def area(self):
        """float: Cross-sectional area."""
        b, h = self.dimensions["b"], self.dimensions["h"]
        return b * h

    @property
    def iy(self):
        """float: Moment of inertia about y-axis."""
        b, h = self.dimensions["b"], self.dimensions["h"]
        return (h * b**3) / 12

    @property
    def iz(self):
        """float: Moment of inertia about z-axis."""
        b, h = self.dimensions["b"], self.dimensions["h"]
        return (b * h**3) / 12

    @property
    def j(self):
        """float: Polar moment of inertia."""
        b, h = self.dimensions["b"], self.dimensions["h"]
        return (h * b * (b**2 + h**2)) / 12


class Circle(CrossSection):
    """Circular cross section.

    Parameters
    ----------
    d : float
        Diameter of the circle.
    """

    def __init__(self, **kwargs):
        if any(not isinstance(val, (int, float, np.number))
               for val in kwargs.values()):
            raise TypeError("Dimensions must be scalar numeric values.")

        if kwargs.get("d") is None:
            raise ValueError("Incomplete dimensions.")
        else:
            self._dimensions = {"d": kwargs["d"]}

    @property
    def area(self):
        """float: Cross-sectional area."""
        d = self.dimensions["d"]
        return np.pi * (d**2) / 64

    @property
    def iy(self):
        """float: Moment of inertia about y-axis."""
        d = self.dimensions["d"]
        return (np.pi * d**4) / 64

    @property
    def iz(self):
        """float: Moment of inertia about z-axis (same as Iy)."""
        return self.iy

    @property
    def j(self):
        """float: Polar moment of inertia."""
        d = self.dimensions["d"]
        return (np.pi * d**4) / 32


class Pipe(CrossSection):
    """Hollow circular pipe cross section.

    Parameters
    ----------
    do : float
        Outer diameter.
    di : float
        Inner diameter.
    t : float
        Wall thickness
    """

    def __init__(self, **kwargs):
        if any(not isinstance(val, (int, float, np.number))
               for val in kwargs.values()):
            raise TypeError("Dimensions must be scalar numeric values.")

        self._dimensions = {}
        # Check if dimensions are complete (either outer diameter & thickness,
        # inner diameter & thickness or outer & inner diameter)
        if kwargs.get("di") is None:
            if any(kwargs.get(dim) is None for dim in ["do", "t"]):
                raise ValueError("Incomplete dimensions.")

            self._dimensions["do"] = kwargs["do"]
            self._dimensions["t"] = kwargs["t"]

            self._dimensions["di"] = kwargs["do"] - 2 * kwargs["t"]
        elif kwargs.get("do") is None:
            if any(kwargs.get(dim) is None for dim in ["di", "t"]):
                raise ValueError("Incomplete dimensions.")

            self._dimensions["di"] = kwargs["di"]
            self._dimensions["t"] = kwargs["t"]

            self._dimensions["do"] = kwargs["di"] + 2 * kwargs["t"]
        else:
            self._dimensions["do"] = kwargs["do"]
            self._dimensions["di"] = kwargs["di"]

    @property
    def area(self):
        """float: Cross-sectional area."""
        do, di = self.dimensions["do"], self.dimensions["di"]
        return np.pi * (do**2 - di**2) / 4

    @property
    def iy(self):
        """float: Moment of inertia about y-axis."""
        do, di = self.dimensions["do"], self.dimensions["di"]
        return (np.pi / 64) * (do**4 - di**4)

    @property
    def iz(self):
        """float: Moment of inertia about z-axis (same as Iy)."""
        return self.iy

    @property
    def j(self):
        """float: Polar moment of inertia."""
        do, di = self.dimensions["do"], self.dimensions["di"]
        return (np.pi / 32) * (do**4 - di**4)


class RectangularPipe(CrossSection):
    """Initialize a cross section.

    Parameters
    ----------
    bo : float
        Outer width.
    ho : float
        Outer height.
    bi : float
        Inner width.
    hi : float
        Inner height.
    t : float
        Wall thickness.
    """

    def __init__(self, **kwargs):
        if any(not isinstance(val, (int, float, np.number))
               for val in kwargs.values()):
            raise TypeError("Dimensions must be scalar numeric values.")

        self._dimensions = {}
        # Check if dimension are complete (either inner & outer dimensions,
        # outer dimensions & thickness or inner dimensions & thickness)
        if any(kwargs.get(dim_i) is None for dim_i in ["bi", "hi"]):
            if kwargs.get("t") is None:
                raise ValueError("Incomplete dimensions.")

            # Check if outer dimension are complete (if only one dimension is
            # given, a square is assumed)
            if kwargs.get("bo") is None:
                if kwargs.get("ho") is not None:
                    self._dimensions["bo"] = kwargs["ho"]
                    self._dimensions["ho"] = kwargs["ho"]
                else:
                    raise ValueError("Incomplete dimensions.")
            else:
                if kwargs.get("ho") is None:
                    self._dimensions["bo"] = kwargs["bo"]
                    self._dimensions["ho"] = kwargs["bo"]
                else:
                    self._dimensions["bo"] = kwargs["bo"]
                    self._dimensions["ho"] = kwargs["ho"]

            self._dimensions["bi"] = self._dimensions["bo"] - 2*kwargs["t"]
            self._dimensions["hi"] = self._dimensions["ho"] - 2*kwargs["t"]
        elif any(kwargs.get(dim_o) is None for dim_o in ["bo", "ho"]):
            if kwargs.get("t") is None:
                raise ValueError("Incomplete dimensions.")

            # Check if outer dimension are complete (if only one dimension is
            # given, a square is assumed)
            if kwargs.get("bi") is None:
                if kwargs.get("hi") is not None:
                    self._dimensions["bi"] = kwargs["hi"]
                    self._dimensions["hi"] = kwargs["hi"]
                else:
                    raise ValueError("Incomplete dimensions.")
            else:
                if kwargs.get("hi") is None:
                    self._dimensions["bi"] = kwargs["bi"]
                    self._dimensions["hi"] = kwargs["bi"]
                else:
                    self._dimensions["bi"] = kwargs["bi"]
                    self._dimensions["hi"] = kwargs["hi"]

            self._dimensions["bo"] = self._dimensions["bi"] + 2*kwargs["t"]
            self._dimensions["ho"] = self._dimensions["hi"] + 2*kwargs["t"]
        else:
            self._dimensions["bi"] = kwargs["bi"]
            self._dimensions["hi"] = kwargs["hi"]
            self._dimensions["bo"] = kwargs["bo"]
            self._dimensions["ho"] = kwargs["ho"]

    @property
    def area(self):
        """float: Cross-sectional area."""
        bi = self.dimensions["bi"]
        hi = self.dimensions["hi"]
        bo = self.dimensions["bo"]
        ho = self.dimensions["ho"]
        return bo * ho - bi * hi

    @property
    def iy(self):
        """float: Moment of inertia about y-axis."""
        bi = self.dimensions["bi"]
        hi = self.dimensions["hi"]
        bo = self.dimensions["bo"]
        ho = self.dimensions["ho"]
        return (bo**3 * ho - bi**3 * hi)/12

    @property
    def iz(self):
        """float: Moment of inertia about z-axis."""
        bi = self.dimensions["bi"]
        hi = self.dimensions["hi"]
        bo = self.dimensions["bo"]
        ho = self.dimensions["ho"]
        return (bo**3 * ho - bi**3 * hi)/12

    @property
    def j(self):
        """float: Polar moment of inertia."""
        bi = self.dimensions["bi"]
        hi = self.dimensions["hi"]
        bo = self.dimensions["bo"]
        ho = self.dimensions["ho"]
        return (bo * ho*(bo**2 + ho**2) - bi * hi*(bi**2 + hi**2))/12


class ManualCrossSection(CrossSection):
    """Cross section with manual settings for the moments of inertia.

    Attributes
    ----------
    dimensions : dict
        Dictionary containing geometric parameters (e.g., width, height,
        diameter).
    """

    def __init__(self, A, iy, iz, j):
        """Initialize a cross section.

        Parameters
        ----------
        A : float
            Cross sectional area.
        iy : float
            Moment of inertia about y-axis (perpendicular).
        iz : float
            Moment of inertia about z-axis (vertical).
        j : float
            Polar moment of inertia.
        """
        self._area = A
        self._iy = iy
        self._iz = iz
        self._j = j

    @property
    def type(self):
        """str: Cross section type (class name)."""
        return self.__class__.__name__

    @property
    def area(self):
        """float: Cross-sectional area."""
        return self._area

    @property
    def iy(self):
        """float: Moment of inertia about y-axis."""
        return self._iy

    @property
    def iz(self):
        """float: Moment of inertia about z-axis."""
        return self._iz

    @property
    def j(self):
        """float: Polar moment of inertia."""
        return self._j

if __name__ == "__main__":
    rect = Rectangle(b=0.1, h=0.2)
    circ = Circle(d=0.05)
    pipe = Pipe(do=0.1, di=0.08)
    rpipe = RectangularPipe(bo=0.2, ho=0.15, t=0.01)

    for section in [rect, circ, pipe, rpipe]:
        print(f"{section.type}: Area={section.area:.6f}, Ix={section.ix:.6e}, Iy={section.iy:.6e}, J={section.j:.6e}")
