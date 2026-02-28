import numpy as np
from numpy.typing import ArrayLike

def _validate_numeric(val: int | float | np.number,
                      allow_neg: bool = True,
                      allow_zero: bool = True):
    """Validate if a variable is a scalar numeric value.

    Parameters
    ----------
    val : int | float | np.number
        Variable to check.
    allow_neg : bool, optional
        Selection whether the variable is allowed to be negative.\n
        The default is True.
    allow_zero : bool, optional
        Selection whether the variable is allowed to be zero.\n
        The default is True.

    Returns
    -------
    bool
        True if the variable is a scalar numeric value (and non-zero if
        allow_zero=False and positive if allow_neg=False), else False.

    """
    if not isinstance(val, (int, float, np.number)) or np.isnan(val) \
            or np.isinf(val):
        return False
    elif not allow_neg and val < 0:
        return False
    elif not allow_zero and val == 0:
        return False
    else:
        return True


def _validate_arraylike_numeric(arr: ArrayLike, name: str = "",
                                ndim: int | None = None,
                                allow_neg: bool = True,
                                allow_zero: bool = True,
                                allow_non_finite: bool = False):
    """Validate if a variabe is array-like and contains only numeric values.

    Parameters
    ----------
    arr : ArrayLike
        The variable to check.
    name : str, optional
        Name of the variable (used for exception texts). If name='', the
        default name 'arr' is used.\n
        The default is "".
    ndim : int or None, optional
        Number of ndimension that arr is supposed to have.
        If None is specified, the ndimension is not checked.\n
        The default is None.
    allow_neg : bool, optional
        Selection whether negative values are allowed in the array.\n
        The default is True.
    allow_zero : bool, optional
        Selection whether zeros are allowed in the array.\n
        The default is True.
    allow_non_finite : bool, optional
        Selection whether non-finite values (this also includes NaN) are
        allowed in the array.\n
        The default is False.

    Raises
    ------
    TypeError
        If name is not a string.\n
        If arr contains non-numeric values.
    ValueError
        If ndim is not a positive integer or None.\n
        If arr does not have the number of ndimensions specified in ndim.\n
        If arr is empty.\n
        If arr contains negative values (only if allow_neg=False).\n
        If arr contains zeros (only if allow_zero=False).

    Returns
    -------
    arr : np.ndarray
        The input array-like as a numpy array.

    """
    if not isinstance(name, str):
        raise TypeError("name must be a string.")

    if not name:
        name = "Array"

    arr = np.asarray(arr)
    if ndim is not None:
        if not isinstance(ndim, int) \
                or not _validate_numeric(ndim,
                                         allow_neg=True, allow_zero=True):
            raise ValueError("ndim must be None or a positive integer.")

        if arr.ndim != ndim:
            raise ValueError(f"{name} must be {ndim}D.")

    if not np.issubdtype(arr.dtype, np.number) or not np.isrealobj(arr):
        raise TypeError(f"{name} must contain only numeric values.")

    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only contain finite, non NaN "
                         "values.")

    if arr.size == 0:
        raise ValueError(f"{name} cannot be empty.")

    if not allow_neg and np.any(arr < 0):
        raise ValueError(f"{name} must contain only positive values.")

    if not allow_zero and np.any(arr == 0):
        raise ValueError(f"{name} must contain only non-zero values.")

    return arr
