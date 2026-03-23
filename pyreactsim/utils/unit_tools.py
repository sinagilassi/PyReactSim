# import libs
import logging
import pycuc


def to_m3(value: float, unit: str) -> float:
    """
    Convert a given value to cubic meters (m3) based on the specified unit.

    Parameters
    ----------
    value : float
        The numerical value to be converted.
    unit : str
        The unit of the input value. Supported units include 'm3', 'L', 'cm3', 'ft3', etc.

    Returns
    -------
    float
        The converted value in cubic meters (m3).

    Raises
    ------
    ValueError
        If the provided unit is not supported for conversion.
    """
    try:
        # Use pycuc for unit conversion
        return pycuc.convert_from_to(value, from_unit=unit, to_unit='m3')
    except Exception as e:
        logging.error(f"Error converting {value} from {unit} to m3: {e}")
        raise ValueError(f"Unsupported unit for conversion: {unit}")


def to_Pa(value: float, unit: str) -> float:
    """
    Convert a given value to Pascals (Pa) based on the specified unit.

    Parameters
    ----------
    value : float
        The numerical value to be converted.
    unit : str
        The unit of the input value. Supported units include 'Pa', 'kPa', 'MPa', 'bar', 'atm', etc.

    Returns
    -------
    float
        The converted value in Pascals (Pa).

    Raises
    ------
    ValueError
        If the provided unit is not supported for conversion.
    """
    try:
        # Use pycuc for unit conversion
        return pycuc.convert_from_to(value, from_unit=unit, to_unit='Pa')
    except Exception as e:
        logging.error(f"Error converting {value} from {unit} to Pa: {e}")
        raise ValueError(f"Unsupported unit for conversion: {unit}")


def to_K(value: float, unit: str) -> float:
    """
    Convert a given value to Kelvin (K) based on the specified unit.

    Parameters
    ----------
    value : float
        The numerical value to be converted.
    unit : str
        The unit of the input value. Supported units include 'K', 'C', 'F', etc.

    Returns
    -------
    float
        The converted value in Kelvin (K).

    Raises
    ------
    ValueError
        If the provided unit is not supported for conversion.
    """
    try:
        # Use pycuc for unit conversion
        return pycuc.convert_from_to(value, from_unit=unit, to_unit='K')
    except Exception as e:
        logging.error(f"Error converting {value} from {unit} to K: {e}")
        raise ValueError(f"Unsupported unit for conversion: {unit}")
