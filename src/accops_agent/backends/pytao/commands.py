"""Tao command builders for common operations."""

from typing import List


def build_read_data_command(element_name: str, attribute: str) -> str:
    """Build command to read data from an element.

    Args:
        element_name: Name of the element
        attribute: Attribute to read

    Returns:
        Tao command string
    """
    return f"python lat_ele_list {element_name} -attribute {attribute}"


def build_set_parameter_command(
    element_name: str, attribute: str, value: float
) -> str:
    """Build command to set parameter value.

    Args:
        element_name: Name of the element
        attribute: Attribute to set
        value: Value to set

    Returns:
        Tao command string
    """
    return f"set ele {element_name} {attribute} = {value}"


def build_get_parameter_command(element_name: str, attribute: str) -> str:
    """Build command to get parameter value.

    Args:
        element_name: Name of the element
        attribute: Attribute to get

    Returns:
        Tao command string
    """
    return f"python lat_ele_list {element_name} -attribute {attribute}"


def build_run_calculation_command() -> str:
    """Build command to run beam calculation.

    Returns:
        Tao command string
    """
    return "set global track_type = single"


def build_read_floor_coordinates(element_name: str) -> str:
    """Build command to read floor coordinates of element.

    Args:
        element_name: Name of the element

    Returns:
        Tao command string
    """
    return f"python lat_ele_list {element_name} -attribute x -attribute y -attribute z"


def build_read_orbit_command(element_names: List[str]) -> str:
    """Build command to read orbit at multiple elements.

    Args:
        element_names: List of element names

    Returns:
        Tao command string
    """
    # For multiple elements, we'll need to read them individually
    # This is a simplified version - real implementation may vary
    return "python lat_ele_list * -attribute orbit_x -attribute orbit_y"


def build_read_beam_size_command(element_name: str) -> str:
    """Build command to read beam size at element.

    Args:
        element_name: Name of the element

    Returns:
        Tao command string
    """
    return f"python lat_ele_list {element_name} -attribute sig_x -attribute sig_y"


def build_read_twiss_command(element_name: str) -> str:
    """Build command to read Twiss parameters.

    Args:
        element_name: Name of the element

    Returns:
        Tao command string
    """
    return f"python lat_ele_list {element_name} -attribute beta_a -attribute alpha_a -attribute beta_b -attribute alpha_b"
