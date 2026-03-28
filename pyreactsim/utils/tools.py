# import libs
import logging
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from pythermodb_settings.models import Component, ComponentKey, CustomProp, Pressure, Temperature, CustomProperty
from pythermodb_settings.utils import set_component_id


def find_components_property(
        components: List[Component],
        prop_values: Dict[str, float],
        component_key: ComponentKey,
):
    """
    Find the specified property values for the components based on the component key.

    Parameters
    ----------
    components : List[Component]
        A list of Component objects representing the components in the reaction.
    prop_values : Dict[str, float]
        A dictionary with component names as keys and the property values to be extracted as values.
    component_key : ComponentKey
        A ComponentKey object representing the key to be used for the components in the model source.

    Returns
    -------
    Tuple[Dict[str, CustomProp], np.ndarray]
        A dictionary with component names as keys and CustomProp objects containing the extracted property values and a numpy array of the extracted property values for each component.

    Raises
    ------
    ValueError
        If a component specified in the prop_values dictionary is not found in the components list based on the component key.
    """
    # NOTE: extract the specified property values for the components based on the component key
    extracted_values = []
    extracted_values_comp = {}

    # iterate through components
    for comp in components:
        comp_id = set_component_id(comp, component_key)
        if comp_id in prop_values:
            # store
            extracted_values.append(prop_values[comp_id])
            extracted_values_comp[comp_id] = prop_values[comp_id]
        else:
            raise ValueError(
                f"Component '{comp_id}' not found in the provided property values."
            )

    # to array
    extracted_values = np.array(extracted_values, dtype=float)

    return extracted_values, extracted_values_comp


# SECTION: collect keys
def collect_keys(
        data: Dict[str, Any]) -> List[str]:
    """Extract keys"""
    keys = set(data.keys())
    # lower case keys for easier access
    keys = [
        key.lower().strip() for key in keys
    ]
    return keys
