# import libs
import logging
from typing import Any, Dict, List, Optional, Tuple
from pythermodb_settings.models import Component, ComponentKey
from pythermodb_settings.utils import set_component_id
from pyThermoLinkDB.thermo import Source
from pyThermoLinkDB.models.component_models import ComponentEquationSource

# locals

# NOTE: logger setup
logger = logging.getLogger(__name__)

# SECTION: Extract data from source


def ext_component_dt(
        component_id: str,
        prop_name: str,
        source: Source
) -> Optional[Dict[str, Any]]:
    """
    Extract data from the source for a given component ID and property name.

    Parameters
    ----------
    component_id : str
        The ID of the component for which to extract data.
    prop_name : str
        The name of the property to extract.
    source : Source
        The source from which to extract the data.

    Returns
    -------
    Optional[Dict[str, Any]]
        A dictionary containing the extracted data, or None if an error occurs.
    """
    try:
        # NOTE: extract data
        data = source.data_extractor(
            component_id=component_id,
            prop_name=prop_name
        )
        return data
    except Exception as e:
        logger.error(
            f"Error extracting data for component_id: {component_id}, prop_name: {prop_name} - {e}")
        return None


def ext_components_dt(
        component_ids: List[str],
        prop_name: str,
        source: Source
) -> Optional[Dict[str, Dict[str, Any]]]:
    """
    Extract data for multiple component IDs from the source for a given property name.

    Parameters
    ----------
    component_ids : List[str]
        A list of component IDs for which to extract data.
    prop_name : str
        The name of the property to extract.
    source : Source
        The source from which to extract the data.

    Returns
    -------
    Optional[Dict[str, Dict[str, Any]]]
        A dictionary mapping component IDs to their extracted data, or None if an error occurs.
    """
    try:
        # NOTE: extract data for each component ID
        data_dict = {}
        for component_id in component_ids:
            data = ext_component_dt(
                component_id=component_id,
                prop_name=prop_name,
                source=source
            )
            if data is not None:
                data_dict[component_id] = data
        return data_dict
    except Exception as e:
        logger.error(
            f"Error extracting data for component_ids: {component_ids}, prop_name: {prop_name} - {e}")
        return None

# SECTION: Extract equation from source


def ext_component_eq(
        component: Component,
        prop_name: str,
        component_key: ComponentKey,
        source: Source
) -> Optional[ComponentEquationSource]:
    """
    Extract an equation from the source for a given component and property name.

    Parameters
    ----------
    component : Component
        The component for which to extract the equation.
    prop_name : str
        The name of the property for which to extract the equation.
    component_key : ComponentKey
        The component key to use for setting component IDs.
    source : Source
        The source from which to extract the equation.

    Returns
    -------
    Optional[ComponentEquationSource]
        A ComponentEquationSource object containing the extracted equation, or None if an error occurs.
    """
    try:
        # NOTE: extract equation
        eq_src = source.eq_builder(
            components=[component],
            prop_name=prop_name
        )

        # >> check
        if eq_src is None:
            logger.warning(
                f"No equation found for component: {component}, prop_name: {prop_name}")
            return None

        # >> component id
        component_id = set_component_id(
            component=component,
            component_key=component_key
        )

        # NOTE: create ComponentEquationSource
        component_eq_src: ComponentEquationSource | None = eq_src.get(
            component_id
        )

        # >> check
        if component_eq_src is None:
            logger.warning(
                f"No equation source found for component_id: {component_id}, prop_name: {prop_name}")
            return None

        return component_eq_src
    except Exception as e:
        logger.error(
            f"Error extracting equation for component: {component}, prop_name: {prop_name} - {e}")
        return None


def ext_components_eq(
        components: List[Component],
        prop_name: str,
        component_key: ComponentKey,
        source: Source
) -> Optional[Dict[str, ComponentEquationSource]]:
    """
    Extract equations from the source for multiple components and a given property name.

    Parameters
    ----------
    components : List[Component]
        A list of components for which to extract equations.
    prop_name : str
        The name of the property for which to extract equations.
    component_key : ComponentKey
        The component key to use for setting component IDs.
    source : Source
        The source from which to extract the equations.

    Returns
    -------
    Optional[Dict[str, ComponentEquationSource]]
        A dictionary mapping component IDs to their extracted ComponentEquationSource objects, or None if an error occurs.
    """
    try:
        # NOTE: extract equation for each component
        eq_dict = {}

        for component in components:
            component_eq_src = ext_component_eq(
                component=component,
                prop_name=prop_name,
                component_key=component_key,
                source=source
            )
            if component_eq_src is not None:
                component_id = set_component_id(
                    component=component,
                    component_key=component_key
                )
                eq_dict[component_id] = component_eq_src

        return eq_dict
    except Exception as e:
        logger.error(
            f"Error extracting equations for components: {components}, prop_name: {prop_name} - {e}")
        return None
