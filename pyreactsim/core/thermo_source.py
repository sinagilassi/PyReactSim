# import libs
import logging
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, cast
from pythermodb_settings.models import Component, ComponentKey
from pyThermoLinkDB.thermo import Source
from pyThermoLinkDB.models.component_models import ComponentEquationSource
# locals
from ..sources.interface import ext_component_dt, ext_components_dt, ext_component_eq, ext_components_eq

# NOTE: logger setup
logger = logging.getLogger(__name__)


class ThermoSource:
    """
    THermo class for handling thermodynamic calculations and properties related to chemical reactions and processes. This class provides methods for calculating various thermodynamic properties, such as heat capacities, enthalpies, and entropies, as well as methods for performing energy balance calculations in chemical systems.
    """

    def __init__(
        self,
        components: List[Component],
        source: Source,
        component_key: ComponentKey,
    ):
        """
        Initializes the THermo instance with default properties and settings for thermodynamic calculations.
        """
        # NOTE: Set attributes
        self.components = components
        self.source = source
        self.component_key = component_key

        # NOTE: Create component ID list

    def prop_eq_src(self, prop_name: str) -> Dict[str, ComponentEquationSource]:
        """
        Extracts the property equation for the components from the source and returns it as a dictionary.

        Returns
        -------
        Dict[str, ComponentEquationSource]
            A dictionary where the keys are component IDs and the values are ComponentEquationSource objects
        """
        # NOTE: Extract property equation source for all components
        eq_src = ext_components_eq(
            components=self.components,
            prop_name=prop_name,
            source=self.source,
            component_key=cast(ComponentKey, self.component_key)
        )
        # >> check
        if eq_src is None:
            logger.error("Failed to extract property equation for components.")
            return {}

        return eq_src

    def prop_dt_src(
            self,
            component_ids: List[str],
            prop_name: str
    ) -> Dict[str, Dict[str, Any]]:
        """
        Extracts the property derivative equation for the components from the source and returns it as a dictionary.

        Returns
        -------
        Dict[str, ComponentEquationSource]
            A dictionary where the keys are component IDs and the values are ComponentEquationSource objects
        """
        # NOTE: Extract property derivative equation source for all components
        dt_src = ext_components_dt(
            component_ids=component_ids,
            prop_name=prop_name,
            source=self.source,
        )
        # >> check
        if dt_src is None:
            logger.error(
                "Failed to extract property derivative equation for components.")
            return {}

        return dt_src
