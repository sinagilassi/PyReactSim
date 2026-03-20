# import libs
import logging
from typing import Any, Dict, List, Optional, Tuple, cast
import pycuc
from pythermodb_settings.models import Component, Temperature, Pressure, ComponentKey
from pythermodb_settings.utils import set_component_id
from pyThermoLinkDB.thermo import Source
from pyThermoLinkDB.models.component_models import ComponentEquationSource
# ! locals
from ..sources.interface import (
    ext_component_dt,
    ext_components_dt,
    ext_component_eq,
    ext_components_eq
)

# NOTE: logger setup
logger = logging.getLogger(__name__)


class BatchReactor:
    """
    Batch Reactor (BR) class for simulating chemical reactions in a batch reactor setup. This class encapsulates the components, source, and component key information necessary for performing simulations and analyses related to batch reactors.
    """
    # NOTE: Properties
    #

    def __init__(
        self,
        components: List[Component],
        source: Source,
        component_key: ComponentKey,
    ):
        """
        Initializes the BatchReactor instance with the provided components, source, and component key.

        Parameters
        ----------
        components : List[Component]
            A list of Component objects representing the chemical components involved in the batch reactor.
        source : Source
            A Source object containing information about the source of the data or equations used in the batch reactor simulations.
        component_key : ComponentKey
            A ComponentKey object that serves as a key for identifying and categorizing the components in the batch reactor.
        """
        # NOTE: Set attributes
        self.components = components
        self.source = source
        self.component_key = component_key

        # SECTION: set component IDs
        self.component_ids = [
            set_component_id(
                component=component,
                component_key=cast(ComponentKey, self.component_key)
            )
            for component in self.components
        ]

        # NOTE: mole fraction components
        self.mole_fractions = [
            c.mole_fraction for c in self.components
        ]

        # NOTE: state components
        self.states = [
            c.state for c in self.components
        ]

    def constant_volume(self):
        pass
