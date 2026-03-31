# import libs
import logging
import numpy as np
from typing import List, Dict, Any, cast
from pythermodb_settings.models import Component, Temperature, Pressure, CustomProperty, ComponentKey
from pyThermoLinkDB.thermo import Source
from pyThermoLinkDB.models.component_models import ComponentEquationSource
# locals
from ..sources.interface import (
    ext_components_dt,
    ext_components_eq,
    exec_component_eq
)
from ..models.br import BatchReactorOptions
from ..models.rate_exp import ReactionRateExpression
from ..utils.tools import config_components_property
from ..utils.unit_tools import to_J_per_mol, to_g_per_mol

# NOTE: logger setup
logger = logging.getLogger(__name__)


class ThermoModelSource:
    # NOTE: Attributes
    # ! sources
    Cp_IG_src: Dict[str, ComponentEquationSource] = {}
    Cp_LIQ_src: Dict[str, ComponentEquationSource] = {}
    rho_LIQ_src: Dict[str, ComponentEquationSource] = {}
    EnFo_IG_298_src: Dict[str, Dict[str, Any]] = {}
    MW_src: Dict[str, Dict[str, Any]] = {}
    # ! properties
    EnFo_IG_298: np.ndarray = np.array([])
    EnFo_IG_298_comp: Dict[str, float] = {}
    MW: np.ndarray = np.array([])
    MW_comp: Dict[str, float] = {}

    def __init__(
        self,
        components: List[Component],
        source: Source,
        model_inputs: Dict[str, Any],
        reactor_inputs: BatchReactorOptions,
        reaction_rates: List[ReactionRateExpression],
        component_key: ComponentKey,
        component_refs: Dict[str, Any],
    ):
        """
        Initializes the ThermoModelSource instance with the provided components, source, component key, and component mapper.

        Parameters
        ----------
        components : List[Component]
            A list of Component objects representing the chemical components involved in the model source.
        source : Source
            A Source object containing information about the source of the data or equations used in the model source.
        component_key : ComponentKey
            A ComponentKey object that serves as a key for identifying and categorizing the components in the model source.
        component_mapper : Dict[str, Dict[ComponentKey, str]]
            A dictionary that maps component IDs to their corresponding component keys for different properties. The keys of the outer dictionary are property names, and the values are dictionaries where the keys are ComponentKey objects and the values are the corresponding component IDs in the model source.
        """
        # NOTE: Set attributes
        self.components = components
        self.source = source
        self.model_inputs = model_inputs
        self.reactor_inputs = reactor_inputs
        self.reaction_rates = reaction_rates
        self.component_key = component_key

        # ! component refs
        self.component_ids = component_refs['component_ids']
        self.component_formula_state = component_refs['component_formula_state']
        self.component_mapper = component_refs['component_mapper']

        # ! phase
        self.phase = reactor_inputs.phase

        # SECTION: Extract property equation sources
        if self.reactor_inputs.heat_transfer_mode == "non-isothermal":
            # check heat capacity mode
            if self.reactor_inputs.gas_heat_capacity_mode == "temperature-dependent":
                # NOTE: extract heat capacity equation source for the components from the model source
                self.Cp_IG_src: Dict[str, ComponentEquationSource] = self.prop_eq_src(
                    prop_name="Cp_IG"
                )

            # NOTE: Enthalpy of formation at 298 K for ideal gas
            # source
            self.EnFo_IG_298_src = self.prop_dt_src(
                component_ids=self.component_ids,
                prop_name="EnFo_IG"
            )

            # values in J/mol
            (
                self.EnFo_IG_298,
                self.EnFo_IG_298_comp
            ) = config_components_property(
                component_ids=self.component_ids,
                prop_source=self.EnFo_IG_298_src,
                unit_conversion_func=to_J_per_mol
            )

        if self.phase == "liquid":
            # MW source
            self.MW_src = self.prop_dt_src(
                component_ids=self.component_ids,
                prop_name="MW"
            )

            # ! values in g/mol
            (
                self.MW,
                self.MW_comp
            ) = config_components_property(
                component_ids=self.component_ids,
                prop_source=self.MW_src,
                unit_conversion_func=to_g_per_mol
            )

            # NOTE: density
            if self.reactor_inputs.liquid_density_mode == "temperature-dependent":
                # NOTE: extract density equation source for the components from the model source
                self.rho_LIQ_src: Dict[str, ComponentEquationSource] = self.prop_eq_src(
                    prop_name="rho_LIQ"
                )

            # NOTE: heat capacity
            if self.reactor_inputs.heat_transfer_mode == "non-isothermal":
                if self.reactor_inputs.liquid_heat_capacity_mode == "temperature-dependent":
                    # extract heat capacity equation source for the components from the model source
                    self.Cp_LIQ_src: Dict[str, ComponentEquationSource] = self.prop_eq_src(
                        prop_name="Cp_LIQ"
                    )

    # ! Extract property equation source for components

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
            component_key=cast(ComponentKey, self.component_key),
            component_mapper=self.component_mapper
        )
        # >> check
        if eq_src is None:
            logger.error("Failed to extract property equation for components.")
            return {}

        return eq_src

    # ! Extract property data source for components
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
