# import libs
import logging
import numpy as np
from typing import List, Dict, Any, cast, Optional, Callable
from pythermodb_settings.models import Component, ComponentKey
from pyThermoLinkDB.thermo import Source
from pyThermoLinkDB.models import ModelSource
from pyThermoLinkDB.models.component_models import ComponentEquationSource
from pythermocalcdb.docs.thermo import build_hsg_properties
from pythermocalcdb.core import HSGProperties
from pyreactsim_core.models import ReactionRateExpression
# locals
from ..sources.interface import (
    ext_components_dt,
    ext_components_eq,
)
from ..models.br import BatchReactorOptions
from ..models.heat import HeatTransferOptions
from ..models.pbr import PBRReactorOptions
from ..models.pfr import PFRReactorOptions
from ..models.cstr import CSTRReactorOptions
from ..utils.tools import config_components_property
from .thermo_config import MODEL_SOURCE_ATTR_CONFIG, MODEL_SOURCE_CRITERIA
from .thermo_model_config import ThermoModelConfig

# NOTE: logger setup
logger = logging.getLogger(__name__)


class ThermoModelSource(ThermoModelConfig):
    """
    ThermoModelSource is a class that represents a model source for thermodynamic properties of components in a chemical reaction system. It is designed to extract and configure the necessary thermodynamic property equations and data from a given model source, which can then be used in reactor simulations. This class is designed to retrieve the following properties for the components in the system:

    - Heat capacity for ideal gas (Cp_IG)
    - Heat capacity for liquid (Cp_LIQ)
    - Density for liquid (rho_LIQ)
    - Enthalpy of formation at 298 K for ideal gas (EnFo_IG)
    - Molecular weight (MW)
    """

    # NOTE: Attributes
    # ! ideal gas heat capacity
    Cp_IG_src: Dict[str, ComponentEquationSource] = {}
    # ! liquid heat capacity
    Cp_LIQ_src: Dict[str, ComponentEquationSource] = {}
    # ! liquid density
    rho_LIQ_src: Dict[str, ComponentEquationSource] = {}
    # ! enthalpy of formation at 298 K for ideal gas
    EnFo_IG_298_src: Dict[str, Dict[str, Any]] = {}
    EnFo_IG_298: np.ndarray = np.array([])
    EnFo_IG_298_comp: Dict[str, float] = {}
    # ! molecular weight
    MW_src: Dict[str, Dict[str, Any]] = {}
    MW: np.ndarray = np.array([])
    MW_comp: Dict[str, float] = {}

    # NOTE: configurations
    attr_config = MODEL_SOURCE_ATTR_CONFIG
    # NOTE: criteria for model source
    criteria = MODEL_SOURCE_CRITERIA

    def __init__(
        self,
        components: List[Component],
        source: Source,
        model_source: ModelSource | None,
        thermo_inputs: Dict[str, Any],
        reactor_options: BatchReactorOptions | CSTRReactorOptions | PFRReactorOptions | PBRReactorOptions,
        heat_transfer_options: HeatTransferOptions,
        reaction_rates: List[ReactionRateExpression],
        component_refs: Dict[str, Any],
        component_key: ComponentKey,
    ):
        """
        Initializes the ThermoModelSource instance with the provided components, source, component key, and component mapper.

        Parameters
        ----------
        components : List[Component]
            A list of Component objects representing the chemical components involved in the model source.
        source : Source
            A Source object containing information about the source of the data or equations used in the model source.
        model_source : ModelSource | None
            A ModelSource object containing information about the model source, including its name, description, and other relevant details.
        thermo_inputs : Dict[str, Any]
            A dictionary of model inputs, where the keys are the names of the inputs and the values are the input values. This can include feed specifications, initial conditions, or any other relevant parameters needed for the simulations.
        reactor_options : BatchReactorOptions | CSTRReactorOptions | PFRReactorOptions | PBRReactorOptions
            A reactor options object containing phase and thermodynamic configuration.
        heat_transfer_options : HeatTransferOptions
            A HeatTransferOptions object containing the inputs for heat transfer in the batch reactor simulation.
        reaction_rates : List[ReactionRateExpression]
            A list of reaction rate expressions, where each expression is represented as a ReactionRateExpression object containing information about the reaction and its rate expression.
        component_refs : Dict[str, Any]
            A dictionary containing references for the components, which can include mappings of component IDs, formulas, and other relevant information for the components in the model source.
        component_key : ComponentKey
            A ComponentKey object that serves as a key for identifying and categorizing the components in the model source.
        """
        # LINK: init
        super().__init__(
            components=components,
            thermo_inputs=thermo_inputs,
            reactor_options=reactor_options,
            heat_transfer_options=heat_transfer_options,
            component_refs=component_refs,
            component_key=component_key,
        )

        # NOTE: Set attributes
        self.source = source
        self.model_source = model_source
        self.reaction_rates = reaction_rates

        # ! phase
        self.phase = reactor_options.phase

        # NOTE: launch property configuration
        self._launch_property_configuration()

    # SECTION: Extract property sources and configure properties
    def _launch_property_configuration(self):
        # NOTE: configure properties based on the defined methods and criteria
        for attr, config in self.attr_config.items():
            method = config["method"]
            prop_name = config["prop_name"]
            unit_conversion_func = config.get("unit_conversion_func")
            prop_criteria = self.criteria.get(prop_name, {})
            phase_criteria = config.get("phase", {})
            heat_transfer_mode_criteria = config.get("heat_transfer_mode", {})

            if method == "property-equation-source":
                # ! extract property equation source and set attribute
                eq_src = self._config_property_equation_source(
                    prop_name=prop_name,
                    prop_criteria=prop_criteria,
                    phase_criteria=phase_criteria,
                    heat_transfer_mode_criteria=heat_transfer_mode_criteria,
                )

                # >> check
                if eq_src is not None:
                    if hasattr(self, f"{attr}_src"):
                        setattr(self, f"{attr}_src", eq_src)
            elif method == "property-data-source":
                # ! extract property data source, configure property values and set attributes
                if unit_conversion_func is None:
                    raise ValueError(
                        f"unit_conversion_func must be provided for data-source attribute '{attr}' ({prop_name})."
                    )

                configured = self._config_property_data_source(
                    prop_name=prop_name,
                    unit_conversion_func=unit_conversion_func,
                    prop_criteria=prop_criteria,
                    phase_criteria=phase_criteria,
                    heat_transfer_mode_criteria=heat_transfer_mode_criteria,
                )

                # >> check
                if configured is None:
                    continue

                # >> unpack configured values and set attributes
                prop_src, prop_values, prop_comp = configured
                src_attr = f"{attr}_src"
                if hasattr(self, src_attr):
                    setattr(self, src_attr, prop_src)

                if hasattr(self, attr):
                    setattr(self, attr, prop_values)

                comp_attr = f"{attr}_comp"
                if hasattr(self, comp_attr):
                    setattr(self, comp_attr, prop_comp)
            else:
                logger.warning(
                    f"Unknown configuration method '{method}' for attribute '{attr}'. Skipping configuration."
                )

    def _config_property_equation_source(
        self,
        prop_name: str,
        prop_criteria: Dict[str, Dict[str, List[Any]]],
        phase_criteria: Dict[str, Dict[str, List[Any]]],
        heat_transfer_mode_criteria: Dict[str, Dict[str, List[Any]]],
    ) -> Dict[str, ComponentEquationSource] | None:
        if not self._should_configure(prop_criteria, phase_criteria, heat_transfer_mode_criteria):
            return None

        return self.prop_eq_src(prop_name=prop_name)

    def _config_property_data_source(
        self,
        prop_name: str,
        unit_conversion_func: Optional[Callable[[float, str], float]],
        prop_criteria: Dict[str, Dict[str, List[Any]]],
        phase_criteria: Dict[str, Dict[str, List[Any]]],
        heat_transfer_mode_criteria: Dict[str, Dict[str, List[Any]]],
    ) -> tuple[Dict[str, Dict[str, Any]], Any, Dict[str, float]] | None:
        if not self._should_configure(prop_criteria, phase_criteria, heat_transfer_mode_criteria):
            return None

        if unit_conversion_func is None:
            raise ValueError(
                f"unit_conversion_func cannot be None for data-source property '{prop_name}'."
            )

        prop_src = self.prop_dt_src(
            component_ids=self.component_ids,
            prop_name=prop_name,
        )

        prop_values, prop_comp = config_components_property(
            component_ids=self.component_ids,
            prop_source=prop_src,
            unit_conversion_func=unit_conversion_func,
        )

        return prop_src, prop_values, prop_comp

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

    # SECTION: Model source configurations
    def _get_args_units(
            self,
            eq_src: ComponentEquationSource
    ):
        # res
        res = {}

        # iterate over inputs
        for name, details in eq_src.arg_mappings.items():
            # get unit
            unit = details.get('unit', '')
            # symbol
            symbol = details.get('symbol', '')

            res[symbol] = unit

        return res

    def _get_return_unit(
            self,
            eq_src: ComponentEquationSource,
            symbol: str
    ) -> str:
        # res
        res = {}

        # iterate over returns
        for name, details in eq_src.returns.items():
            # get unit
            unit = details.get('unit', '')
            # symbol
            symbol = details.get('symbol', '')

            res[symbol] = unit

        # check
        if symbol not in res:
            logger.warning(
                f"Symbol {symbol} not found in equation source returns. Returning empty string for unit.")
            return ""

        return res[symbol]

    def _get_inputs(
            self,
            eq_src: ComponentEquationSource,
            except_args: Optional[List[str]] = None
    ):
        # res
        res = {}

        # iterate over inputs
        for name, details in eq_src.inputs.items():

            # get symbol
            symbol = details.get('symbol', '')

            if except_args is not None and symbol in except_args:
                continue

            # get value from component source
            value = details.get('value', None)

            res[symbol] = value

        return res

    # SECTION: HSG properties
    def _config_components_hsg_properties(
            self,
            temperature: float
    ) -> Dict[str, HSGProperties]:
        # NOTE: check model source
        if self.model_source is None:
            logger.warning(
                "No model source provided. Cannot build HSG properties without a valid model source."
            )
            return {}

        # build properties
        component_hsg_properties = {}

        for i, comp_id in enumerate(self.component_formula_state):
            # build
            hsg_props = build_hsg_properties(
                component=self.components[i],
                model_source=self.model_source
            )

            # >> check
            if hsg_props is None:
                logger.error(
                    f"Failed to build HSG properties for component {comp_id}.")
                raise ValueError(
                    f"Failed to build HSG properties for component {comp_id}.")

            component_hsg_properties[comp_id] = hsg_props

        return component_hsg_properties
