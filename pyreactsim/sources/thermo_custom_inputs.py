# import libs
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from pythermodb_settings.models import Component, CustomProp, ComponentKey
# locals
from ..models.br import BatchReactorOptions
from ..models.cstr import CSTRReactorOptions
from ..models.pfr import PFRReactorOptions
from ..models.pbr import PBRReactorOptions
from ..models.heat import HeatTransferOptions
from ..utils.tools import config_components_property
from .thermo_source_config import ThermoSourceConfig
from .thermo_config import CUSTOM_INPUTS_ATTR_CONFIG, CUSTOM_INPUTS_CRITERIA

# NOTE: logger
logger = logging.getLogger(__name__)


class ThermoCustomInputs(ThermoSourceConfig):
    """
    ThermoCustomInputs is a class that encapsulates the inputs required for configuring the thermodynamic properties in the reactor models.
    This class is designed to retrieve the following properties for the components in the system:

    - Ideal gas heat capacity (Cp_IG)
    - Liquid heat capacity (Cp_LIQ)
    - Liquid density (rho_LIQ)
    - Ideal gas formation enthalpy at 298 K (EnFo_IG)
    - Molecular weight (MW)
    """
    # NOTE: Attributes
    # ?? properties defined for each component
    # ! heat capacity of ideal gas
    Cp_IG_src: Dict[str, Dict[str, Any]] = {}
    Cp_IG: np.ndarray = np.array([])
    Cp_IG_comp: Dict[str, float] = {}
    # ! heat capacity of liquid
    Cp_LIQ_src: Dict[str, Dict[str, Any]] = {}
    Cp_LIQ: np.ndarray = np.array([])
    Cp_LIQ_comp: Dict[str, float] = {}
    # ! liquid density
    rho_LIQ_src: Dict[str, Dict[str, Any]] = {}
    rho_LIQ: np.ndarray = np.array([])
    rho_LIQ_comp: Dict[str, float] = {}
    # ! ideal gas formation enthalpy at 298 K
    EnFo_IG_298_src: Dict[str, Dict[str, Any]] = {}
    EnFo_IG_298: np.ndarray = np.array([])
    EnFo_IG_298_comp: Dict[str, float] = {}
    # ! molecular weight
    MW_src: Dict[str, Dict[str, Any]] = {}
    MW: np.ndarray = np.array([])
    MW_comp: Dict[str, float] = {}

    # ?? properties defined as constants (not component-specific)
    # ! enthalpy of reaction at T
    dH_rxn: Optional[Dict[str, Dict[str, Any]]] = None
    # ! mixture liquid density
    rho_LIQ_MIX: Optional[CustomProp] = None
    # ! total heat capacity of gas mixture
    Cp_IG_MIX_TOTAL: Optional[CustomProp] = None
    # ! total heat capacity of liquid mixture
    Cp_LIQ_MIX_TOTAL: Optional[CustomProp] = None
    # ! volumetric heat capacity of liquid mixture
    Cp_LIQ_MIX_VOLUMETRIC: Optional[CustomProp] = None

    # NOTE: configurations
    attr_config = CUSTOM_INPUTS_ATTR_CONFIG
    # NOTE: criteria for model inputs
    criteria = CUSTOM_INPUTS_CRITERIA

    def __init__(
        self,
        components: List[Component],
        custom_inputs: Dict[str, Any] | None,
        reactor_options: BatchReactorOptions | CSTRReactorOptions | PFRReactorOptions | PBRReactorOptions,
        heat_transfer_options: HeatTransferOptions,
        component_refs: Dict[str, Any],
        component_key: ComponentKey,
    ):
        """
        Initializes the ThermoInputs instance with the provided components, source, model inputs, reactor inputs, reaction rates, and component key.

        Parameters
        ----------
        components : List[Component]
            A list of Component objects representing the chemical components involved in the model source.
        thermo_inputs : Dict[str, Any]
            A dictionary of model inputs, where the keys are the names of the inputs and the values are the input values. This can include feed specifications, initial conditions, or any other relevant parameters needed for the simulations.
        reactor_inputs : BatchReactorOptions | CSTRReactorOptions | PFRReactorOptions | PBRReactorOptions
            A reactor options object containing phase and thermodynamic configuration.
        heat_transfer_options : HeatTransferOptions
            A HeatTransferOptions object containing the inputs for heat transfer in the batch reactor simulation.
        component_refs : Dict[str, Any]
            A dictionary of component references, where the keys are the names of the references and the values are the reference values or objects.
        component_key : ComponentKey
            A ComponentKey object that serves as a key for identifying and categorizing the components in the model source.
        """
        # LINK: init
        super().__init__(
            components=components,
            custom_inputs=custom_inputs,
            reactor_options=reactor_options,
            heat_transfer_options=heat_transfer_options,
            component_refs=component_refs,
            component_key=component_key,
        )

        # SECTION: Reactor configuration
        # ! gas heat capacity mode
        self.gas_heat_capacity_mode = reactor_options.gas_heat_capacity_mode
        # ! liquid heat capacity mode
        self.liquid_heat_capacity_mode = reactor_options.liquid_heat_capacity_mode
        # ! density mode
        self.liquid_density_mode = reactor_options.liquid_density_mode
        # ! phase
        self.phase = reactor_options.phase
        # ! operation mode
        self.operation_mode = reactor_options.operation_mode

        # SECTION: heat transfer options
        # ! heat transfer mode
        self.heat_transfer_mode = heat_transfer_options.heat_transfer_mode

        # custom inputs keys
        self.custom_inputs_keys = list(
            self.custom_inputs.keys()) if self.custom_inputs is not None else []

        # NOTE: launch property configuration
        self._launch_property_configuration()

    # SECTION: Custom input validation and property configuration methods
    def _check_custom_inputs(self) -> bool:
        if self.custom_inputs is None:
            logger.info(
                "No custom inputs provided. Skipping thermo property configuration.")
            return False
        return True

    # SECTION: Extract property sources and configure properties
    def _launch_property_configuration(self):
        # NOTE: check custom inputs
        if not self._check_custom_inputs():
            return

        # NOTE: configure properties based on the defined methods and criteria
        for attr, config in self.attr_config.items():
            # ! config details
            method = config["method"]
            prop_name = config["prop_name"]
            unit_conversion_func = config["unit_conversion_func"]
            expected_unit = config["expected_unit"]
            strict_unit_check = config["strict_unit_check"]
            phase_criteria = config.get("phase", {})
            heat_transfer_mode_criteria = config.get("heat_transfer_mode", {})
            # ! criteria
            prop_criteria = self.criteria.get(attr, {})

            if method == "property-source":
                # ! property source configuration
                # ?? property contains component-wise source, values, and component mapping
                configured = self._config_property_source(
                    prop_name=prop_name,
                    unit_conversion_func=unit_conversion_func,
                    expected_unit=expected_unit,
                    prop_criteria=prop_criteria,
                    phase_criteria=phase_criteria,
                    heat_transfer_mode_criteria=heat_transfer_mode_criteria,
                    strict_unit_check=strict_unit_check,
                )

                # >> check
                if configured is None:
                    continue

                # >>> unpack configured values
                prop_src, prop_values, prop_comp = configured
                setattr(self, attr, prop_values)
                if hasattr(self, f"{attr}_comp"):
                    setattr(self, f"{attr}_comp", prop_comp)
                if hasattr(self, f"{attr}_src"):
                    setattr(self, f"{attr}_src", prop_src)
            elif method == "property-constant":
                # ! property constant configuration
                # ?? property contains a constant value for the whole system (not component-specific)
                configured_value = self._config_property_constant(
                    prop_name=prop_name,
                    unit_conversion_func=unit_conversion_func,
                    expected_unit=expected_unit,
                    prop_criteria=prop_criteria,
                    phase_criteria=phase_criteria,
                    heat_transfer_mode_criteria=heat_transfer_mode_criteria,
                    strict_unit_check=strict_unit_check,
                )

                # >> check
                if configured_value is None:
                    continue

                # >> set attribute
                setattr(self, attr, configured_value)
            elif method == "property-constants":
                # ! property configuration
                # ?? property independent of component mapping
                configured_value = self._config_property(
                    prop_name=prop_name,
                    unit_conversion_func=unit_conversion_func,
                    expected_unit=expected_unit,
                    prop_criteria=prop_criteria,
                    phase_criteria=phase_criteria,
                    heat_transfer_mode_criteria=heat_transfer_mode_criteria,
                    strict_unit_check=strict_unit_check,
                )

                # >> check
                if configured_value is None:
                    continue

                # >> set attribute
                setattr(self, attr, configured_value)
            else:
                logger.warning(
                    f"Unknown configuration method '{method}' for attribute '{attr}'. Skipping configuration."
                )

    # SECTION: Universal methods for property retrieval and configuration
    # NOTE: configure property values and component mapping based on criteria matching
    def _config_property(
        self,
        prop_name: str,
        unit_conversion_func: Callable[[float, str], float],
        expected_unit: str,
        prop_criteria: Dict[str, Dict[str, List[Any]]],
        phase_criteria: Dict[str, Dict[str, List[Any]]],
        heat_transfer_mode_criteria: Dict[str, Dict[str, List[Any]]],
        strict_unit_check: bool = True,
    ) -> Dict[str, float] | None:
        # ! check custom inputs
        if not self._check_custom_inputs():
            return None

        # NOTE: check criteria for property configuration
        if not self._should_configure(prop_criteria, phase_criteria, heat_transfer_mode_criteria):
            return None

        # NOTE: if all checks pass, proceed to configure the property
        if prop_name not in self.custom_inputs_keys:
            raise ValueError(
                f"{prop_name} must be provided in model_inputs for {prop_name} configuration."
            )

        # get property source
        if self.custom_inputs is None:
            return None

        prop_: Dict[str, CustomProp] = self.custom_inputs[prop_name]

        # set property values and component mapping
        # >> component-wise property values
        prop_comp = {}

        # NOTE: prepare array and component mapping dict for property source
        # iterate through components
        for id in prop_.keys():
            # > value
            value = prop_[id].value
            # > unit
            unit = prop_[id].unit

            # >> check expected unit
            if strict_unit_check:
                if unit != expected_unit:
                    # convert to expected unit
                    value = unit_conversion_func(
                        value,
                        unit
                    )

            # add to component mapping
            prop_comp[id] = value

        return prop_comp

    # NOTE: configure property source dict based on criteria matching
    def _config_property_source(
        self,
        prop_name: str,
        unit_conversion_func: Callable[[float, str], float],
        expected_unit: str,
        prop_criteria: Dict[str, Dict[str, List[Any]]],
        phase_criteria: Dict[str, Dict[str, List[Any]]],
        heat_transfer_mode_criteria: Dict[str, Dict[str, List[Any]]],
        strict_unit_check: bool = True,
    ) -> Tuple[Dict[str, Dict[str, Any]], Any, Dict[str, float]] | None:
        # ! check custom inputs
        if not self._check_custom_inputs():
            return None

        # NOTE: check criteria for property configuration
        if not self._should_configure(prop_criteria, phase_criteria, heat_transfer_mode_criteria):
            return None

        # NOTE: if all checks pass, proceed to configure the property
        if prop_name not in self.custom_inputs_keys:
            raise ValueError(
                f"{prop_name} must be provided in model_inputs for {prop_name} configuration."
            )

        # get property source
        if self.custom_inputs is None:
            return None

        # get property source
        prop_: Dict[str, CustomProp] = self.custom_inputs[prop_name]

        # init property source dict
        prop_src = {}

        # iterate through components
        # ! component_ids constructed based on component key (default is name-formula)
        for formula_state, component_id in zip(self.component_formula_state, self.component_ids):
            if formula_state in prop_.keys():
                # value
                value = prop_[formula_state].value
                # unit
                unit = prop_[formula_state].unit

                # add to property source dict with name-formula key
                prop_src[component_id] = {
                    "value": value,
                    "unit": unit,
                }
            else:
                raise ValueError(
                    f"{prop_name} value for component '{formula_state}' not found in model_inputs."
                )

        # NOTE: prepare array and component mapping dict for property source
        prop_values, prop_comp = config_components_property(
            component_ids=self.component_ids,
            prop_source=prop_src,
            unit_conversion_func=unit_conversion_func,
        )

        return prop_src, prop_values, prop_comp

    # NOTE: configure property constant based on criteria matching
    def _config_property_constant(
        self,
        prop_name: str,
        unit_conversion_func: Callable[[float, str], float],
        expected_unit: str,
        prop_criteria: Dict[str, Dict[str, List[Any]]],
        phase_criteria: Dict[str, Dict[str, List[Any]]],
        heat_transfer_mode_criteria: Dict[str, Dict[str, List[Any]]],
        strict_unit_check: bool = True,
    ) -> CustomProp | None:
        # ! check custom inputs
        if not self._check_custom_inputs():
            return None

        # NOTE: check criteria for property configuration
        if not self._should_configure(prop_criteria, phase_criteria, heat_transfer_mode_criteria):
            return None

        # NOTE: if all checks pass, proceed to configure the property
        if prop_name not in self.custom_inputs_keys:
            raise ValueError(
                f"{prop_name} must be provided in model_inputs for {prop_name} configuration."
            )

        # get property source
        if self.custom_inputs is None:
            return None

        # set
        prop_: CustomProp = self.custom_inputs[prop_name]

        # check unit
        if strict_unit_check:
            if prop_.unit != expected_unit:
                # convert to expected unit
                value = unit_conversion_func(
                    prop_.value,
                    prop_.unit
                )
            else:
                value = prop_.value
        else:
            value = prop_.value

        # create property constant
        prop_constant = CustomProp(
            value=value,
            unit=expected_unit,
        )

        return prop_constant
