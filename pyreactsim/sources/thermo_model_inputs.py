# import libs
import logging
import numpy as np
from typing import Dict, List, Any, Optional
from pythermodb_settings.models import Component, CustomProp, ComponentKey
# locals
from ..models.br import BatchReactorOptions
from ..models.cstr import CSTRReactorOptions
from ..models.pfr import PFRReactorOptions
from ..models.pbr import PBRReactorOptions
from ..models.heat import HeatTransferOptions
from .thermo_model_config import ThermoModelConfig
from .thermo_config import MODEL_INPUTS_ATTR_CONFIG, MODEL_INPUTS_CRITERIA

# NOTE: logger
logger = logging.getLogger(__name__)


class ThermoModelInputs(ThermoModelConfig):
    """
    ThermoModelInputs is a class that encapsulates the inputs required for configuring the thermodynamic properties in the reactor models.
    This class is designed to retrieve the following properties for the components in the system:

    - Ideal gas heat capacity (Cp_IG)
    - Liquid heat capacity (Cp_LIQ)
    - Liquid density (rho_LIQ)
    - Ideal gas formation enthalpy at 298 K (EnFo_IG)
    - Molecular weight (MW)
    """
    # NOTE: Attributes
    # ! heat capacity of ideal gas
    Cp_IG: np.ndarray = np.array([])
    Cp_IG_comp: Dict[str, float] = {}
    # ! heat capacity of liquid
    Cp_LIQ: np.ndarray = np.array([])
    Cp_LIQ_comp: Dict[str, float] = {}
    # ! liquid density
    rho_LIQ: np.ndarray = np.array([])
    rho_LIQ_comp: Dict[str, float] = {}
    # ! mixture liquid density
    rho_LIQ_MIX: Optional[CustomProp] = None
    # ! ideal gas formation enthalpy at 298 K
    EnFo_IG_298_src: Dict[str, Dict[str, Any]] = {}
    EnFo_IG_298: np.ndarray = np.array([])
    EnFo_IG_298_comp: Dict[str, float] = {}
    # ! molecular weight
    MW_src: Dict[str, Dict[str, Any]] = {}
    MW: np.ndarray = np.array([])
    MW_comp: Dict[str, float] = {}
    # ! enthalpy of reaction
    dH_rxn_src: Optional[Dict[str, Dict[str, Any]]] = None
    # ! total heat capacity of gas mixture
    Cp_IG_MIX_TOTAL: Optional[CustomProp] = None
    # ! total heat capacity of liquid mixture
    Cp_LIQ_MIX_TOTAL: Optional[CustomProp] = None
    # ! volumetric heat capacity of liquid mixture
    Cp_LIQ_MIX_VOLUMETRIC: Optional[CustomProp] = None

    # NOTE: configurations
    attr_config = MODEL_INPUTS_ATTR_CONFIG
    # NOTE: criteria for model inputs
    criteria = MODEL_INPUTS_CRITERIA

    def __init__(
        self,
        components: List[Component],
        thermo_inputs: Dict[str, Any],
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
            thermo_inputs=thermo_inputs,
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

        # NOTE: launch property configuration
        self._launch_property_configuration()

    # SECTION: Extract property sources and configure properties
    def _launch_property_configuration(self):
        # NOTE: configure properties based on the defined methods and criteria
        for attr, config in self.attr_config.items():
            method = config["method"]
            prop_name = config["prop_name"]
            unit_conversion_func = config["unit_conversion_func"]
            expected_unit = config["expected_unit"]
            strict_unit_check = config["strict_unit_check"]
            prop_criteria = self.criteria.get(prop_name, {})
            phase_criteria = config.get("phase", {})
            heat_transfer_mode_criteria = config.get("heat_transfer_mode", {})

            if method == "property-source":
                # ! property source configuration
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
            elif method == "property":
                # ! property configuration
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
