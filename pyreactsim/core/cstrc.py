import logging
import numpy as np
import pycuc
from typing import Any, Dict, List, Optional, Tuple
from pythermodb_settings.models import Component, ComponentKey, Pressure, Temperature, Volume
# locals
from ..models.cstr import CSTRReactorOptions
from ..models.heat import HeatTransferOptions
from ..utils.tools import collect_keys
from ..utils.unit_tools import to_K, to_Pa, to_m2, to_m3, to_W, to_W_per_m2_K, to_mol_per_s
from .rc import ReactorCore

# NOTE: logger setup
logger = logging.getLogger(__name__)


class CSTRReactorCore(ReactorCore):
    """
    CSTR reactor core configuration helper.

    This class validates and normalizes model inputs and heat-transfer settings
    used by gas-phase CSTR dynamic simulations.
    """

    def __init__(
        self,
        components: List[Component],
        model_inputs: Dict[str, Any],
        cstr_reactor_options: CSTRReactorOptions,
        heat_transfer_options: HeatTransferOptions,
        component_refs: Dict[str, Any],
        component_key: ComponentKey,
    ):
        """
        Initializes the CSTRReactorCore instance with the provided components, model inputs, reactor options, heat transfer options, component references, and component key.

        Parameters
        ----------
        components : List[Component]
            A list of Component objects representing the chemical components involved in the CSTR reactor simulation.
        model_inputs : Dict[str, Any]
            A dictionary of model inputs, where the keys are the names of the inputs and the values are the input values. This can include feed specifications, initial conditions, or any other relevant parameters needed for the simulations.
        cstr_reactor_options : CSTRReactorOptions
            A CSTRReactorOptions object containing the inputs for the CSTR reactor simulation, such as volume, heat transfer properties, etc.
        heat_transfer_options : HeatTransferOptions
            A HeatTransferOptions object containing the inputs for heat transfer in the CSTR reactor simulation.
        component_refs : Dict[str, Any]
            A dictionary of component references, which may include mappings, indices, or any other relevant information related to the components used in the simulation.
        component_key : ComponentKey
            A ComponentKey object representing the key to be used for the components in the CSTR reactor simulation.
        """
        # LINK: ReactorCore initialization
        super().__init__(
            components=components,
            model_inputs=model_inputs,
            heat_transfer_options=heat_transfer_options,
            component_refs=component_refs,
            component_key=component_key,
        )

        # SECTION: reactor configuration
        self.phase = cstr_reactor_options.phase
        self.gas_model = cstr_reactor_options.gas_model
        self.operation_mode = cstr_reactor_options.operation_mode
        self.gas_heat_capacity_mode = cstr_reactor_options.gas_heat_capacity_mode
        self.liquid_heat_capacity_mode = cstr_reactor_options.liquid_heat_capacity_mode
        self.liquid_density_mode = cstr_reactor_options.liquid_density_mode

        # SECTION: Process model configuration
        # >> temperature
        # ! to Kelvin
        self.temperature_initial: Temperature = self.config_initial_temperature()
        self.temperature_initial_value = self.temperature_initial.value
        self._T0 = self.temperature_initial_value

        # SECTION: heat transfer configuration
        # NOTE: heat transfer mode
        (
            self.temperature_fixed,
            self._T0,
        ) = self.config_heat_transfer_mode(
            temperature_value=self.temperature_initial_value
        )

        # NOTE: feed temperature [K]
        self.feed_temperature = self._T0

        # NOTE: heat exchange configuration
        (
            self.heat_exchange,
            self.jacket_temperature_value,
            self.heat_transfer_coefficient_value,
            self.heat_transfer_area_value,
        ) = self.config_heat_exchange()

        # NOTE: heat rate configuration
        self.heat_rate_value = self.config_heat_rate()

        # SECTION: Reactor configuration
        # NOTE: is constant pressure
        self.is_constant_pressure = True if self.operation_mode == 'constant_pressure' else False

        # NOTE: is constant volume
        self.is_constant_volume = True if self.operation_mode == 'constant_volume' else False

        # NOTE: holdup volume mode
        self.holdup_volume_mode = cstr_reactor_options.holdup_volume_mode if self.is_constant_pressure else None

        # NOTE: outlet flow mode
        self.outlet_flow_mode = cstr_reactor_options.outlet_flow_mode

    # SECTION: Main configuration methods
    # NOTE: initial temperature configuration
    # ! [K]
    def config_initial_temperature(self) -> Temperature:
        if "initial_temperature" not in self.model_inputs_keys:
            raise ValueError(
                "initial_temperature must be provided in model_inputs."
            )

        initial_temperature = self.model_inputs["initial_temperature"]

        value = to_K(
            value=float(initial_temperature.value),
            unit=initial_temperature.unit
        )

        return Temperature(value=value, unit="K")

    # NOTE: inlet temperature configuration
    # ! [K]
    def config_inlet_temperature(self) -> Temperature:
        if "inlet_temperature" not in self.model_inputs_keys:
            raise ValueError(
                "inlet_temperature must be provided in model_inputs."
            )

        inlet_temperature = self.model_inputs["inlet_temperature"]

        value = to_K(
            value=float(inlet_temperature.value),
            unit=inlet_temperature.unit
        )

        return Temperature(value=value, unit="K")

    # NOTE: initial mole configuration
    # ! [mol]
    def config_initial_mole(self) -> Tuple[Dict[str, float], np.ndarray]:
        if "initial_mole" not in self.model_inputs_keys:
            raise ValueError(
                "initial_mole must be provided in model_inputs."
            )

        initial_mole = self.model_inputs["initial_mole"]

        # res
        res = []
        res_comp = {}

        for comp_id in self.component_ids:
            if comp_id not in initial_mole:
                raise ValueError(
                    f"Missing initial mole entry for component '{comp_id}'."
                )

            value_ = to_mol_per_s(
                value=float(initial_mole[comp_id].value),
                unit=initial_mole[comp_id].unit
            )

            res_comp[comp_id] = value_
            res.append(value_)

        # convert to numpy array
        res = np.array(res, dtype=float)

        return res_comp, res

    # NOTE: outlet mole flow configuration
    # ! [mol/s]
    def config_outlet_mole_flow_total(self) -> float:
        # res0
        res = 0

        # check outlet flow mode
        if self.outlet_flow_mode == "fixed":
            if "outlet_mole_flow_total" not in self.model_inputs_keys:
                raise ValueError(
                    "outlet_mole_flow_total must be provided in model_inputs."
                )

            outlet_mole_flow_total = self.model_inputs["outlet_mole_flow_total"]

            res = to_mol_per_s(
                value=float(outlet_mole_flow_total.value),
                unit=outlet_mole_flow_total.unit
            )

            return res

        return 0
