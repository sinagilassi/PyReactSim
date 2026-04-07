import logging
from collections.abc import Mapping
from typing import Any, Dict, List, Tuple
from pythermodb_settings.models import Component, ComponentKey, Temperature, CustomProp, CustomProperty
# locals
from ..models.heat import HeatTransferOptions
from ..models.pbr import PBRReactorOptions
from ..utils.unit_tools import to_K, to_Pa, to_kg_per_m3
from .rc import ReactorCore

# NOTE: logger setup
logger = logging.getLogger(__name__)


class PBRReactorCore(ReactorCore):
    """
    PBR reactor core configuration helper.

    PBR follows PFR state structure and closures, with one key difference:
    reaction rates are interpreted per catalyst mass and converted by bulk
    catalyst density to reactor-volume basis.
    """

    def __init__(
        self,
        components: List[Component],
        model_inputs: Dict[str, Any],
        pbr_reactor_options: PBRReactorOptions,
        heat_transfer_options: HeatTransferOptions,
        component_refs: Dict[str, Any],
        component_key: ComponentKey,
    ):
        """
        Initialize PBR reactor core with validated/normalized configuration.

        Parameters
        ----------
        components : List[Component]
            Components participating in the reactor model.
        model_inputs : Dict[str, Any]
            User model inputs (inlet flows, inlet temperature, reactor volume,
            pressure closure inputs, and packed-bed bulk density).
        pbr_reactor_options : PBRReactorOptions
            PBR phase/operation/pressure/thermo options.
        heat_transfer_options : HeatTransferOptions
            Heat-transfer mode and optional jacket/constant-heat settings.
        component_refs : Dict[str, Any]
            Component lookup/index references.
        component_key : ComponentKey
            Key used for component mapping and stoichiometry indexing.
        """
        super().__init__(
            components=components,
            model_inputs=model_inputs,
            heat_transfer_options=heat_transfer_options,
            component_refs=component_refs,
            component_key=component_key,
        )

        # SECTION: reactor configuration
        self.phase = pbr_reactor_options.phase
        self.operation_mode = pbr_reactor_options.operation_mode
        self.pressure_mode = pbr_reactor_options.pressure_mode
        self.gas_model = pbr_reactor_options.gas_model
        self.gas_heat_capacity_mode = pbr_reactor_options.gas_heat_capacity_mode
        self.liquid_heat_capacity_mode = pbr_reactor_options.liquid_heat_capacity_mode
        self.liquid_density_mode = pbr_reactor_options.liquid_density_mode

        # SECTION: process model configuration
        self.temperature_inlet: Temperature = self.config_inlet_temperature()
        self.temperature_inlet_value = self.temperature_inlet.value
        self._T_in = self.temperature_inlet_value

        self.volume = self.config_reactor_volume()
        self.reactor_volume_value = self.volume.value

        _, self._F_in, self._F_in_total = self.config_inlet_mole_flows()
        self._P0 = self._config_pressure_initial()

        # NOTE: packed-bed specific conversion parameter [kg/m3]
        self._rho_B_value, self.rho_B = self.config_bulk_density()

        # SECTION: heat transfer configuration
        (
            self.heat_exchange,
            self.jacket_temperature_value,
            self.heat_transfer_coefficient_value,
            self.heat_transfer_area_value,
        ) = self.config_heat_exchange()
        self.heat_rate_value = self.config_heat_rate()

        # SECTION: mode flags
        self.is_constant_pressure = self.pressure_mode == "constant"
        self.is_non_isothermal = self.heat_transfer_mode == "non-isothermal"
        self.is_pressure_state_variable = self.pressure_mode == "state_variable"

    # NOTE: bulk density configuration
    def config_model(self):
        """
        Validate selected PBR option combinations.
        """
        if self.phase == "gas" and self.pressure_mode == "state_variable":
            raise NotImplementedError(
                "PBR pressure_mode='state_variable' is not implemented yet. "
                "Use pressure_mode='constant' or pressure_mode='shortcut'."
            )

        if (
            self.phase == "gas" and
            self.pressure_mode == "constant" and
            self.operation_mode == "constant_volume"
        ):
            raise ValueError(
                "Invalid gas PBR setup: operation_mode='constant_volume' is incompatible with pressure_mode='constant'."
            )

    # NOTE: inlet temperature configuration and normalization
    def config_inlet_temperature(self) -> Temperature:
        """
        Configure and normalize inlet temperature to Kelvin.

        Accepted input shapes
        ---------------------
        - Temperature model
        - Mapping with {'value', 'unit'}
        - numeric scalar (assumed Kelvin)
        """
        if "inlet_temperature" not in self.model_inputs_keys:
            raise ValueError(
                "inlet_temperature must be provided in model_inputs.")

        inlet_temperature = self.model_inputs["inlet_temperature"]

        if isinstance(inlet_temperature, Temperature):
            value = float(inlet_temperature.value)
            unit = inlet_temperature.unit
        elif isinstance(inlet_temperature, Mapping):
            if "value" not in inlet_temperature or "unit" not in inlet_temperature:
                raise ValueError(
                    "inlet_temperature mapping must contain 'value' and 'unit'."
                )
            value = float(inlet_temperature["value"])
            unit = str(inlet_temperature["unit"])
        else:
            value = float(inlet_temperature)
            unit = "K"

        unit = str(unit).strip()
        if not unit:
            raise ValueError(
                "inlet_temperature unit must be a non-empty string.")

        if unit.upper() != "K":
            value = to_K(value=value, unit=unit)

        return Temperature(value=value, unit="K")

    # NOTE: initial pressure configuration
    def _config_pressure_initial(self) -> float:
        """
        Configure initial pressure [Pa] for PBR initialization.

        Rules
        -----
        - gas phase:
            inlet pressure is required for supported closures
            (constant and shortcut)
        - liquid phase:
            pressure is optional and defaults to 0.0
        """
        if self.phase != "gas":
            if "inlet_pressure" in self.model_inputs_keys:
                pressure = self.model_inputs["inlet_pressure"]
                return to_Pa(value=float(pressure.value), unit=pressure.unit)
            return 0.0

        if "inlet_pressure" not in self.model_inputs_keys:
            raise ValueError(
                "inlet_pressure must be provided in model_inputs for gas-phase PBR."
            )

        pressure = self.model_inputs["inlet_pressure"]
        return to_Pa(value=float(pressure.value), unit=pressure.unit)

    # NOTE: Bulk density
    def config_bulk_density(self) -> Tuple[float, CustomProperty]:
        """
        Configure packed-bed bulk catalyst density without unit conversion.

        The value is used in the PBR-specific rate conversion:
        r_V = rho_B * r'
        """
        if "bulk_density" not in self.model_inputs_keys:
            # set to 1
            return 1.0, CustomProperty(
                value=1.0,
                unit="kg/m3",
                symbol="rho_B"
            )

        # set bulk density from model inputs and convert to kg/m3
        bulk_density = self.model_inputs["bulk_density"]
        rho_B_value = float(bulk_density.value)
        rho_B_unit = bulk_density.unit

        if rho_B_value <= 0.0:
            raise ValueError(
                "bulk_density must be greater than zero."
            )

        # set
        rho_B = CustomProperty(
            value=rho_B_value,
            unit=rho_B_unit,
            symbol="rho_B"
        )

        return rho_B_value, rho_B
