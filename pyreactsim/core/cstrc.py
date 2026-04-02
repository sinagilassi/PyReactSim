import logging
import numpy as np
import pycuc
from typing import Any, Dict, List, Optional, Tuple
from pythermodb_settings.models import Component, ComponentKey, Pressure, Temperature, Volume
# locals
from ..models.cstr import CSTRReactorOptions
from ..models.heat import HeatTransferOptions
from ..utils.tools import collect_keys
from ..utils.unit_tools import to_K, to_Pa, to_m2, to_m3, to_W, to_W_per_m2_K

# NOTE: logger setup
logger = logging.getLogger(__name__)


class CSTRReactorCore:
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
        self.components = components
        self.model_inputs = model_inputs
        self.cstr_reactor_options = cstr_reactor_options
        self.heat_transfer_options = heat_transfer_options
        self.component_refs = component_refs
        self.component_key = component_key

        self.phase = cstr_reactor_options.phase
        self.case = cstr_reactor_options.case
        self.gas_model = cstr_reactor_options.gas_model
        self.gas_heat_capacity_mode = cstr_reactor_options.gas_heat_capacity_mode
        self.liquid_heat_capacity_mode = cstr_reactor_options.liquid_heat_capacity_mode
        self.liquid_density_mode = cstr_reactor_options.liquid_density_mode

        self.heat_transfer_mode = heat_transfer_options.heat_transfer_mode
        self.jacket_temperature = heat_transfer_options.jacket_temperature
        self.heat_transfer_coefficient = heat_transfer_options.heat_transfer_coefficient
        self.heat_transfer_area = heat_transfer_options.heat_transfer_area
        self.heat_rate = heat_transfer_options.heat_rate

        self.model_inputs_keys = collect_keys(self.model_inputs)

        self.component_num = component_refs["component_num"]
        self.component_ids = component_refs["component_ids"]
        self.component_formula_state = component_refs["component_formula_state"]
        self.component_mapper = component_refs["component_mapper"]
        self.component_id_to_index = component_refs["component_id_to_index"]

        self.initial_temperature = self.config_initial_temperature()
        self.initial_temperature_value = self.initial_temperature.value
        self.feed_temperature = self.config_feed_temperature()
        self.feed_temperature_value = self.feed_temperature.value

        self.temperature_fixed, self._T0 = self.config_heat_transfer_mode()
        (
            self.heat_exchange,
            self.jacket_temperature_value,
            self.heat_transfer_coefficient_value,
            self.heat_transfer_area_value,
        ) = self.config_heat_exchange()
        self.heat_rate_value = self.config_heat_flux()

    def _to_mol_per_s(self, value: float, unit: str) -> float:
        try:
            return pycuc.convert_from_to(value=value, from_unit=unit, to_unit="mol/s")
        except Exception:
            # fallback to raw value for unit systems that are already mol/s but not
            # explicitly recognized by converter
            return float(value)

    def config_initial_temperature(self) -> Temperature:
        if "initial_temperature" not in self.model_inputs_keys:
            raise ValueError("initial_temperature must be provided in model_inputs.")

        temperature_: Temperature = self.model_inputs["initial_temperature"]
        temperature_value = to_K(temperature_.value, temperature_.unit)
        return Temperature(value=temperature_value, unit="K")

    def config_feed_temperature(self) -> Temperature:
        if "feed_temperature" not in self.model_inputs_keys:
            raise ValueError("feed_temperature must be provided in model_inputs.")

        temperature_: Temperature = self.model_inputs["feed_temperature"]
        temperature_value = to_K(temperature_.value, temperature_.unit)
        return Temperature(value=temperature_value, unit="K")

    def config_pressure(self) -> Pressure:
        if "pressure" not in self.model_inputs_keys:
            raise ValueError("pressure must be provided in model_inputs.")

        pressure_: Pressure = self.model_inputs["pressure"]
        pressure_value = to_Pa(pressure_.value, pressure_.unit)
        return Pressure(value=pressure_value, unit="Pa")

    def config_reactor_volume(self) -> Volume:
        if "reactor_volume" not in self.model_inputs_keys:
            raise ValueError("reactor_volume must be provided in model_inputs.")

        reactor_volume_: Volume = self.model_inputs["reactor_volume"]
        reactor_volume_value = to_m3(reactor_volume_.value, reactor_volume_.unit)
        return Volume(value=reactor_volume_value, unit="m3")

    def config_initial_mole(self) -> Tuple[Dict[str, float], np.ndarray]:
        if "initial_mole" not in self.model_inputs_keys:
            raise ValueError("initial_mole must be provided in model_inputs.")

        res_comp: Dict[str, float] = {}
        res: List[float] = []
        initial_mole = self.model_inputs["initial_mole"]
        initial_mole_keys = set(initial_mole.keys())

        for comp_id in self.component_formula_state:
            if comp_id not in initial_mole_keys:
                raise ValueError(
                    f"Missing initial_mole entry for component '{comp_id}'."
                )

            value_ = float(initial_mole[comp_id].value)
            res_comp[comp_id] = value_
            res.append(value_)

        return res_comp, np.array(res, dtype=float)

    def config_feed_mole_flow(self) -> Tuple[Dict[str, float], np.ndarray]:
        if "feed_mole_flow" not in self.model_inputs_keys:
            raise ValueError("feed_mole_flow must be provided in model_inputs.")

        res_comp: Dict[str, float] = {}
        res: List[float] = []
        feed_mole_flow = self.model_inputs["feed_mole_flow"]
        feed_mole_flow_keys = set(feed_mole_flow.keys())

        for comp_id in self.component_formula_state:
            if comp_id not in feed_mole_flow_keys:
                raise ValueError(
                    f"Missing feed_mole_flow entry for component '{comp_id}'."
                )

            value_ = self._to_mol_per_s(
                value=float(feed_mole_flow[comp_id].value),
                unit=feed_mole_flow[comp_id].unit
            )
            res_comp[comp_id] = value_
            res.append(value_)

        return res_comp, np.array(res, dtype=float)

    def config_outlet_mole_flow_total(self) -> float:
        if "outlet_mole_flow_total" not in self.model_inputs_keys:
            raise ValueError(
                "outlet_mole_flow_total must be provided in model_inputs."
            )

        outlet_mole_flow_total = self.model_inputs["outlet_mole_flow_total"]
        return self._to_mol_per_s(
            value=float(outlet_mole_flow_total.value),
            unit=outlet_mole_flow_total.unit
        )

    def config_heat_transfer_mode(self) -> Tuple[Optional[float], float]:
        if self.cstr_reactor_options.is_isothermal:
            if self.heat_transfer_mode != "isothermal":
                raise ValueError(
                    f"Case {self.case} is isothermal, but heat_transfer_mode is '{self.heat_transfer_mode}'."
                )
            return self.initial_temperature_value, self.initial_temperature_value

        if self.heat_transfer_mode != "non-isothermal":
            raise ValueError(
                f"Case {self.case} is non-isothermal, but heat_transfer_mode is '{self.heat_transfer_mode}'."
            )
        return None, self.initial_temperature_value

    def config_heat_exchange(self) -> Tuple[bool, float, float, float]:
        if (
            self.jacket_temperature is not None
            and self.heat_transfer_coefficient is not None
            and self.heat_transfer_area is not None
            and self.heat_transfer_mode == "non-isothermal"
        ):
            jacket_temperature_value = to_K(
                self.jacket_temperature.value,
                self.jacket_temperature.unit
            )
            heat_transfer_coefficient_value = to_W_per_m2_K(
                self.heat_transfer_coefficient.value,
                self.heat_transfer_coefficient.unit
            )
            heat_transfer_area_value = to_m2(
                self.heat_transfer_area.value,
                self.heat_transfer_area.unit
            )
            return (
                True,
                jacket_temperature_value,
                heat_transfer_coefficient_value,
                heat_transfer_area_value,
            )

        return False, 0.0, 0.0, 0.0

    def config_heat_flux(self) -> Optional[float]:
        if self.heat_rate is not None and self.heat_transfer_mode == "non-isothermal":
            return to_W(self.heat_rate.value, self.heat_rate.unit)
        return None
