# import libs
import logging
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from pythermodb_settings.models import Component, Temperature, Pressure, ComponentKey, Volume
# ! locals
from ..utils.unit_tools import to_m3, to_Pa, to_K, to_W_per_m2_K, to_m2, to_W
from ..utils.tools import collect_keys
from ..models.br import BatchReactorOptions
from ..models.heat import HeatTransferOptions

# NOTE: logger setup
logger = logging.getLogger(__name__)


class BatchReactorCore:
    """
    Batch Reactor Core (BRC) class for simulating chemical reactions in a batch reactor setup. This class encapsulates the components, source, and component key information necessary for performing simulations and analyses related to batch reactors.
    """

    def __init__(
        self,
        components: List[Component],
        model_inputs: Dict[str, Any],
        batch_reactor_options: BatchReactorOptions,
        heat_transfer_options: HeatTransferOptions,
        component_refs: Dict[str, Any],
        component_key: ComponentKey,
    ):
        """
        Initializes the BatchReactor instance with the provided components, source, and component key.

        Parameters
        ----------
        components : List[Component]
            A list of Component objects representing the chemical components involved in the batch reactor.
        model_inputs : Dict[str, Any]
            A dictionary of model inputs, where the keys are the names of the inputs and the values are the input values. This can include feed specifications, initial conditions, or any other relevant parameters needed for the simulations.
        batch_reactor_options : BatchReactorOptions
            A BatchReactorOptions object containing the options for configuring the batch reactor, such as phase, operation mode, gas model, and heat capacity modes.
        heat_transfer_options : HeatTransferOptions
            A HeatTransferOptions object containing the options for configuring the heat transfer in the batch reactor, such as heat transfer mode, jacket temperature, heat transfer coefficient, and heat transfer area.
        component_refs : Dict[str, Any]
            A dictionary containing references for the components, which can include component IDs, formula states, mappers, and any other relevant information needed for the simulations.
        component_key : ComponentKey
            A ComponentKey object that serves as a key for identifying and categorizing the components in the batch reactor.
        """
        # NOTE: Set attributes
        self.components = components
        self.model_inputs = model_inputs
        self.batch_reactor_options = batch_reactor_options
        self.heat_transfer_options = heat_transfer_options
        self.component_refs = component_refs
        self.component_key = component_key

        # SECTION: reactor configuration
        # >> extract
        self.phase = batch_reactor_options.phase
        self.gas_model = batch_reactor_options.gas_model
        self.operation_mode = batch_reactor_options.operation_mode
        self.gas_heat_capacity_mode = batch_reactor_options.gas_heat_capacity_mode
        self.liquid_heat_capacity_mode = batch_reactor_options.liquid_heat_capacity_mode
        self.liquid_density_mode = batch_reactor_options.liquid_density_mode

        # SECTION: Heat exchange configuration
        self.heat_transfer_mode = heat_transfer_options.heat_transfer_mode
        self.jacket_temperature = heat_transfer_options.jacket_temperature
        self.heat_transfer_coefficient = heat_transfer_options.heat_transfer_coefficient
        self.heat_transfer_area = heat_transfer_options.heat_transfer_area
        self.heat_rate = heat_transfer_options.heat_rate

        # SECTION: Process model configuration
        # lower case keys for easier access
        self.model_inputs_keys = collect_keys(self.model_inputs)
        # >> temperature
        # ! to Kelvin
        self.temperature: Temperature = self.config_temperature()
        self.temperature_value = self.temperature.value
        self._T0 = self.temperature_value

        # SECTION: component IDs and related properties
        # ! component refs
        self.component_num = component_refs['component_num']
        self.component_ids = component_refs['component_ids']
        self.component_formula_state = component_refs['component_formula_state']
        self.component_mapper = component_refs['component_mapper']
        self.component_id_to_index = component_refs['component_id_to_index']

        # SECTION: heat transfer configuration
        # NOTE: heat transfer mode
        (
            self.temperature_fixed,
            self._T0,
        ) = self.config_heat_transfer_mode()

        # NOTE: heat exchange configuration
        (
            self.heat_exchange,
            self.jacket_temperature_value,
            self.heat_transfer_coefficient_value,
            self.heat_transfer_area_value,
        ) = self.config_heat_exchange()

        # NOTE: heat rate configuration
        self.heat_rate_value = self.config_heat_flux()

    # SECTION: Model Inputs configuration
    # NOTE: temperature configuration
    # ! [K]

    def config_temperature(
            self,
    ):
        if "temperature" in self.model_inputs_keys:
            temperature_: Temperature = self.model_inputs["temperature"]
            temperature_value = to_K(
                temperature_.value,
                temperature_.unit
            )
            # >> update
            temperature = Temperature(
                value=temperature_value,
                unit="K"
            )

            return temperature
        else:
            raise ValueError("Temperature must be provided in model_inputs.")

    # NOTE: pressure configuration
    # ! [Pa]
    def config_pressure(
            self,
    ):
        """Configure the pressure for the batch reactor based on the model inputs."""
        if "pressure" in self.model_inputs_keys:
            pressure_: Pressure = self.model_inputs["pressure"]
            pressure_value = to_Pa(
                pressure_.value,
                pressure_.unit
            )
            # >> update
            pressure = Pressure(
                value=pressure_value,
                unit="Pa"
            )

            return pressure
        else:
            raise ValueError("Pressure must be provided in model_inputs.")

    # NOTE: reactor volume configuration
    # ! [m3]
    def config_reactor_volume(
            self,
    ):
        """Configure the reactor volume for the batch reactor based on the model inputs."""
        if 'reactor_volume' not in self.model_inputs_keys:
            raise ValueError(
                "Reactor volume must be provided in model_inputs."
            )

        # set reactor volume from model inputs
        reactor_volume = self.model_inputs['reactor_volume']

        if reactor_volume is not None:
            reactor_volume_value = to_m3(
                reactor_volume.value,
                reactor_volume.unit
            )
            # >> update
            reactor_volume = Volume(
                value=reactor_volume_value,
                unit="m3"
            )

            return reactor_volume
        else:
            raise ValueError(
                "Reactor volume must be provided for constant volume mode.")

    # NOTE: heat transfer mode configuration
    def config_heat_transfer_mode(
            self,
    ) -> Tuple[Optional[float], float]:
        """
        Configure the temperature fixed and initial temperature based on the heat transfer mode.
        """
        if self.heat_transfer_mode == "isothermal":
            return self.temperature_value, self.temperature_value
        elif self.heat_transfer_mode == "non-isothermal":
            return None, self.temperature_value
        else:
            raise ValueError(
                "Invalid heat_transfer_mode. Must be 'isothermal' or 'non-isothermal'."
            )

    # NOTE: heat exchange configuration
    def config_heat_exchange(
            self,
    ) -> Tuple[bool, float, float, float]:
        """Configure the heat exchange parameters for the batch reactor based on the model inputs."""
        if (
            self.jacket_temperature is not None and
            self.heat_transfer_coefficient is not None and
            self.heat_transfer_area is not None and
            self.heat_transfer_mode == 'non-isothermal'
        ):
            # >> conversions for heat exchange parameters
            self.jacket_temperature = Temperature(
                value=to_K(
                    self.jacket_temperature.value,
                    self.jacket_temperature.unit
                ),
                unit="K"
            )
            # ! [K]
            jacket_temperature_value = self.jacket_temperature.value

            # >> heat transfer coefficient
            # ! [W/m2.K]
            heat_transfer_coefficient_value = to_W_per_m2_K(
                self.heat_transfer_coefficient.value,
                self.heat_transfer_coefficient.unit
            )

            # >> heat transfer area
            # ! [m2]
            heat_transfer_area_value = to_m2(
                self.heat_transfer_area.value,
                self.heat_transfer_area.unit
            )

            return True, jacket_temperature_value, heat_transfer_coefficient_value, heat_transfer_area_value
        else:
            return False, 0.0, 0.0, 0.0

    def config_mole(
            self
    ):
        """Configure the mole feed for the batch reactor based on the model inputs."""
        # check
        if "mole" not in self.model_inputs_keys:
            raise ValueError("Mole feed must be provided in model_inputs.")

        # NOTE: res
        res = []
        res_comp = {}

        # mole feed value dict
        mole_feed = self.model_inputs["mole"]
        mole_feed_keys = list(mole_feed.keys())

        # iterate through component IDs and extract mole feed values
        for comp_id in self.component_formula_state:
            # check if component ID is in mole feed
            if comp_id in mole_feed_keys:
                value_ = mole_feed[comp_id].value
                unit_ = mole_feed[comp_id].unit

                # add to res
                res.append(value_)
                res_comp[comp_id] = value_

        # convert to array
        res = np.array(res, dtype=float)

        return res_comp, res

    def config_heat_flux(
            self,
    ) -> Optional[float]:
        """Configure the heat flux for the batch reactor based on the model inputs."""
        if (
            self.heat_rate is not None and
            self.heat_transfer_mode == 'non-isothermal'
        ):
            # >> conversion for heat flux
            # ! [W/s]
            heat_flux_value = to_W(
                self.heat_rate.value,
                self.heat_rate.unit
            )

            return heat_flux_value
        else:
            return None
