# import libs
import logging
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, cast
import pycuc
from pythermodb_settings.models import Component, Temperature, Pressure, ComponentKey, CustomProp, Volume
from pythermodb_settings.utils import set_component_id, set_feed_specification
from pyThermoLinkDB.thermo import Source
from pyThermoLinkDB.models.component_models import ComponentEquationSource
from scipy.integrate import solve_ivp
# ! locals
from ..utils.unit_tools import to_m3, to_Pa, to_K, to_W_per_m2_K, to_m2
from ..utils.tools import collect_keys
from ..models.br import BatchReactorOptions, BatchReactorResult
from ..models.rate_exp import ReactionRateExpression
from ..models.br import GasModel
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
        input_stream: Dict[str, Any],
        batch_reactor_options: BatchReactorOptions,
        reactor_inputs: Dict[str, Any],
        heat_transfer_options: HeatTransferOptions,
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
        model_inputs : Dict[str, Any]
            A dictionary of model inputs, where the keys are the names of the inputs and the values are the input values. This can include feed specifications, initial conditions, or any other relevant parameters needed for the simulations.
        reactor_inputs : BatchReactorOptions
            A BatchReactorOptions object containing the inputs for the batch reactor simulation, such as volume, heat transfer properties, etc.
        component_key : ComponentKey
            A ComponentKey object that serves as a key for identifying and categorizing the components in the batch reactor.
        """
        # NOTE: Set attributes
        self.components = components
        self.input_stream = input_stream
        self.batch_reactor_options = batch_reactor_options
        self.reactor_inputs = reactor_inputs
        self.heat_transfer_options = heat_transfer_options
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

        # SECTION: Process model configuration
        # lower case keys for easier access
        self.input_stream_keys = collect_keys(self.input_stream)
        # >> temperature
        # ! to Kelvin
        self.temperature: Temperature = self._config_temperature()
        self.temperature_value = self.temperature.value
        self._T0 = self.temperature_value

        # SECTION: component IDs and related properties
        self.component_num = len(components)

        # >> ids
        self.component_ids = [
            set_component_id(
                component=component,
                component_key=cast(ComponentKey, self.component_key)
            )
            for component in self.components
        ]

        # >>> formula-state
        self.component_formula_state = [
            set_component_id(
                component=component,
                component_key='Formula-State'
            )
            for component in self.components
        ]

        # >> index mapping
        self.component_id_to_index = {
            comp_id: idx for idx, comp_id in enumerate(self.component_ids)
        }

        # NOTE: mole fraction components
        self.mole_fractions = [
            c.mole_fraction for c in self.components
        ]

        # NOTE: state components
        self.states = [
            c.state for c in self.components
        ]

        # SECTION: heat transfer configuration
        # NOTE: heat transfer mode
        (
            self.temperature_fixed,
            self._T0,
        ) = self._config_heat_transfer_mode()

        # NOTE: heat exchange configuration
        (
            self.heat_exchange,
            self.jacket_temperature_value,
            self.heat_transfer_coefficient_value,
            self.heat_transfer_area_value,
        ) = self._config_heat_exchange()

    # SECTION: Model Inputs configuration
    # NOTE: temperature configuration [K]
    def _config_temperature(
            self,
    ):
        if "temperature" in self.input_stream_keys:
            temperature_: Temperature = self.input_stream["temperature"]
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

    # NOTE: pressure configuration [Pa]
    def _config_pressure(
            self,
    ):
        """Configure the pressure for the batch reactor based on the model inputs."""
        if "pressure" in self.input_stream_keys:
            pressure_: Pressure = self.input_stream["pressure"]
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

    # NOTE: reactor volume configuration [m3]
    def _config_reactor_volume(
            self,
    ):
        """Configure the reactor volume for the batch reactor based on the model inputs."""
        if self.reactor_inputs['reactor_volume'] is None:
            raise ValueError(
                "reactor_volume must be provided for constant volume mode."
            )

        # set reactor volume from model inputs
        reactor_volume = self.reactor_inputs['reactor_volume']

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
    def _config_heat_transfer_mode(
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
    def _config_heat_exchange(
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
