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
from .rc import ReactorCore

# NOTE: logger setup
logger = logging.getLogger(__name__)


class BatchReactorCore(ReactorCore):
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
        # LINK: ReactorCore initialization
        super().__init__(
            components=components,
            model_inputs=model_inputs,
            heat_transfer_options=heat_transfer_options,
            component_refs=component_refs,
            component_key=component_key,
        )

        # SECTION: reactor configuration
        # >> extract
        self.phase = batch_reactor_options.phase
        self.gas_model = batch_reactor_options.gas_model
        self.operation_mode = batch_reactor_options.operation_mode
        self.gas_heat_capacity_mode = batch_reactor_options.gas_heat_capacity_mode
        self.liquid_heat_capacity_mode = batch_reactor_options.liquid_heat_capacity_mode
        self.liquid_density_mode = batch_reactor_options.liquid_density_mode
        self.use_gas_mixture_total_heat_capacity = batch_reactor_options.use_gas_mixture_total_heat_capacity
        self.use_liquid_mixture_volumetric_heat_capacity = batch_reactor_options.use_liquid_mixture_volumetric_heat_capacity
        self.reaction_enthalpy_mode = batch_reactor_options.reaction_enthalpy_mode

        # SECTION: Process model configuration
        # >> temperature
        # ! to Kelvin
        self.temperature: Temperature = self.config_temperature()
        self.temperature_value = self.temperature.value
        self._T0 = self.temperature_value

        # SECTION: heat transfer configuration
        # NOTE: heat transfer mode
        (
            self.temperature_fixed,
            self._T0,
        ) = self.config_heat_transfer_mode(
            temperature_value=self.temperature_value
        )

        # NOTE: heat exchange configuration
        (
            self.heat_exchange,
            self.jacket_temperature_value,
            self.heat_transfer_coefficient_value,
            self.heat_transfer_area_value,
        ) = self.config_heat_exchange()

        # NOTE: heat rate configuration
        self.heat_rate_value = self.config_heat_rate()
