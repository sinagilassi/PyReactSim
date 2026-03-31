# import libs
import logging
import numpy as np
from typing import Dict, List, Tuple, Any
from pythermodb_settings.models import Component, Temperature, Pressure, CustomProperty, CustomProp, ComponentKey
from pyThermoLinkDB.thermo import Source
# locals
from ..models.br import BatchReactorOptions
from ..models.rate_exp import ReactionRateExpression
from ..models import GasModel
from ..utils.unit_tools import to_J_per_mol_K, to_g_per_m3


class ThermoModelInputs:
    # NOTE: Attributes
    gas_heat_capacity_constant_values: np.ndarray = np.array([])
    gas_heat_capacity_constant_comp: Dict[str, float] = {}
    liquid_heat_capacity_constant_values: np.ndarray = np.array([])
    liquid_heat_capacity_constant_comp: Dict[str, float] = {}
    liquid_density_constant_values: np.ndarray = np.array([])
    liquid_density_constant_comp: Dict[str, float] = {}

    def __init__(
        self,
        components: List[Component],
        source: Source,
        model_inputs: Dict[str, Any],
        reactor_inputs: BatchReactorOptions,
        component_key: ComponentKey,
        component_formula_state: List[str],
        model_inputs_keys: List[str],
    ):
        """
        Initializes the ThermoInputs instance with the provided components, source, model inputs, reactor inputs, reaction rates, and component key.

        Parameters
        ----------
        components : List[Component]
            A list of Component objects representing the chemical components involved in the model source.
        source : Source
            A Source object containing information about the source of the data or equations used in the model source.
        model_inputs : Dict[str, Any]
            A dictionary of model inputs, where the keys are the names of the inputs and the values are the input values. This can include feed specifications, initial conditions, or any other relevant parameters needed for the simulations.
        reactor_inputs : BatchReactorOptions
            A BatchReactorOptions object containing the inputs for the batch reactor simulation, such as volume, heat transfer properties, etc.
        reaction_rates : List[ReactionRateExpression]
            A list of reaction rate expressions, where each expression is represented as a ReactionRateExpression object containing information about the reaction and its rate expression.
        component_key : ComponentKey
            A ComponentKey object that serves as a key for identifying and categorizing the components in the model source.
        """
        # NOTE: Set attributes
        self.components = components
        self.source = source
        self.model_inputs = model_inputs
        self.component_key = component_key

        # SECTION: heat capacity and density modes
        self.component_formula_state = component_formula_state
        self.model_inputs_keys = model_inputs_keys

        # NOTE: Reactor configuration
        # ! gas heat capacity mode
        self.gas_heat_capacity_mode = reactor_inputs.gas_heat_capacity_mode
        # ! liquid heat capacity mode
        self.liquid_heat_capacity_mode = reactor_inputs.liquid_heat_capacity_mode
        # ! density mode
        self.liquid_density_mode = reactor_inputs.liquid_density_mode
        # ! phase
        self.phase = reactor_inputs.phase
        # ! heat transfer more
        self.heat_transfer_mode = reactor_inputs.heat_transfer_mode

        # SECTION: Extract property sources and configure properties
        # ! Ideal Gas Heat Capacity at reference temperature (e.g., 298 K)
        if self.heat_transfer_mode == "non-isothermal":
            # check heat capacity mode
            if self.gas_heat_capacity_mode == "constant":
                # NOTE: use constant heat capacity from model inputs
                # >> constant heat capacity
                # ! to J/mol.K
                (
                    self.gas_heat_capacity_constant_values,
                    self.gas_heat_capacity_constant_comp
                ) = self._config_constant_gas_heat_capacity()

        # ! phase
        if self.phase == "liquid":
            # check heat capacity mode
            if self.liquid_heat_capacity_mode == "constant":
                # NOTE: use constant heat capacity from model inputs
                # >> constant heat capacity
                # ! to J/mol.K
                (
                    self.liquid_heat_capacity_constant_values,
                    self.liquid_heat_capacity_constant_comp
                ) = self._config_constant_liquid_heat_capacity()

            # check density mode
            if self.liquid_density_mode == "constant":
                # NOTE: use constant density from model inputs
                # >> constant density
                # ! to g/m3
                (
                    self.liquid_density_constant_values,
                    self.liquid_density_constant_comp
                ) = self._config_constant_liquid_density()

    # NOTE: heat capacity configuration
    # ! gas phase

    def _config_constant_gas_heat_capacity(
            self,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Configure the heat capacity for the batch reactor based on the model inputs and reactor configuration."""
        # check heat capacity mode
        if self.gas_heat_capacity_mode is None:
            raise ValueError(
                "Heat capacity mode must be specified in reactor_inputs for non-isothermal reactors.")

        # heat capacity constant
        if "gas_heat_capacity" in self.model_inputs_keys:
            heat_capacity_: dict[
                str,
                CustomProp
            ] = self.model_inputs["gas_heat_capacity"]

            # iterate through components and extract heat capacity values
            heat_capacity_values = []
            heat_capacity_comp = {}

            for id in self.component_formula_state:
                if id in heat_capacity_:
                    cp_value = to_J_per_mol_K(
                        heat_capacity_[id].value,
                        heat_capacity_[id].unit
                    )

                    # add
                    heat_capacity_values.append(cp_value)
                    heat_capacity_comp[id] = cp_value
                else:
                    raise ValueError(
                        f"Heat capacity value for component '{id}' not found in model_inputs."
                    )

            heat_capacity_array = np.array(heat_capacity_values)

            # res
            return heat_capacity_array, heat_capacity_comp
        else:
            raise ValueError(
                "Heat capacity must be provided in model_inputs for constant heat capacity mode."
            )

    # ! constant liquid phase heat capacity
    def _config_constant_liquid_heat_capacity(
            self,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Configure the heat capacity for the batch reactor based on the model inputs and reactor configuration.
        """
        # check heat capacity mode
        if self.liquid_heat_capacity_mode is None:
            raise ValueError(
                "Heat capacity mode must be specified in reactor_inputs for non-isothermal reactors.")

        # heat capacity constant
        if "liquid_heat_capacity" in self.model_inputs_keys:
            heat_capacity_: dict[
                str,
                CustomProp
            ] = self.model_inputs["liquid_heat_capacity"]

            # iterate through components and extract heat capacity values
            heat_capacity_values = []
            heat_capacity_comp = {}

            for id in self.component_formula_state:
                if id in heat_capacity_:
                    cp_value = to_J_per_mol_K(
                        heat_capacity_[id].value,
                        heat_capacity_[id].unit
                    )

                    # add
                    heat_capacity_values.append(cp_value)
                    heat_capacity_comp[id] = cp_value
                else:
                    raise ValueError(
                        f"Heat capacity value for component '{id}' not found in model_inputs."
                    )

            heat_capacity_array = np.array(heat_capacity_values)

            # res
            return heat_capacity_array, heat_capacity_comp
        else:
            raise ValueError(
                "Heat capacity must be provided in model_inputs for constant heat capacity mode."
            )

    # ! liquid density
    def _config_constant_liquid_density(
            self
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Configure the density for the batch reactor based on the model inputs and reactor configuration.
        """
        # check density mode
        if self.liquid_density_mode is None:
            raise ValueError(
                "Density mode must be specified in reactor_inputs for liquid phase.")

        # density constant
        if "density" in self.model_inputs_keys:
            density_: dict[
                str,
                CustomProp
            ] = self.model_inputs["density"]

            # iterate through components and extract density values
            density_values = []
            density_comp = {}

            for id in self.component_formula_state:
                if id in density_:
                    density_value = to_g_per_m3(
                        density_[id].value,
                        density_[id].unit
                    )

                    # add
                    density_values.append(density_value)
                    density_comp[id] = density_value
                else:
                    raise ValueError(
                        f"Density value for component '{id}' not found in model_inputs."
                    )

            density_array = np.array(density_values)

            # res
            return density_array, density_comp
        else:
            raise ValueError(
                "Density must be provided in model_inputs for constant density mode."
            )
