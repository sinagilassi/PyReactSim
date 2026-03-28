# import libs
import logging
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, cast
from pythermodb_settings.models import Component, Temperature, Pressure, ComponentKey, CustomProperty
from pyThermoLinkDB.thermo import Source
# locals
from .br import BatchReactor
from .thermo_source import ThermoSource
from ..models.br import BatchReactorOptions
from ..models.rate_exp import ReactionRateExpression
from ..utils.reaction_tools import stoichiometry_mat_key, stoichiometry_mat
from ..utils.thermo_tools import calc_total_heat_capacity, calc_rxn_heat_generation
from ..utils.opt_tools import calc_heat_exchange, set_component_X

# NOTE: logger setup
logger = logging.getLogger(__name__)


class LiquidBatchReactor(BatchReactor, ThermoSource):
    """
    Liquid Batch Reactor (LBR) class for simulating batch reactions in liquid phase.

    Modeling assumptions
    --------------------
    - The reactor is perfectly mixed.
    - The liquid phase is spatially homogeneous.
    - Reactions occur in a single liquid phase.
    - There is no inlet and no outlet.
    - Primary state variables are moles, not concentrations.
    - Concentrations are derived quantities.
    - Internal energy balance is written on a molar volumetric heat-capacity basis.
    - SI units are used throughout.
    """
    # NOTE: Attributes

    def __init__(
        self,
        components: List[Component],
        source: Source,
        model_inputs: Dict[str, Any],
        reactor_inputs: BatchReactorOptions,
        reaction_rates: Dict[str, ReactionRateExpression],
        component_key: ComponentKey,
        **kwargs
    ):
        """
        Initializes the GasBatchReactor instance with the provided components, source, and component key.

        Parameters
        ----------
        components : List[Component]
            A list of Component objects representing the chemical components involved in the gas-phase batch reactor.
        source : Source
            A Source object containing information about the source of the data or equations used in the gas-phase batch reactor simulations.
        model_inputs : Dict[str, Any]
            A dictionary containing the model inputs for the gas-phase batch reactor simulations, such as temperature, pressure, and initial mole numbers.
        reactor_inputs : BatchReactorOptions
            A BatchReactorOptions object containing the options and parameters specific to the gas-phase batch reactor setup, such as heat transfer mode, volume mode, and gas model.
        reaction_rates : Dict[str, ReactionRateExpression]
            A dictionary containing the reaction rate expressions for the reactions occurring in the gas-phase batch reactor,
            where the keys are the names of the reactions and the values are ReactionRateExpression objects.
        component_key : ComponentKey
            A ComponentKey object representing the key to be used for the components in the model source.
        **kwargs
            Additional keyword arguments that can be passed to the initialization of the GasBatchReactor instance.
        """
        # LINK: Initialize the parent BatchReactor class
        BatchReactor.__init__(
            self,
            components=components,
            source=source,
            model_inputs=model_inputs,
            reactor_inputs=reactor_inputs,
            component_key=component_key
        )
        # LINK: Initialize the parent ThermoSource class
        ThermoSource.__init__(
            self,
            components=components,
            source=source,
            model_inputs=model_inputs,
            reactor_inputs=reactor_inputs,
            reaction_rates=reaction_rates,
            component_key=component_key
        )

        # ! N: initial mole [-]
        _, self._N0 = set_component_X(
            components=components,
            prop_name="mole",
            component_key=component_key
        )

        # ! P: initial pressure [Pa]
        if self.operation_mode == "constant_volume":
            # retrieve
            self.volume = self._config_reactor_volume()
            self._V0 = self.volume.value

            # calc
            self._P0 = self.calc_tot_pressure(
                n_total=np.sum(self._N0),
                temperature=self._T0,
                reactor_volume_value=self._V0,
                R=self.R,
                gas_model=self.gas_model
            )

        # ! V: initial volume [m3]
        elif self.operation_mode == "constant_pressure":
            # retrieve
            self.pressure = self._config_pressure()
            self._P0 = self.pressure.value

            # calc
            # FIXME
            # self._V0 = self.calc_liquid_volume(
            #     n_total=np.sum(self._N0),
            #     molecular_weights=self.molecular_weights,
            #     density=self.density
            # )

        else:
            raise ValueError(
                f"Invalid operation mode '{self.operation_mode}'. Must be constant pressure or volume."
            )

        # SECTION: Model inputs
        self.model_inputs = model_inputs

        # SECTION: GasBatchReactor-specific properties
        self.reactor_inputs = reactor_inputs

        # SECTION: Reaction rates
        self.reaction_rates = reaction_rates
        # >> build reactions
        self.reactions = self.build_reactions()
        # >>> build stoichiometry matrix
        self.reaction_stoichiometry: List[Dict[str, float]] = stoichiometry_mat_key(
            reactions=self.reactions,
            component_key=component_key
        )
        # >> matrix
        self.reaction_stoichiometry_matrix = stoichiometry_mat(
            reactions=self.reactions,
            components=self.components,
            component_key=component_key,
        )

        # SECTION: Thermodynamic properties
        # ! Ideal Gas Heat Capacity at reference temperature (e.g., 298 K)
        # ! Ideal Gas Enthalpy of formation at 298 K
