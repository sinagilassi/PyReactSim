# import libs
import logging
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, cast
from pythermodb_settings.models import Component, ComponentKey, CustomProp, CustomProperty, Temperature
from pythermodb_settings.utils import set_component_id
from pyThermoLinkDB.thermo import Source
from pyThermoLinkDB.models.component_models import ComponentEquationSource
from pyreactlab_core.models.reaction import Reaction
from pyThermoCalcDB.reactions.reactions import dH_rxn_STD

# locals
from ..sources.interface import (
    ext_component_dt,
    ext_components_dt,
    ext_component_eq,
    ext_components_eq,
    exec_component_eq
)
from ..utils.unit_tools import to_K
from ..utils.reaction_tools import stoichiometry_mat
from ..models.rate_exp import ReactionRateExpression
from ..models.br import GasModel
from ..models.br import BatchReactorOptions

# NOTE: logger setup
logger = logging.getLogger(__name__)


class ThermoSource:
    """
    THermo class for handling thermodynamic calculations and properties related to chemical reactions and processes. This class provides methods for calculating various thermodynamic properties, such as heat capacities, enthalpies, and entropies, as well as methods for performing energy balance calculations in chemical systems.
    """

    def __init__(
        self,
        components: List[Component],
        source: Source,
        reactor_inputs: BatchReactorOptions,
        reaction_rates: Dict[str, ReactionRateExpression],
        component_key: ComponentKey,
    ):
        """
        Initializes the THermo instance with default properties and settings for thermodynamic calculations.
        """
        # NOTE: Set attributes
        self.components = components
        self.source = source
        self.reactor_inputs = reactor_inputs
        self.reaction_rates = reaction_rates
        self.component_key = component_key

        # NOTE: Create component ID list
        self.component_ids = [
            set_component_id(component=comp, component_key=self.component_key)
            for comp in self.components
        ]

        # NOTE: model source
        self.model_source = self.source.model_source

        # NOTE: reactions
        self.reactions: List[Reaction] = self.build_reactions()

        # SECTION: Thermodynamic properties
        # ! Ideal Gas Heat Capacity at reference temperature (e.g., 298 K)
        if self.reactor_inputs.heat_transfer_mode == "non-isothermal":
            self.Cp_IG_src = self.prop_eq_src(prop_name="Cp_IG")

        # ! Ideal Gas Enthalpy of formation at 298 K
        if self.reactor_inputs.heat_transfer_mode == "non-isothermal":
            self.EnFo_IG_298_src = self.prop_dt_src(
                component_ids=self.component_ids,
                prop_name="EnFo_IG"
            )

    # SECTION: Property equation source extraction methods
    # ! Extract property equation source for components
    def prop_eq_src(self, prop_name: str) -> Dict[str, ComponentEquationSource]:
        """
        Extracts the property equation for the components from the source and returns it as a dictionary.

        Returns
        -------
        Dict[str, ComponentEquationSource]
            A dictionary where the keys are component IDs and the values are ComponentEquationSource objects
        """
        # NOTE: Extract property equation source for all components
        eq_src = ext_components_eq(
            components=self.components,
            prop_name=prop_name,
            source=self.source,
            component_key=cast(ComponentKey, self.component_key)
        )
        # >> check
        if eq_src is None:
            logger.error("Failed to extract property equation for components.")
            return {}

        return eq_src

    # ! Extract property data source for components
    def prop_dt_src(
            self,
            component_ids: List[str],
            prop_name: str
    ) -> Dict[str, Dict[str, Any]]:
        """
        Extracts the property derivative equation for the components from the source and returns it as a dictionary.

        Returns
        -------
        Dict[str, ComponentEquationSource]
            A dictionary where the keys are component IDs and the values are ComponentEquationSource objects
        """
        # NOTE: Extract property derivative equation source for all components
        dt_src = ext_components_dt(
            component_ids=component_ids,
            prop_name=prop_name,
            source=self.source,
        )
        # >> check
        if dt_src is None:
            logger.error(
                "Failed to extract property derivative equation for components.")
            return {}

        return dt_src

    # SECTION: Reaction and stoichiometry related methods
    # ! Extract all reactions
    def build_reactions(self):
        """
        Build the list of Reaction objects for the gas-phase batch reactor using the provided reaction rates and components.
        """
        reactions = []
        for rxn_name, rate_exp in self.reaction_rates.items():
            rxn = rate_exp.reaction
            reactions.append(rxn)
        return reactions

    # ! Build stoichiometry matrix
    def build_stoichiometry(self):
        """
        Build the stoichiometry matrix for the reactions in the gas-phase batch reactor using the provided reaction rates and components.
        """
        # >> extract reactions from reaction rates
        reactions = []

        for rxn_name, rate_exp in self.reaction_rates.items():
            rxn = rate_exp.reaction
            reactions.append(rxn)

        # >> build stoichiometry matrix
        mat = stoichiometry_mat(
            reactions=reactions,
            components=self.components,
            component_key=cast(ComponentKey, self.component_key)
        )

        return mat

    # SECTION: Thermodynamic property calculations
    # ! Heat capacity at ideal gas at temperature T (Cp_IG)
    def calc_Cp_IG(
            self,
            inputs: Dict[str, Any],
            Cp_IG_src: Dict[str, ComponentEquationSource],
            output_unit: Optional[str] = "J/mol.K"
    ) -> np.ndarray:
        # NOTE: extract Cp_IG at reference temperature (e.g., 298 K)
        Cp_IG_ref: Dict[str, CustomProperty] = {}
        for comp in self.component_ids:
            eq_src = Cp_IG_src.get(comp)
            if eq_src is None:
                raise ValueError(
                    f"No Cp_IG source found for component: {comp}")

            # >> execute equation to get Cp_IG at reference conditions
            cp_value = exec_component_eq(
                component_eq_src=eq_src,
                inputs=inputs,
                output_unit=output_unit
            )
            if cp_value is None:
                raise ValueError(
                    f"Failed to extract Cp_IG value for component: {comp}")
            Cp_IG_ref[comp] = cp_value

        # NOTE: build a list of Cp_IG values for all components
        Cp_IG_values = []

        # iterate over components
        for comp in self.component_ids:
            cp_value = Cp_IG_ref.get(comp)
            if cp_value is None:
                raise ValueError(
                    f"No Cp_IG value found for component: {comp}")
            Cp_IG_values.append(cp_value.value)

        # >> to numpy
        Cp_IG_values = np.array(Cp_IG_values, dtype=float)

        return Cp_IG_values

    # ! Calculate reaction enthalpies (ΔH) for reactions
    def calc_dH_rxns(
            self,
            temperature: Temperature
    ) -> np.ndarray:
        """
        Calculate the reaction enthalpies (ΔH) for the reactions in the gas-phase batch reactor using the provided reaction rates and components.

        Parameters
        ----------
        temperature : Temperature
            The temperature at which to calculate the reaction enthalpies (ΔH) for the reactions
            in the gas-phase batch reactor.

        Returns
        -------
        np.ndarray
            An array of reaction enthalpies (ΔH) for the reactions in the gas-phase batch reactor, calculated at the specified temperature.
        """
        # create model source
        model_source = self.source.model_source
        # >> check
        if model_source is None:
            raise ValueError(
                "Model source is required to calculate reaction enthalpies.")
        #
        dH_rxns = []
        for rxn in self.reactions:
            dH_rxn = dH_rxn_STD(
                reaction=rxn,
                temperature=temperature,
                model_source=model_source,
            )

            # >> check
            if dH_rxn is None:
                raise ValueError(
                    f"Failed to calculate reaction enthalpy for reaction: {rxn.name}")

            dH_rxns.append(dH_rxn)

        # NOTE: convert to numpy array
        res = [dH.value for dH in dH_rxns]
        res = np.array(res, dtype=float)

        return res

    # SECTION: EOS related methods
    # ! Calculate total pressure using ideal gas law
    def calc_tot_pressure(
            self,
            n_total: float,
            temperature: float,
            reactor_volume_value: float,
            R: float,
            gas_model: GasModel
    ) -> float:
        """
        Total pressure [Pa].
        Default: ideal gas
            P = N_total * R * T / V

        Parameters
        ----------
        n_total : float
            Total moles of gas in the reactor.
        temperature : float
            Temperature of the gas in the reactor [K].

        Returns
        -------
        float
            Total pressure of the gas in the reactor [Pa].
        """
        if gas_model == "real":
            # FIXME: implement real gas model
            return 0

        # ideal gas model
        return n_total * R * temperature / float(reactor_volume_value)
