# import libs
import logging
from typing import List, Dict, Any, cast
from pythermodb_settings.models import Component, Temperature, Pressure, CustomProperty, CustomProp, ComponentKey

from pyThermoLinkDB.thermo import Source
from pyThermoLinkDB.models.component_models import ComponentEquationSource
# locals
from ..utils.reaction_tools import stoichiometry_mat
from ..models.rate_exp import ReactionRateExpression

# NOTE: logger setup
logger = logging.getLogger(__name__)


class ThermoReaction:
    """
    A class representing the thermodynamic and kinetic information for a chemical reaction system.
    """

    def __init__(
        self,
        components: List[Component],
        source: Source,
        model_inputs: Dict[str, Any],
        reaction_rates: List[ReactionRateExpression],
        component_key: ComponentKey,
    ):
        """
        Initializes the ThermoReaction instance with the provided components, source, model inputs, reactor inputs, reaction rates, and component key.

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
        self.reaction_rates = reaction_rates
        self.component_key = component_key

    # SECTION: Reaction and stoichiometry related methods
    # ! Extract all reactions

    def build_reactions(self):
        """
        Build the list of Reaction objects for the gas-phase batch reactor using the provided reaction rates and components.
        """
        reactions = []
        for rate_exp in self.reaction_rates:
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

        for rate_exp in self.reaction_rates:
            rxn = rate_exp.reaction
            reactions.append(rxn)

        # >> build stoichiometry matrix
        mat = stoichiometry_mat(
            reactions=reactions,
            components=self.components,
            component_key=cast(ComponentKey, self.component_key)
        )

        return mat

    def get_reaction_names(self) -> List[str]:
        """
        Get the list of reaction names for the reactions in the gas-phase batch reactor using the provided reaction rates.

        Returns
        -------
        List[str]
            A list of reaction names for the reactions in the gas-phase batch reactor.
        """
        reaction_names = []
        for rate_exp in self.reaction_rates:
            rxn = rate_exp.reaction
            reaction_names.append(rxn.name)
        return reaction_names

    def get_reaction_index(self) -> Dict[str, int]:
        """
        Get the reaction index for the reactions in the reactor using the provided reaction rates.
        Returns
        -------
        Dict[str, int]
            A dictionary where the keys are reaction names and the values are the reaction orders for the reactions in the gas-phase batch reactor.
        """
        res = {}
        for index, rate_exp in enumerate(self.reaction_rates):
            rxn = rate_exp.reaction
            res[rxn.name] = index
        return res
