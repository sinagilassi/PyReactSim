# import libs
import logging
from typing import List, Dict, Any, cast
from pythermodb_settings.models import Component, Temperature, Pressure, CustomProperty, ComponentKey
from pythermodb_settings.utils import set_component_id, build_components_mapper
from pyThermoLinkDB.models import ModelSource
from pyThermoLinkDB.thermo import Source
from pyThermoLinkDB.models.component_models import ComponentEquationSource
# locals
from ..sources.thermo_model_source import ThermoModelSource
from ..sources.thermo_model_inputs import ThermoModelInputs
from ..sources.thermo_reaction import ThermoReaction
from ..models.br import BatchReactorOptions
from ..models.rate_exp import ReactionRateExpression
from .thermo_source_core import ThermoSourceCore


# NOTE: logger setup
logger = logging.getLogger(__name__)


class ThermoSource(ThermoSourceCore):
    def __init__(
        self,
        components: List[Component],
        source: Source,
        model_inputs: Dict[str, Any],
        reactor_inputs: BatchReactorOptions,
        reaction_rates: List[ReactionRateExpression],
        component_key: ComponentKey,
        component_refs: Dict[str, Any],
    ):
        """
        Initializes the ThermoSource instance with the provided components, source, model inputs, reactor inputs, reaction rates, and component key.

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
        # SECTION: Set attributes
        self.components = components

        # component reference
        self.component_refs = component_refs

        # LINK: ThermoModelSource initialization
        ThermoModelSource_ = ThermoModelSource(
            components=components,
            source=source,
            model_inputs=model_inputs,
            reactor_inputs=reactor_inputs,
            reaction_rates=reaction_rates,
            component_key=component_key,
            component_refs=component_refs
        )

        # LINK: ThermoInputs initialization
        ThermoModelInputs_ = ThermoModelInputs(
            components=components,
            source=source,
            model_inputs=model_inputs,
            reactor_inputs=reactor_inputs,
            component_key=component_key,
            component_formula_state=self.component_formula_state,
            model_inputs_keys=self.model_inputs_keys,
        )

        # LINK: ThermoReaction initialization
        ThermoReaction_ = ThermoReaction(
            components=components,
            source=source,
            model_inputs=model_inputs,
            reaction_rates=reaction_rates,
            component_key=component_key
        )

        # section: Initialize ThermoSourceCore
        ThermoSourceCore.__init__(
            self,
            components=components,
            source=source,
            model_inputs=model_inputs,
            reactor_inputs=reactor_inputs,
            reaction_rates=reaction_rates,
            component_key=component_key,
            thermal_model_source=ThermoModelSource_,
            thermal_model_inputs=ThermoModelInputs_,
            thermal_reaction=ThermoReaction_,
            component_refs=component_refs,
        )
