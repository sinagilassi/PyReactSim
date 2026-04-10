# import libs
import logging
import time
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
from ..models.heat import HeatTransferOptions
from ..models.rate_exp import ReactionRateExpression
from .thermo_source_core import ThermoSourceCore
from ..models.cstr import CSTRReactorOptions
from ..models.pfr import PFRReactorOptions
from ..models.pbr import PBRReactorOptions


# NOTE: logger setup
logger = logging.getLogger(__name__)


class ThermoSource(ThermoSourceCore):
    def __init__(
        self,
        components: List[Component],
        source: Source,
        thermo_inputs: Dict[str, Any],
        reactor_options: BatchReactorOptions | CSTRReactorOptions | PFRReactorOptions | PBRReactorOptions,
        heat_transfer_options: HeatTransferOptions,
        reaction_rates: List[ReactionRateExpression],
        component_refs: Dict[str, Any],
        component_key: ComponentKey,
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

        # LINK: ThermoReaction initialization
        t_start = time.perf_counter()
        ThermoReaction_ = ThermoReaction(
            components=components,
            thermo_inputs=thermo_inputs,
            reaction_rates=reaction_rates,
            component_key=component_key
        )
        thermo_reaction_ms = (time.perf_counter() - t_start) * 1000.0
        logger.debug(
            "ThermoReaction initialization time: %.2f ms",
            thermo_reaction_ms
        )

        # LINK: ThermoModelSource initialization
        t_start = time.perf_counter()
        ThermoModelSource_ = ThermoModelSource(
            components=components,
            source=source,
            thermo_inputs=thermo_inputs,
            reactor_options=reactor_options,
            heat_transfer_options=heat_transfer_options,
            reaction_rates=reaction_rates,
            component_refs=component_refs,
            component_key=component_key,
        )
        thermo_model_source_ms = (time.perf_counter() - t_start) * 1000.0
        logger.debug(
            "ThermoModelSource initialization time: %.2f ms",
            thermo_model_source_ms
        )

        # LINK: ThermoInputs initialization
        t_start = time.perf_counter()
        ThermoModelInputs_ = ThermoModelInputs(
            components=components,
            thermo_inputs=thermo_inputs,
            reactor_options=reactor_options,
            heat_transfer_options=heat_transfer_options,
            component_refs=component_refs,
            component_key=component_key,
        )
        thermo_model_inputs_ms = (time.perf_counter() - t_start) * 1000.0
        logger.debug(
            "ThermoModelInputs initialization time: %.2f ms",
            thermo_model_inputs_ms
        )

        logger.debug(
            "ThermoSource helper initialization total time: %.2f ms",
            thermo_model_source_ms + thermo_model_inputs_ms + thermo_reaction_ms
        )

        # section: Initialize ThermoSourceCore
        ThermoSourceCore.__init__(
            self,
            components=components,
            source=source,
            thermo_inputs=thermo_inputs,
            reactor_options=reactor_options,
            heat_transfer_options=heat_transfer_options,
            reaction_rates=reaction_rates,
            component_key=component_key,
            thermo_model_source=ThermoModelSource_,
            thermo_model_inputs=ThermoModelInputs_,
            thermo_reaction=ThermoReaction_,
            component_refs=component_refs,
        )
