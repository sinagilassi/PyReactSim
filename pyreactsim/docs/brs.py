# import libs
import logging
from typing import Any, Dict, List
from pythermodb_settings.models import Component, ComponentKey
from pyThermoLinkDB.models import ModelSource
from pyThermoLinkDB.thermo import Source

# locals
from ..models.br import BatchReactorOptions
from ..models.rate_exp import ReactionRateExpression
from ..core.gas_br import GasBatchReactor

# NOTE: set logger
logger = logging.getLogger(__name__)


def sample():
    """
    Sample function for testing and demonstration purposes.
    """
    logger.info(
        "This is a sample function for testing and demonstration purposes."
    )


def batch_react(
    components: List[Component],
    model_inputs: Dict[str, Any],
    reactor_inputs: BatchReactorOptions,
    reaction_rates: Dict[str, ReactionRateExpression],
    model_source: ModelSource,
    component_key: ComponentKey,
    **kwargs,
) -> None:
    """
    Simulate a batch reactor with the given components, model inputs, reactor inputs, and reaction rates.

    Parameters
    ----------
    components : List[Component]
        A list of Component objects representing the components in the reaction.
    model_inputs : Dict[str, Any]
        A dictionary of model inputs, where the keys are the names of the
        inputs and the values are the input values.
    reactor_inputs : BatchReactorOptions
        A BatchReactorOptions object containing the inputs for the batch reactor simulation.
    reaction_rates : Dict[str, ReactionRateExpression]
        A dictionary of reaction rate expressions, where the keys are the names
        of the reactions and the values are the ReactionRateExpression objects.
    model_source : ModelSource
        A ModelSource object containing the source of the model to be used
        in the simulation.
    component_key : ComponentKey
        A ComponentKey object representing the key to be used for the components in the model source.
    **kwargs
        Additional keyword arguments to be passed to the simulation function.
    """
    try:
        # NOTE: set feed specification
        n = model_inputs.get("mole")
        # >> check
        if n is None:
            raise ValueError("Mole input is required for feed specification.")

        # NOTE: Create source
        source = Source(
            model_source=model_source,
            component_key=component_key,
        )

        # NOTE: create batch reactor model
        _ = GasBatchReactor(
            components=components,
            source=source,
            model_inputs=model_inputs,
            reactor_inputs=reactor_inputs,
            reaction_rates=reaction_rates,
            component_key=component_key,
            **kwargs,
        )

        # NOTE: create function
        #

    except Exception as e:
        logger.error(f"Error during batch reactor simulation: {e}")
