# import libs
import logging
from typing import Dict, List, Optional, Any
from pythermodb_settings.models import Component
from pyThermoLinkDB.models import ModelSource
# locals
from .docs.br import BatchReactor
from .docs.cstr import CSTRReactor
from .models import BatchReactorOptions, HeatTransferOptions
from .models.rate_exp import ReactionRateExpression
from .sources.thermo_source import ThermoSource


# NOTE: logger setup
logger = logging.getLogger(__name__)

# SECTION: Create Batch Reactor


def create_batch_reactor(
    model_inputs: Dict[str, Any],
    thermo_source: ThermoSource,
    **kwargs,
) -> BatchReactor:
    """
    Factory function to create a GasBatchReactor instance.

    Parameters
    ----------
    model_inputs : Dict[str, Any]
        A dictionary of model inputs, where the keys are the names of the inputs and the values are the input values.
        - feed mole: Dict[str, CustomProp]
        - feed temperature: Temperature
        - feed pressure: Pressure
    thermo_source : ThermoSource
        A ThermoSource object containing the thermodynamic source information for the batch reactor simulation.
    **kwargs
        Additional keyword arguments for future extensions.

    Returns
    -------
    BatchReactor
        An instance of the BatchReactor class configured for gas phase reactions.
    """
    # NOTE: create batch reactor instance
    batch_reactor = BatchReactor(
        model_inputs=model_inputs,
        thermo_source=thermo_source,
        **kwargs,
    )

    return batch_reactor


def create_cstr_reactor(
    model_inputs: Dict[str, Any],
    thermo_source: ThermoSource,
    **kwargs,
) -> CSTRReactor:
    """
    Factory function to create a CSTRReactor instance.

    Parameters
    ----------
    model_inputs : Dict[str, Any]
        A dictionary of model inputs for CSTR simulation.
    thermo_source : ThermoSource
        Thermodynamic source for CSTR simulation.
    **kwargs
        Additional keyword arguments for future extensions.

    Returns
    -------
    CSTRReactor
        An instance of the CSTRReactor class.
    """
    cstr_reactor = CSTRReactor(
        model_inputs=model_inputs,
        thermo_source=thermo_source,
        **kwargs,
    )

    return cstr_reactor
