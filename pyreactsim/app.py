# import libs
import logging
from typing import Dict, Any
# locals
from .docs.br import BatchReactor
from .docs.cstr import CSTRReactor
from .docs.pfr import PFRReactor
from .docs.pbr import PBRReactor
from .sources.thermo_source import ThermoSource
from .core.gas_br import GasBatchReactor
from .core.gas_brx import GasBatchReactorX
from .core.liquid_br import LiquidBatchReactor
from .core.liquid_brx import LiquidBatchReactorX
from .observables.gas_br import GasBatchReactorObservables
from .observables.liquid_br import LiquidBatchReactorObservables


# NOTE: logger setup
logger = logging.getLogger(__name__)

# SECTION: Create Reactor

# NOTE: Batch Reactor Factory Functions


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

# NOTE: CSTR Factory Function


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

# NOTE: PFR Factory Function


def create_pfr_reactor(
    model_inputs: Dict[str, Any],
    thermo_source: ThermoSource,
    **kwargs,
) -> PFRReactor:
    """
    Factory function to create a PFRReactor instance.

    Parameters
    ----------
    model_inputs : Dict[str, Any]
        A dictionary of model inputs for PFR simulation.
    thermo_source : ThermoSource
        Thermodynamic source for PFR simulation.
    **kwargs
        Additional keyword arguments for future extensions.

    Returns
    -------
    PFRReactor
        An instance of the PFRReactor class.
    """
    pfr_reactor = PFRReactor(
        model_inputs=model_inputs,
        thermo_source=thermo_source,
        **kwargs,
    )

    return pfr_reactor

# NOTE: PBR Factory Function


def create_pbr_reactor(
    model_inputs: Dict[str, Any],
    thermo_source: ThermoSource,
    **kwargs,
) -> PBRReactor:
    """
    Factory function to create a PBRReactor instance.

    Parameters
    ----------
    model_inputs : Dict[str, Any]
        A dictionary of model inputs for PBR simulation.
    thermo_source : ThermoSource
        Thermodynamic source for PBR simulation.
    **kwargs
        Additional keyword arguments for future extensions.

    Returns
    -------
    PBRReactor
        An instance of the PBRReactor class.
    """
    pbr_reactor = PBRReactor(
        model_inputs=model_inputs,
        thermo_source=thermo_source,
        **kwargs,
    )

    return pbr_reactor

# SECTION: Evaluate Batch Reactor Simulation Results
# NOTE: Evaluate Batch Reactor Simulation Results


def evaluate_batch_reactor(
    batch_reactor: BatchReactor,
    simulation_results: Any,
) -> Dict[str, Any]:
    """
    Evaluate trajectory-aligned observables for a batch reactor simulation result.

    Parameters
    ----------
    batch_reactor : BatchReactor
        BatchReactor instance used for simulation.
    simulation_results : Any
        Output from `batch_reactor.simulate(...)` (BatchReactorResult-like).

    Returns
    -------
    Dict[str, Any]
        Dictionary of evaluated observables.
    """
    if simulation_results is None:
        logger.warning("No simulation results provided for evaluation.")
        return {}

    time = getattr(simulation_results, "time", None)
    state = getattr(simulation_results, "state", None)

    if time is None and isinstance(simulation_results, dict):
        time = simulation_results.get("time")
    if state is None and isinstance(simulation_results, dict):
        state = simulation_results.get("state")

    if time is None or state is None:
        logger.error(
            "Simulation results do not contain 'time' and 'state' attributes or keys."
        )
        return {}

    reactor = batch_reactor.reactor
    if isinstance(reactor, (GasBatchReactor, GasBatchReactorX)):
        return GasBatchReactorObservables(reactor).evaluate_all(t=time, y=state)
    if isinstance(reactor, (LiquidBatchReactor, LiquidBatchReactorX)):
        return LiquidBatchReactorObservables(reactor).evaluate_all(t=time, y=state)

    logger.error(
        f"Unsupported reactor type for observables evaluation: {type(reactor)}"
    )
    return {}
