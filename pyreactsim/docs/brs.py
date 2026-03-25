# import libs
import logging
from scipy.integrate import solve_ivp
from typing import Any, Dict, List, Optional, Tuple, Union
from pythermodb_settings.models import Component, ComponentKey
from pyThermoLinkDB.models import ModelSource
from pyThermoLinkDB.thermo import Source
# locals
from ..models.br import BatchReactorOptions, BatchReactorResult, HeatTransferMode
from ..models.rate_exp import ReactionRateExpression
from ..core.gas_br import GasBatchReactor


# NOTE: set logger
logger = logging.getLogger(__name__)


# SECTION: Batch Reactor Simulation Function
def batch_react(
    components: List[Component],
    model_inputs: Dict[str, Any],
    reactor_inputs: BatchReactorOptions,
    reaction_rates: Dict[str, ReactionRateExpression],
    model_source: ModelSource,
    component_key: ComponentKey,
    solver_options: Optional[Dict[str, Any]] = None,
    **kwargs,
):
    """
    Simulate a batch reactor with the given components, model inputs, reactor inputs, and reaction rates.

    Parameters
    ----------
    components : List[Component]
        A list of Component objects representing the components in the reaction.
    model_inputs : Dict[str, Any]
        A dictionary of model inputs, where the keys are the names of the inputs and the values are the input values.
        - feed mole: Dict[str, CustomProp]
        - feed temperature: Temperature
        - feed pressure: Pressure
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
    solver_options : Optional[Dict[str, Any]], optional
        A dictionary of solver options, where the keys are the names of the options and the values as:
        - method: str, the integration method to use (e.g., 'BDF', 'RK45', etc.)
        - time_span: Tuple[float, float], the time span for the simulation (start, end)
        - rtol: float, the relative tolerance for the solver
        - atol: float, the absolute tolerance for the solver (default is None, which will use default solver settings).
    **kwargs
        Additional keyword arguments to be passed to the simulation function.
    """
    try:
        # SECTION: Input Validation
        # NOTE: Component
        if not components:
            raise ValueError("The components list cannot be empty.")

        if not isinstance(components, list) or not all(isinstance(c, Component) for c in components):
            raise TypeError(
                "The components must be a list of Component objects.")

        # length
        if len(components) < 2:
            raise ValueError(
                "At least two components are required for a reaction.")

        # NOTE: Solver options
        # ! method
        method = solver_options.get(
            'method', 'BDF'
        ) if solver_options else 'BDF'
        # ! time span
        time_span = solver_options.get(
            'time_span', (0, 100)
        ) if solver_options else (0, 100)
        # ! relative tolerance
        rtol = solver_options.get('rtol', 1e-6) if solver_options else 1e-6
        # ! absolute tolerance
        atol = solver_options.get('atol', 1e-9) if solver_options else 1e-9

        # SECTION: Create source
        source = Source(
            model_source=model_source,
            component_key=component_key,
        )

        # SECTION: create batch reactor model
        GasBatchReactor_ = GasBatchReactor(
            components=components,
            source=source,
            model_inputs=model_inputs,
            reactor_inputs=reactor_inputs,
            reaction_rates=reaction_rates,
            component_key=component_key,
            **kwargs,
        )

        # SECTION: Solve ode
        # NOTE: create function
        def fun(t, y):
            return GasBatchReactor_.rhs(t, y)

        # NOTE: build initial conditions
        y0 = GasBatchReactor_.build_y0()

        # NOTE: solve ode
        sol = solve_ivp(
            fun,
            time_span,
            y0,
            method=method,
            rtol=rtol,
            atol=atol,
        )

        # NOTE: process results
        # ! solver success
        if not sol.success:
            logger.error(f"ODE solver failed: {sol.message}")
            return None

        # ! extract time and state variables
        t = sol.t
        y = sol.y

        return BatchReactorResult(
            time=t,
            state=y,
            success=sol.success,
            message=sol.message,
        )
    except Exception as e:
        logger.error(f"Error during batch reactor simulation: {e}")
