import logging
from scipy.integrate import solve_ivp
from typing import Any, Dict, List, Optional, cast
from pythermodb_settings.models import Component, ComponentKey
from pythermodb_settings.utils import measure_time
from pyThermoLinkDB.models import ModelSource
from pyThermoLinkDB.thermo import Source
# locals
from ..models.heat import HeatTransferOptions
from ..models.br import BatchReactorOptions, BatchReactorResult
from ..models.rate_exp import ReactionRateExpression
from ..core.gas_br import GasBatchReactor
from ..core.gas_brx import GasBatchReactorX
from ..core.liquid_br import LiquidBatchReactor
from ..core.liquid_brx import LiquidBatchReactorX
from ..core.brc import BatchReactorCore
from ..sources.thermo_source import ThermoSource
from ..utils.tools import configure_solver_options

# NOTE: set logger
logger = logging.getLogger(__name__)


class BatchReactor:
    def __init__(
        self,
        model_inputs: Dict[str, Any],
        thermo_source: ThermoSource,
        **kwargs,
    ):
        """
        Initializes the BatchReactor instance with the provided components, model inputs, reactor inputs, reaction rates, model source, and component key.

        Parameters
        ----------
        model_inputs : Dict[str, Any]
            A dictionary of model inputs, where the keys are the names of the inputs and the values are the input values.
            - feed mole: Dict[str, CustomProp]
            - feed temperature: Temperature
            - feed pressure: Pressure
        thermo_source : ThermoSource
            A ThermoSource object containing the thermodynamic source information for the batch reactor simulation.
        """
        # NOTE: set attributes
        self.model_inputs = model_inputs
        self.thermo_source = thermo_source

        # NOTE: components
        self.components = thermo_source.components

        # NOTE: generate component references
        self.component_refs = thermo_source.component_refs

        # NOTE: component key
        self.component_key = thermo_source.component_key

        # NOTE: batch reactor options
        # ! batch reactor options
        self.batch_reactor_options = cast(
            BatchReactorOptions,
            thermo_source.reactor_options
        )
        # ! heat transfer options
        self.heat_transfer_options = thermo_source.heat_transfer_options
        # ! phase
        self.phase = self.batch_reactor_options.phase
        self.modeling_type = self.batch_reactor_options.modeling_type

        # ! reaction rates
        self.reaction_rates = thermo_source.reaction_rates

        # SECTION: Create batch reactor core
        self.batch_reactor_core = BatchReactorCore(
            components=self.components,
            model_inputs=model_inputs,
            batch_reactor_options=self.batch_reactor_options,
            heat_transfer_options=self.heat_transfer_options,
            component_refs=self.component_refs,
            component_key=cast(ComponentKey, self.component_key)
        )

        # SECTION: Create reactor
        self.reactor: GasBatchReactor | GasBatchReactorX | LiquidBatchReactor | LiquidBatchReactorX = self._create_reactor()

    # SECTION: Reactor creation method
    def _create_reactor(self) -> GasBatchReactor | GasBatchReactorX | LiquidBatchReactor | LiquidBatchReactorX:
        if self.phase == "gas" and self.modeling_type == "physical":
            return GasBatchReactor(
                components=self.components,
                reaction_rates=self.reaction_rates,
                thermo_source=self.thermo_source,
                batch_reactor_core=self.batch_reactor_core,
                component_key=cast(ComponentKey, self.component_key),
            )
        elif self.phase == "gas" and self.modeling_type == "scale":
            return GasBatchReactorX(
                components=self.components,
                reaction_rates=self.reaction_rates,
                thermo_source=self.thermo_source,
                batch_reactor_core=self.batch_reactor_core,
                component_key=cast(ComponentKey, self.component_key),
            )
        elif self.phase == "liquid" and self.modeling_type == "physical":
            return LiquidBatchReactor(
                components=self.components,
                reaction_rates=self.reaction_rates,
                thermo_source=self.thermo_source,
                batch_reactor_core=self.batch_reactor_core,
                component_key=cast(ComponentKey, self.component_key),
            )
        elif self.phase == "liquid" and self.modeling_type == "scale":
            return LiquidBatchReactorX(
                components=self.components,
                reaction_rates=self.reaction_rates,
                thermo_source=self.thermo_source,
                batch_reactor_core=self.batch_reactor_core,
                component_key=cast(ComponentKey, self.component_key),
            )
        else:
            raise NotImplementedError(
                f"Batch reactor for phase '{self.phase}' and modeling_type '{self.modeling_type}' is not implemented yet."
            )

    # SECTION: Simulation method
    @measure_time
    def simulate(
            self,
            time_span: tuple[float, float],
            solver_options: Optional[Dict[str, Any]] = None,
            **kwargs
    ) -> Optional[BatchReactorResult]:
        """
        Run batch reactor simulation over time using scipy's solve_ivp ODE solver.

        Parameters
        ----------
        time_span : tuple[float, float]
            A tuple specifying the start and end times for the simulation (e.g., (0, 100)).
        solver_options : Optional[Dict[str, Any]], optional
            A dictionary of solver options to pass to `scipy.integrate.solve_ivp`. If None, default options will be used.
            Supported options include:
            - method: ODE solver method (e.g., 'BDF', 'RK45', etc.)
            - rtol: Relative tolerance for the solver
            - atol: Absolute tolerance for the solver
            - first_step: Initial step size for the solver
            - max_step: Maximum step size for the solver
        **kwargs
            Additional keyword arguments.
            - mode : Literal['silent', 'log', 'attach'], optional
                Mode for time measurement logging. Default is 'silent'.

        Returns
        -------
        Optional[BatchReactorResult]
            A BatchReactorResult object containing the simulation results, or None if the solver failed.

        Notes
        -----
        - The method uses `scipy.integrate.solve_ivp` to solve the ODEs defined by the batch reactor model.
        - The `mode` keyword argument can be used to control how the execution time is logged:
            - 'silent': No logging of execution time.
            - 'log': Logs the execution time to the logger.
            - 'attach': Logs the execution time and attaches it to the result object.
        - The solver options can be customized by passing a dictionary to `solver_options`. If not provided, default options will be used for the solver. The default values are as:
            - method: 'BDF'
            - rtol: 1e-6
            - atol: 1e-9
            - first_step: 1e-8
            - max_step: 1e-3
        """
        # NOTE: Solver options
        configured_solver_options = configure_solver_options(
            solver_options=solver_options
        )

        # NOTE: run simulation
        # >>> create function
        def fun(t, y):
            if isinstance(self.reactor, (GasBatchReactor, LiquidBatchReactor)):
                return self.reactor.rhs(t, y)
            elif isinstance(self.reactor, (GasBatchReactorX, LiquidBatchReactorX)):
                return self.reactor.rhs_scaled(t, y)
            else:
                raise NotImplementedError(
                    f"ODE function for reactor type '{type(self.reactor)}' is not implemented yet."
                )

        # NOTE: build initial conditions
        if isinstance(self.reactor, (GasBatchReactor, LiquidBatchReactor)):
            y0 = self.reactor.build_y0()
        elif isinstance(self.reactor, (GasBatchReactorX, LiquidBatchReactorX)):
            y0 = self.reactor.build_y0_scaled()
        else:
            raise NotImplementedError(
                f"Initial condition builder for reactor type '{type(self.reactor)}' is not implemented yet."
            )

        # NOTE: solve ode
        sol = solve_ivp(
            fun,
            time_span,
            y0,
            **configured_solver_options,
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
