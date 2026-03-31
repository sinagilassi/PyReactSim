import logging
from scipy.integrate import solve_ivp
from typing import Any, Dict, List, Optional, cast
from pythermodb_settings.models import Component, ComponentKey
from pyThermoLinkDB.models import ModelSource
from pyThermoLinkDB.thermo import Source
# locals
from ..models.heat import HeatTransferOptions
from ..models.br import BatchReactorOptions, BatchReactorResult
from ..models.rate_exp import ReactionRateExpression
from ..core.gas_br import GasBatchReactor
from ..core.liquid_br import LiquidBatchReactor
from ..core.brc import BatchReactorCore
from ..sources.thermo_source import ThermoSource

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
        self.batch_reactor_options = thermo_source.batch_reactor_options
        # ! heat transfer options
        self.heat_transfer_options = thermo_source.heat_transfer_options
        # ! phase
        self.phase = self.batch_reactor_options.phase

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
        self.reactor: GasBatchReactor | LiquidBatchReactor = self._create_reactor(
            thermo_source=self.thermo_source
        )

    # SECTION: Reactor creation method
    def _create_reactor(self, thermo_source) -> GasBatchReactor | LiquidBatchReactor:
        if self.phase == "gas":
            gas_br = GasBatchReactor(
                components=self.components,
                reaction_rates=self.reaction_rates,
                thermo_source=self.thermo_source,
                batch_reactor_core=self.batch_reactor_core,
                component_key=cast(ComponentKey, self.component_key),
            )

            return gas_br
        elif self.phase == "liquid":
            liquid_br = LiquidBatchReactor(
                components=self.components,
                reaction_rates=self.reaction_rates,
                thermo_source=self.thermo_source,
                batch_reactor_core=self.batch_reactor_core,
                component_key=cast(ComponentKey, self.component_key),
            )

            return liquid_br
        else:
            raise NotImplementedError(
                f"Batch reactor for phase '{self.phase}' is not implemented yet.")

    # SECTION: Simulation method
    def simulate(
            self,
            solver_options: Optional[Dict[str, Any]] = None,
            **kwargs
    ) -> Optional[BatchReactorResult]:
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

        # NOTE: run simulation
        # >>> create function
        def fun(t, y):
            return self.reactor.rhs(t, y)

        # NOTE: build initial conditions
        y0 = self.reactor.build_y0()

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
