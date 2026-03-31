import logging
import numpy as np
from scipy.integrate import solve_ivp
from typing import Any, Dict, List, Optional, cast
from pythermodb_settings.models import Component, ComponentKey
from pyThermoLinkDB.models import ModelSource
from pyThermoLinkDB.thermo import Source

# locals
from ..models.streams import HeatExchanger
from ..models.br import BatchReactorOptions, BatchReactorResult
from ..models.rate_exp import ReactionRateExpression
from ..core.gas_br import GasBatchReactor
from ..core.brc import BatchReactorCore
from ..sources.thermo_source import ThermoSource
from ..utils.opt_tools import set_component_X
from ..utils.tools import generate_component_references

# NOTE: set logger
logger = logging.getLogger(__name__)


class BatchReactor:
    def __init__(
        self,
        components: List[Component],
        input_stream: Dict[str, Any],
        model_source: ModelSource,
        model_inputs: Dict[str, Any],
        batch_reactor_option: BatchReactorOptions,
        reactor_inputs: Dict[str, Any],
        reaction_rates: List[ReactionRateExpression],
        heat_exchanger: HeatExchanger,
        component_key: ComponentKey,
        **kwargs,
    ):
        """
        Initializes the BatchReactor instance with the provided components, model inputs, reactor inputs, reaction rates, model source, and component key.

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
        reaction_rates : List[ReactionRateExpression]
            A list of reaction rate expressions, where the keys are the names
            of the reactions and the values are the ReactionRateExpression objects.
        model_source : ModelSource
            A ModelSource object containing the source of the model to be used
            in the simulation.
        component_key : ComponentKey
            A ComponentKey object representing the key to be used for the components in the model source.
        """
        # NOTE: set attributes
        self.components = components
        self.model_inputs = model_inputs
        self.batch_reactor_option = batch_reactor_option
        self.reactor_inputs = reactor_inputs
        self.reaction_rates = reaction_rates
        self.model_source = model_source
        self.heat_exchanger = heat_exchanger
        self.component_key = component_key

        # NOTE: generate component references
        (
            self.component_ids,
            self.component_formula_state,
            self.component_mapper,
            self.component_id_to_index
        ) = generate_component_references(
            components=components,
            component_key=component_key
        )

        # ! component
        self.component_refs = {
            "component_ids": self.component_ids,
            "component_formula_state": self.component_formula_state,
            "component_mapper": self.component_mapper,
            "component_id_to_index": self.component_id_to_index
        }

        # ! phase
        self.phase = self.batch_reactor_option.phase

        # SECTION: Create source
        source = Source(
            model_source=model_source,
            component_key=component_key,
        )

        # SECTION: Create batch reactor core
        self.batch_reactor_core = BatchReactorCore(
            components=components,
            input_stream=input_stream,
            batch_reactor_options=batch_reactor_option,
            reactor_inputs=reactor_inputs,
            heat_exchanger=heat_exchanger,
            component_key=component_key,
        )

        # SECTION: Create thermo source
        self.thermo_source = ThermoSource(
            components=components,
            source=source,
            model_inputs=model_inputs,
            batch_reactor_options=batch_reactor_option,
            reaction_rates=reaction_rates,
            component_key=component_key,
            component_refs=self.component_refs
        )

        # SECTION: Create reactor
        self.reactor: GasBatchReactor = self._create_reactor(
            source=source,
            thermo_source=self.thermo_source
        )

    # SECTION: Reactor creation method
    def _create_reactor(self, source, thermo_source) -> GasBatchReactor:
        if self.phase == "gas":
            gas_br = GasBatchReactor(
                components=self.components,
                source=source,
                model_inputs=self.model_inputs,
                reactor_inputs=self.reactor_inputs,
                reaction_rates=self.reaction_rates,
                component_key=cast(ComponentKey, self.component_key),
                thermo_source=thermo_source,
                batch_reactor_core=self.batch_reactor_core
            )

            return gas_br
        else:
            raise NotImplementedError(
                f"Batch reactor for phase '{self.phase}' is not implemented yet.")

    # SECTION: Simulation method
    def simulate(
            self,

            heat_exchanger: Optional[HeatExchanger] = None,
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
