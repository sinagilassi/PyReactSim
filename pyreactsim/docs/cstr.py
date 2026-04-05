import logging
from scipy.integrate import solve_ivp
from typing import Any, Dict, Optional, cast
from pythermodb_settings.models import ComponentKey
# locals
from ..core.cstrc import CSTRReactorCore
from ..core.gas_cstr import GasCSTRReactor
from ..core.liquid_cstr import LiquidCSTRReactor
from ..models.cstr import CSTRReactorOptions
from ..models.cstr import CSTRReactorResult
from ..sources.thermo_source import ThermoSource

# NOTE: set logger
logger = logging.getLogger(__name__)


class CSTRReactor:
    """
    CSTR reactor interface.

    The current implementation supports gas-phase dynamic ODE simulations.
    """

    def __init__(
        self,
        model_inputs: Dict[str, Any],
        thermo_source: ThermoSource,
        **kwargs,
    ):
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
        self.cstr_reactor_options = cast(
            CSTRReactorOptions,
            thermo_source.reactor_options
        )
        # ! heat transfer options
        self.heat_transfer_options = thermo_source.heat_transfer_options
        # ! phase
        self.phase = self.cstr_reactor_options.phase

        # ! reaction rates
        self.reaction_rates = thermo_source.reaction_rates

        self.cstr_reactor_core = CSTRReactorCore(
            components=self.components,
            model_inputs=model_inputs,
            cstr_reactor_options=self.cstr_reactor_options,
            heat_transfer_options=self.heat_transfer_options,
            component_refs=self.component_refs,
            component_key=cast(ComponentKey, self.component_key),
        )

        # SECTION: Create reactor
        self.reactor: GasCSTRReactor | LiquidCSTRReactor = self._create_reactor()

    # SECTION: Reactor creation method
    def _create_reactor(self) -> GasCSTRReactor | LiquidCSTRReactor:
        if self.phase == "gas":
            return GasCSTRReactor(
                components=self.components,
                reaction_rates=self.reaction_rates,
                thermo_source=self.thermo_source,
                cstr_reactor_core=self.cstr_reactor_core,
                component_key=cast(ComponentKey, self.component_key),
            )
        elif self.phase == "liquid":
            return LiquidCSTRReactor(
                components=self.components,
                reaction_rates=self.reaction_rates,
                thermo_source=self.thermo_source,
                cstr_reactor_core=self.cstr_reactor_core,
                component_key=cast(ComponentKey, self.component_key),
            )

        raise NotImplementedError(
            f"CSTR reactor for phase '{self.phase}' is not implemented yet."
        )

    # SECTION: Simulation method
    def simulate(
        self,
        solver_options: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Optional[CSTRReactorResult]:
        # NOTE: set default solver options
        method = solver_options.get(
            "method",
            "BDF"
        ) if solver_options else "BDF"
        time_span = solver_options.get(
            "time_span",
            (0, 100)
        ) if solver_options else (0, 100)
        rtol = solver_options.get("rtol", 1e-6) if solver_options else 1e-6
        atol = solver_options.get("atol", 1e-9) if solver_options else 1e-9

        # NOTE: define ODE function
        def fun(t, y):
            return self.reactor.rhs(t, y)

        # NOTE: build initial state vector
        y0 = self.reactor.build_y0()

        # NOTE: solve ODEs
        sol = solve_ivp(
            fun,
            time_span,
            y0,
            method=method,
            rtol=rtol,
            atol=atol,
        )

        if not sol.success:
            logger.error(f"ODE solver failed: {sol.message}")
            return None

        return CSTRReactorResult(
            time=sol.t,
            state=sol.y,
            success=sol.success,
            message=sol.message,
        )
