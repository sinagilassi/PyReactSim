import logging
from scipy.integrate import solve_ivp
from typing import Any, Dict, Optional, cast
from pythermodb_settings.models import ComponentKey
# locals
from ..core.gas_pfr import GasPFRReactor
from ..core.liquid_pfr import LiquidPFRReactor
from ..core.pfrc import PFRReactorCore
from ..models.pfr import PFRReactorOptions, PFRReactorResult
from ..sources.thermo_source import ThermoSource

# NOTE: set logger
logger = logging.getLogger(__name__)


class PFRReactor:
    """
    PFR reactor interface.
    """

    def __init__(
        self,
        model_inputs: Dict[str, Any],
        thermo_source: ThermoSource,
        **kwargs,
    ):
        self.model_inputs = model_inputs
        self.thermo_source = thermo_source

        self.components = thermo_source.components
        self.component_refs = thermo_source.component_refs
        self.component_key = thermo_source.component_key

        self.pfr_reactor_options = cast(
            PFRReactorOptions,
            thermo_source.reactor_options
        )
        self.heat_transfer_options = thermo_source.heat_transfer_options
        self.phase = self.pfr_reactor_options.phase
        self.reaction_rates = thermo_source.reaction_rates

        self.pfr_reactor_core = PFRReactorCore(
            components=self.components,
            model_inputs=model_inputs,
            pfr_reactor_options=self.pfr_reactor_options,
            heat_transfer_options=self.heat_transfer_options,
            component_refs=self.component_refs,
            component_key=cast(ComponentKey, self.component_key),
        )

        self.reactor: GasPFRReactor | LiquidPFRReactor = self._create_reactor()

    def _create_reactor(self) -> GasPFRReactor | LiquidPFRReactor:
        if self.phase == "gas":
            return GasPFRReactor(
                components=self.components,
                reaction_rates=self.reaction_rates,
                thermo_source=self.thermo_source,
                pfr_reactor_core=self.pfr_reactor_core,
                component_key=cast(ComponentKey, self.component_key),
            )
        if self.phase == "liquid":
            return LiquidPFRReactor(
                components=self.components,
                reaction_rates=self.reaction_rates,
                thermo_source=self.thermo_source,
                pfr_reactor_core=self.pfr_reactor_core,
                component_key=cast(ComponentKey, self.component_key),
            )

        raise NotImplementedError(
            f"PFR reactor for phase '{self.phase}' is not implemented yet."
        )

    def simulate(
        self,
        solver_options: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Optional[PFRReactorResult]:
        method = solver_options.get("method", "BDF") if solver_options else "BDF"
        volume_span = (
            solver_options.get("volume_span", (0.0, self.pfr_reactor_core.reactor_volume_value))
            if solver_options else
            (0.0, self.pfr_reactor_core.reactor_volume_value)
        )
        rtol = solver_options.get("rtol", 1e-6) if solver_options else 1e-6
        atol = solver_options.get("atol", 1e-9) if solver_options else 1e-9

        def fun(V, y):
            return self.reactor.rhs(V, y)

        y0 = self.reactor.build_y0()

        sol = solve_ivp(
            fun,
            volume_span,
            y0,
            method=method,
            rtol=rtol,
            atol=atol,
        )

        if not sol.success:
            logger.error(f"PFR ODE solver failed: {sol.message}")
            return None

        return PFRReactorResult(
            volume=sol.t,
            state=sol.y,
            success=sol.success,
            message=sol.message,
        )
