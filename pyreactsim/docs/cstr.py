import logging
from scipy.integrate import solve_ivp
from typing import Any, Dict, Optional, cast
from pythermodb_settings.models import ComponentKey
# locals
from ..core.cstrc import CSTRReactorCore
from ..core.gas_cstr import GasCSTRReactor
from ..models.br import BatchReactorOptions
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
        self.model_inputs = model_inputs
        self.thermo_source = thermo_source

        self.components = thermo_source.components
        self.component_refs = thermo_source.component_refs
        self.component_key = thermo_source.component_key

        cstr_reactor_options = kwargs.get("cstr_reactor_options")
        if cstr_reactor_options is None:
            cstr_reactor_options = getattr(thermo_source, "cstr_reactor_options", None)

        if cstr_reactor_options is None:
            batch_opts: BatchReactorOptions | None = getattr(
                thermo_source, "batch_reactor_options", None
            )
            if batch_opts is None:
                raise ValueError(
                    "cstr_reactor_options is required when thermo_source does not provide batch_reactor_options."
                )
            cstr_reactor_options = self._build_cstr_options_from_batch(batch_opts)

        self.cstr_reactor_options = cast(CSTRReactorOptions, cstr_reactor_options)
        self.heat_transfer_options = thermo_source.heat_transfer_options
        self.phase = self.cstr_reactor_options.phase

        self.reaction_rates = thermo_source.reaction_rates

        self.cstr_reactor_core = CSTRReactorCore(
            components=self.components,
            model_inputs=model_inputs,
            cstr_reactor_options=self.cstr_reactor_options,
            heat_transfer_options=self.heat_transfer_options,
            component_refs=self.component_refs,
            component_key=cast(ComponentKey, self.component_key),
        )

        self.reactor = self._create_reactor()

    def _build_cstr_options_from_batch(
        self,
        batch_opts: BatchReactorOptions
    ) -> CSTRReactorOptions:
        if batch_opts.operation_mode == "constant_pressure":
            case = 1 if self.heat_transfer_options.heat_transfer_mode == "isothermal" else 5
        elif batch_opts.operation_mode == "constant_volume":
            case = 2 if self.heat_transfer_options.heat_transfer_mode == "isothermal" else 6
        else:
            case = 7 if self.heat_transfer_options.heat_transfer_mode == "non-isothermal" else 4

        return CSTRReactorOptions(
            phase=batch_opts.phase,
            case=case,
            gas_model=batch_opts.gas_model,
            gas_heat_capacity_mode=batch_opts.gas_heat_capacity_mode,
            liquid_heat_capacity_mode=batch_opts.liquid_heat_capacity_mode,
            liquid_density_mode=batch_opts.liquid_density_mode,
        )

    def _create_reactor(self) -> GasCSTRReactor:
        if self.phase == "gas":
            return GasCSTRReactor(
                components=self.components,
                reaction_rates=self.reaction_rates,
                thermo_source=self.thermo_source,
                cstr_reactor_core=self.cstr_reactor_core,
                component_key=cast(ComponentKey, self.component_key),
            )

        raise NotImplementedError(
            f"CSTR reactor for phase '{self.phase}' is not implemented yet."
        )

    def simulate(
        self,
        solver_options: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Optional[CSTRReactorResult]:
        method = solver_options.get("method", "BDF") if solver_options else "BDF"
        time_span = solver_options.get("time_span", (0, 100)) if solver_options else (0, 100)
        rtol = solver_options.get("rtol", 1e-6) if solver_options else 1e-6
        atol = solver_options.get("atol", 1e-9) if solver_options else 1e-9

        def fun(t, y):
            return self.reactor.rhs(t, y)

        y0 = self.reactor.build_y0()

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
