import logging
from scipy.integrate import solve_ivp
from typing import Any, Dict, Optional, cast
from pythermodb_settings.models import ComponentKey
# locals
from ..core.gas_pbr import GasPBRReactor
from ..core.liquid_pbr import LiquidPBRReactor
from ..core.pbrc import PBRReactorCore
from ..models.pbr import PBRReactorOptions, PBRReactorResult
from ..sources.thermo_source import ThermoSource

# NOTE: set logger
logger = logging.getLogger(__name__)


class PBRReactor:
    """
    PBR reactor interface.
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

        self.pbr_reactor_options = cast(
            PBRReactorOptions,
            thermo_source.reactor_options
        )
        self.heat_transfer_options = thermo_source.heat_transfer_options
        self.phase = self.pbr_reactor_options.phase
        self.reaction_rates = thermo_source.reaction_rates

        self.pbr_reactor_core = PBRReactorCore(
            components=self.components,
            model_inputs=model_inputs,
            pbr_reactor_options=self.pbr_reactor_options,
            heat_transfer_options=self.heat_transfer_options,
            component_refs=self.component_refs,
            component_key=cast(ComponentKey, self.component_key),
        )

        self.reactor: GasPBRReactor | LiquidPBRReactor = self._create_reactor()

    def _create_reactor(self) -> GasPBRReactor | LiquidPBRReactor:
        if self.phase == "gas":
            return GasPBRReactor(
                components=self.components,
                reaction_rates=self.reaction_rates,
                thermo_source=self.thermo_source,
                pbr_reactor_core=self.pbr_reactor_core,
                component_key=cast(ComponentKey, self.component_key),
            )
        if self.phase == "liquid":
            return LiquidPBRReactor(
                components=self.components,
                reaction_rates=self.reaction_rates,
                thermo_source=self.thermo_source,
                pbr_reactor_core=self.pbr_reactor_core,
                component_key=cast(ComponentKey, self.component_key),
            )

        raise NotImplementedError(
            f"PBR reactor for phase '{self.phase}' is not implemented yet."
        )

    def simulate(
        self,
        solver_options: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Optional[PBRReactorResult]:
        """
        Run steady-state PBR simulation along reactor-volume coordinate.

        Parameters
        ----------
        solver_options : Optional[Dict[str, Any]]
            ODE solver controls. Supported keys include:
            - method: scipy solver method (default "BDF")
            - volume_span: integration interval in m3 (default [0, V_R])
            - rtol: relative tolerance
            - atol: absolute tolerance
        """
        # NOTE: set default solver options if not provided
        # ! method
        method = solver_options.get(
            "method", "BDF") if solver_options else "BDF"
        # ! volume span
        volume_span = (
            solver_options.get(
                "volume_span", (0.0, self.pbr_reactor_core.reactor_volume_value))
            if solver_options else
            (0.0, self.pbr_reactor_core.reactor_volume_value)
        )
        # ! tolerances
        rtol = solver_options.get("rtol", 1e-6) if solver_options else 1e-6
        atol = solver_options.get("atol", 1e-9) if solver_options else 1e-9

        # ! max step
        max_step = solver_options.get(
            "max_step", None
        ) if solver_options else None

        # ! dense output
        dense_output = solver_options.get(
            "dense_output", None
        ) if solver_options else None

        # NOTE: create kwargs
        kwargs = {
            "method": method,
            "volume_span": volume_span,
            "rtol": rtol,
            "atol": atol,
        }

        # >> max step is optional and only added if not inf
        if max_step is not None:
            kwargs["max_step"] = max_step

        # >> dense output
        if dense_output is not None:
            kwargs["dense_output"] = dense_output

        # NOTE: define ODE function for PFR simulation

        def fun(V, y):
            return self.reactor.rhs(V, y)

        y0 = self.reactor.build_y0()

        sol = solve_ivp(
            fun,
            volume_span,
            y0,
            **kwargs,
        )

        if not sol.success:
            logger.error(f"PBR ODE solver failed: {sol.message}")
            return None

        return PBRReactorResult(
            volume=sol.t,
            state=sol.y,
            success=sol.success,
            message=sol.message,
        )
