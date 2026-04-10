import logging
import numpy as np
from scipy.integrate import solve_ivp
from typing import Any, Dict, Optional, cast
from pythermodb_settings.models import ComponentKey
from pythermodb_settings.utils import measure_time
# locals
from ..core.gas_pbr import GasPBRReactor
from ..core.gas_pbrx import GasPBRReactorX
from ..core.liquid_pbr import LiquidPBRReactor
from ..core.liquid_pbrx import LiquidPBRReactorX
from ..core.pbrc import PBRReactorCore
from ..models.pbr import PBRReactorOptions, PBRReactorResult
from ..sources.thermo_source import ThermoSource
from ..utils.tools import configure_solver_options

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
        self.modeling_type = self.pbr_reactor_options.modeling_type

        self.pbr_reactor_core = PBRReactorCore(
            components=self.components,
            model_inputs=model_inputs,
            pbr_reactor_options=self.pbr_reactor_options,
            heat_transfer_options=self.heat_transfer_options,
            component_refs=self.component_refs,
            component_key=cast(ComponentKey, self.component_key),
        )

        self.reactor: GasPBRReactor | GasPBRReactorX | LiquidPBRReactor | LiquidPBRReactorX = self._create_reactor()

    # SECTION: reactor creation and simulation methods
    def _create_reactor(self) -> GasPBRReactor | GasPBRReactorX | LiquidPBRReactor | LiquidPBRReactorX:
        # check phase and modeling type to determine reactor class
        if (
            self.phase == "gas" and
            self.modeling_type == "physical"
        ):
            return GasPBRReactor(
                components=self.components,
                reaction_rates=self.reaction_rates,
                thermo_source=self.thermo_source,
                pbr_reactor_core=self.pbr_reactor_core,
                component_key=cast(ComponentKey, self.component_key),
            )
        elif (
            self.phase == "gas" and
            self.modeling_type == "scale"
        ):
            return GasPBRReactorX(
                components=self.components,
                reaction_rates=self.reaction_rates,
                thermo_source=self.thermo_source,
                pbr_reactor_core=self.pbr_reactor_core,
                component_key=cast(ComponentKey, self.component_key),
            )
        elif (
            self.phase == "liquid" and
            self.modeling_type == "physical"
        ):
            return LiquidPBRReactor(
                components=self.components,
                reaction_rates=self.reaction_rates,
                thermo_source=self.thermo_source,
                pbr_reactor_core=self.pbr_reactor_core,
                component_key=cast(ComponentKey, self.component_key),
            )
        elif (
            self.phase == "liquid" and
            self.modeling_type == "scale"
        ):
            return LiquidPBRReactorX(
                components=self.components,
                reaction_rates=self.reaction_rates,
                thermo_source=self.thermo_source,
                pbr_reactor_core=self.pbr_reactor_core,
                component_key=cast(ComponentKey, self.component_key),
            )

        raise NotImplementedError(
            f"PBR reactor for phase '{self.phase}' and modeling_type '{self.modeling_type}' is not implemented yet."
        )

    # NOTE: simulation method
    @measure_time
    def simulate(
        self,
        volume_span: tuple[float, float],
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
            - max_step: maximum step size in m3 (optional)
            - first_step: initial step size in m3 (optional)
            - dense_output: whether to compute a continuous solution (optional)
        **kwargs
            Additional keyword arguments.
            - mode : Literal['silent', 'log', 'attach'], optional
                Mode for time measurement logging. Default is 'silent'.

        Returns
        -------
        Optional[PBRReactorResult]
            PBRReactorResult containing volume points, state trajectories, and solver information. Returns None if the solver fails.

        Notes
        -----
        - The method uses `scipy.integrate.solve_ivp` to solve the ODEs defined by PBR reactor model.
        - The `mode` keyword argument can be used to control how the execution time is logged:
            - 'silent': No logging of execution time.
            - 'log': Logs the execution time to the logger.
            - 'attach': Logs the execution time and attaches it to the result object.
        - The solver options can be customized by passing a dictionary to `solver_options`. If not provided, default options will be used for the solver. The default values are as:
            - method: 'BDF'
            - rtol: 1e-6
            - atol: 1e-9
        """
        # NOTE: set default solver options if not provided
        configured_solver_options = configure_solver_options(
            solver_options=solver_options
        )

        # NOTE: define ODE function for PFR simulation

        def fun(V, y):
            '''ODE function for PBR simulation.'''
            if isinstance(self.reactor, (GasPBRReactor, LiquidPBRReactor)):
                return self.reactor.rhs(V, y)
            elif isinstance(self.reactor, (GasPBRReactorX, LiquidPBRReactorX)):
                return self.reactor.rhs_scaled(V, y)
            else:
                raise NotImplementedError(
                    f"ODE function for reactor type '{type(self.reactor)}' is not implemented yet."
                )

        # NOTE: get initial conditions
        # >> check physical vs scale model to determine y0
        if (
            self.modeling_type == "physical" and
            isinstance(self.reactor, (GasPBRReactor, LiquidPBRReactor))
        ):
            y0 = self.reactor.build_y0()
        elif (
            self.modeling_type == "scale" and
            isinstance(self.reactor, (GasPBRReactorX, LiquidPBRReactorX))
        ):
            y0 = self.reactor.build_y0_scaled()
        else:
            raise NotImplementedError(
                f"Initial condition builder for modeling type '{self.modeling_type}' and reactor type '{type(self.reactor)}' is not implemented yet."
            )

        # NOTE: run ODE solver
        sol = solve_ivp(
            fun,
            volume_span,
            y0,
            **configured_solver_options,
        )

        # NOTE: check solver success and return results
        if not sol.success:
            logger.error(f"PBR ODE solver failed: {sol.message}")
            return None

        return PBRReactorResult(
            volume=sol.t,
            state=sol.y,
            success=sol.success,
            message=sol.message,
        )

    # NOTE: simulation method using diffeqpy
    @measure_time
    def simulate_diffeqpy(
            self,
            solver_options: Optional[Dict[str, Any]] = None,
            **kwargs
    ) -> Optional[PBRReactorResult]:
        """Run steady-state PBR simulation along reactor-volume coordinate using diffeqpy."""
        from diffeqpy import de as _de

        # diffeqpy.de replaces itself dynamically from Julia packages at runtime.
        de: Any = cast(Any, _de)

        # NOTE: set default solver options if not provided
        # solver options
        method = solver_options.get(
            "method", "Rodas5"
        ) if solver_options else "Rodas5"
        volume_span = (
            solver_options.get(
                "volume_span", (0.0, self.pbr_reactor_core.reactor_volume_value))
            if solver_options else
            (0.0, self.pbr_reactor_core.reactor_volume_value)
        )
        rtol = solver_options.get("rtol", 1e-6) if solver_options else 1e-6
        atol = solver_options.get("atol", 1e-9) if solver_options else 1e-9

        # NOTE: define ODE function for PBR simulation
        # ! in-place RHS for diffeqpy: f(du, u, p, t)
        # This avoids Julia type-instability errors from out-of-place Python returns.
        def fun(du, u, p, V):
            '''In-place ODE function for PBR simulation.'''
            # NOTE: Julia vectors from diffeqpy are safest to consume by
            # element-wise conversion instead of direct np.asarray(u, dtype=float).
            u_vec = np.array([float(ui) for ui in u], dtype=float)

            if isinstance(self.reactor, (GasPBRReactor, LiquidPBRReactor)):
                rhs = self.reactor.rhs(V, u_vec)
            elif isinstance(self.reactor, (GasPBRReactorX, LiquidPBRReactorX)):
                rhs = self.reactor.rhs_scaled(V, u_vec)
            else:
                raise NotImplementedError(
                    f"ODE function for reactor type '{type(self.reactor)}' is not implemented yet."
                )

            rhs = np.asarray(rhs, dtype=float)
            for i in range(len(rhs)):
                du[i] = rhs[i]

        # NOTE: get initial conditions
        # >> check physical vs scale model to determine y0
        if (
            self.modeling_type == "physical" and
            isinstance(self.reactor, (GasPBRReactor, LiquidPBRReactor))
        ):
            y0 = self.reactor.build_y0()
        elif (
            self.modeling_type == "scale" and
            isinstance(self.reactor, (GasPBRReactorX, LiquidPBRReactorX))
        ):
            y0 = self.reactor.build_y0_scaled()
        else:
            raise NotImplementedError(
                f"Initial condition builder for modeling type '{self.modeling_type}' and reactor type '{type(self.reactor)}' is not implemented yet."
            )

        # NOTE: run diffeqpy solver
        # >> create ODE problem
        u0 = [float(v) for v in np.asarray(y0, dtype=float).ravel()]
        tspan = (float(volume_span[0]), float(volume_span[1]))
        prob = de.ODEProblem(fun, u0, tspan, None)

        # NOTE: set solver options
        # choose solver
        method_name = str(method).strip().lower()
        if method_name == "rodas5":
            alg = de.Rodas5(autodiff=False)
        elif method_name == "cvode_bdf":
            alg = de.CVODE_BDF()
        elif method_name == "radauiia5":
            alg = de.RadauIIA5()
        else:
            raise ValueError(f"Unsupported diffeqpy method: {method}")

        # >> solve with specific solver
        sol = de.solve(prob, alg, reltol=rtol, abstol=atol)

        success = str(sol.retcode) == "Success"
        if not success:
            logger.error(f"PBR diffeqpy solver failed: {sol.retcode}")
            return None

        state = np.asarray(sol.u, dtype=float)
        if state.ndim == 1:
            state = state.reshape(1, -1)
        elif state.shape[0] == len(sol.t):
            state = state.T

        return PBRReactorResult(
            volume=sol.t,
            state=state,
            success=success,
            message=str(sol.retcode),
        )
