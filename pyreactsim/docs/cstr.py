import logging
from scipy.integrate import solve_ivp
from typing import Any, Dict, Optional, cast
from pythermodb_settings.models import ComponentKey
from pythermodb_settings.utils import measure_time
# locals
from ..core.cstrc import CSTRReactorCore
from ..core.gas_cstr import GasCSTRReactor
from ..core.gas_cstrx import GasCSTRReactorX
from ..core.liquid_cstr import LiquidCSTRReactor
from ..core.liquid_cstrx import LiquidCSTRReactorX
from ..models.cstr import CSTRReactorOptions
from ..models.cstr import CSTRReactorResult
from ..sources.thermo_source import ThermoSource
from ..utils.tools import configure_solver_options

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
        self.modeling_type = self.cstr_reactor_options.modeling_type

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
        self.reactor: GasCSTRReactor | GasCSTRReactorX | LiquidCSTRReactor | LiquidCSTRReactorX = self._create_reactor()

    # SECTION: Reactor creation method
    def _create_reactor(self) -> GasCSTRReactor | GasCSTRReactorX | LiquidCSTRReactor | LiquidCSTRReactorX:
        if self.phase == "gas" and self.modeling_type == "physical":
            return GasCSTRReactor(
                components=self.components,
                reaction_rates=self.reaction_rates,
                thermo_source=self.thermo_source,
                cstr_reactor_core=self.cstr_reactor_core,
                component_key=cast(ComponentKey, self.component_key),
            )
        elif self.phase == "gas" and self.modeling_type == "scale":
            return GasCSTRReactorX(
                components=self.components,
                reaction_rates=self.reaction_rates,
                thermo_source=self.thermo_source,
                cstr_reactor_core=self.cstr_reactor_core,
                component_key=cast(ComponentKey, self.component_key),
            )
        elif self.phase == "liquid" and self.modeling_type == "physical":
            return LiquidCSTRReactor(
                components=self.components,
                reaction_rates=self.reaction_rates,
                thermo_source=self.thermo_source,
                cstr_reactor_core=self.cstr_reactor_core,
                component_key=cast(ComponentKey, self.component_key),
            )
        elif self.phase == "liquid" and self.modeling_type == "scale":
            return LiquidCSTRReactorX(
                components=self.components,
                reaction_rates=self.reaction_rates,
                thermo_source=self.thermo_source,
                cstr_reactor_core=self.cstr_reactor_core,
                component_key=cast(ComponentKey, self.component_key),
            )

        raise NotImplementedError(
            f"CSTR reactor for phase '{self.phase}' and modeling_type '{self.modeling_type}' is not implemented yet."
        )

    # SECTION: Simulation method
    @measure_time
    def simulate(
        self,
        time_span: tuple[float, float],
        solver_options: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Optional[CSTRReactorResult]:
        """
        Run CSTR simulation over the specified time span with given solver options.

        Parameters
        ----------
        time_span : tuple[float, float]
            Start and end times for the simulation.
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
        Optional[CSTRReactorResult]
            CSTRReactorResult containing time points, state trajectories, and solver information. Returns None if the solver fails.

        - The method uses `scipy.integrate.solve_ivp` to solve the ODEs defined by the CSTR reactor model.
        - The `mode` keyword argument can be used to control how the execution time is logged:
            - 'silent': No logging of execution time.
            - 'log': Logs the execution time to the logger.
            - 'attach': Logs the execution time and attaches it to the result object.
        - The solver options can be customized by passing a dictionary to `solver_options`. If not provided, default options will be used for the solver. The default values are as:
            - method: 'BDF'
            - rtol: 1e-6
            - atol: 1e-9
        """
        # NOTE: set default solver options
        configured_solver_options = configure_solver_options(
            solver_options=solver_options
        )

        # NOTE: define ODE function
        def fun(t, y):
            if isinstance(self.reactor, (GasCSTRReactor, LiquidCSTRReactor)):
                return self.reactor.rhs(t, y)
            elif isinstance(self.reactor, (GasCSTRReactorX, LiquidCSTRReactorX)):
                return self.reactor.rhs_scaled(t, y)
            else:
                raise NotImplementedError(
                    f"ODE function for reactor type '{type(self.reactor)}' is not implemented yet."
                )

        # NOTE: build initial state vector
        if isinstance(self.reactor, (GasCSTRReactor, LiquidCSTRReactor)):
            y0 = self.reactor.build_y0()
        elif isinstance(self.reactor, (GasCSTRReactorX, LiquidCSTRReactorX)):
            y0 = self.reactor.build_y0_scaled()
        else:
            raise NotImplementedError(
                f"Initial condition builder for reactor type '{type(self.reactor)}' is not implemented yet."
            )

        # NOTE: solve ODEs
        sol = solve_ivp(
            fun,
            time_span,
            y0,
            **configured_solver_options,
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
