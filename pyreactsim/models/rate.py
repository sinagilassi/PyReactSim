# Import libs
import logging
from typing import Any, Dict, List, Literal, Optional, Callable
from pythermodb_settings.models import Component, CustomProperty, Pressure, Temperature, Volume, CustomProp, ComponentKey
from pythermodb_settings.utils import set_component_id
from pyreactlab_core.models.reaction import Reaction
# locals
from .reaction_exp import X, rXs, rArgs, rParams, rRet

# NOTE: logger setup
logger = logging.getLogger(__name__)


class ReactionRate:
    def __init__(
        self,
        basis: Literal['concentration', 'pressure'],
        components: List[Component],
        reaction: Reaction,
        params: rParams,
        args: rArgs,
        returns: rRet,
        eq: Callable[[rXs, rArgs, rParams], rRet],
        state: Dict[str, X],
        component_key: ComponentKey,
    ):
        # NOTE: set inputs
        self.basis = basis
        self.components = components
        self.reaction = reaction
        self.params = params
        self.args = args
        self.returns = returns
        self.eq = eq
        self.state = state or {}
        self.component_key = component_key

        # NOTE: component ids
        self.component_ids = [
            set_component_id(comp, self.component_key) for comp in self.components
        ]

    @property
    def component_id_set(self) -> list[str]:
        return self.component_ids

    @property
    def parameters(self):
        return self.params

    @property
    def arguments(self):
        return self.args

    @property
    def return_(self):
        return self.returns

    # NOTE: calculate reaction rate
    def calc(
            self,
            xi: Dict[str, CustomProperty],
            *,
            args: Optional[rArgs] = None,
            temperature: Optional[Temperature] = None,
            pressure: Optional[Pressure] = None
    ) -> rRet:
        """
        Calculate the reaction rate based on the provided state (xi), arguments, temperature, and pressure.

        Parameters
        ----------
        xi : Dict[str, CustomProperty]
            A dictionary of component states keyed by id containing value and unit depending on the basis of the rate expression (e.g., concentration or pressure).
        args : Optional[rArgs], optional
            Additional arguments that may be used in the rate expression, such as temperature and pressure. These will override any temperature and pressure values provided directly to the method.
        temperature : Optional[Temperature],
            The temperature at which to calculate the reaction rate. This will be included in the arguments passed to the rate expression.
        pressure : Optional[Pressure],
            The pressure at which to calculate the reaction rate. This will be included in the arguments passed to the rate expression.

        Returns
        -------
        rRet
            A dictionary containing the calculated reaction rate and any other return values defined by the rate expression.
        """
        # NOTE: Build call args from defaults + call overrides
        call_args: rArgs = {}
        if args is not None:
            call_args.update(args)

        # NOTE: Update xi values while preserving per-component metadata (e.g., order)
        xi_converted: rXs = {}

        # import locally to avoid import cycle during module initialization

        for comp in self.components:
            current_x = self.state.get(comp.name)
            if current_x is None:
                current_x = X(component=comp)

            if comp.name in xi:
                xi_converted[comp.name] = X(
                    component=comp,
                    order=current_x.order,
                    value=xi[comp.name].value,
                    unit=xi[comp.name].unit
                )
            else:
                logger.warning(
                    f"Component {comp.name} not found in xi. Reusing previous/default value.")
                xi_converted[comp.name] = X(
                    component=comp, order=current_x.order, value=current_x.value, unit=current_x.unit)

        # NOTE: Add Temperature and Pressure to args if provided
        if temperature is not None:
            call_args['T'] = CustomProperty(
                value=temperature.value,
                unit=temperature.unit,
                symbol="T"
            )

        if pressure is not None:
            call_args['P'] = CustomProperty(
                value=pressure.value,
                unit=pressure.unit,
                symbol="P"
            )

        # NOTE: persist updated state for subsequent calculations
        self.state.update(xi_converted)

        # NOTE: Calculate rate
        return self.eq(
            xi_converted,
            call_args,
            self.params
        )
