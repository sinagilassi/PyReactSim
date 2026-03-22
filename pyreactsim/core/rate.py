# Import libs
import logging
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, TypeAlias, Callable
from pythermodb_settings.models import Component, CustomProperty, Pressure, Temperature, Volume, CustomProp
from pyreactlab_core.models.reaction import Reaction
# locals
from ..models.reaction_exp import rArgs, rParams, rRet, X, Xi, ReactionRateExpression

# NOTE: logger setup
logger = logging.getLogger(__name__)


class ReactionRate:
    def __init__(
        self,
        basis: Literal['concentration', 'pressure'],
        components: List[Component],
        reaction: Reaction,
        parameters: rParams,
        arguments: rArgs,
        returns: rRet,
        eq: Callable[[Xi, rArgs, rParams], rRet]
    ):
        self.basis = basis
        self.components = components
        self.reaction = reaction
        self.parameters = parameters
        self.arguments = arguments
        self.returns = returns
        self.eq = eq

    def calc(
            self,
            xi: Dict[str, CustomProperty],
            *,
            args: Optional[rArgs] = None,
            temperature: Optional[Temperature] = None,
            pressure: Optional[Pressure] = None
    ) -> rRet:
        # NOTE: check args
        if args is None:
            args = {}

        # NOTE: Update xi
        # convert xi to Xi
        xi_converted = {}

        for comp in self.components:
            if comp.name in xi:
                xi_converted[comp.name] = X(
                    component=comp,
                    value=xi[comp.name].value,
                    unit=xi[comp.name].unit
                )
            else:
                logger.warning(
                    f"Component {comp.name} not found in xi. Setting value to 0.")
                xi_converted[comp.name] = X(
                    component=comp,
                    value=0,
                    unit=""
                )

        # NOTE: Add Temperature and Pressure to args if provided
        if temperature is not None:
            args['T'] = CustomProperty(
                value=temperature.value,
                unit=temperature.unit,
                symbol="T"
            )

        if pressure is not None:
            args['P'] = CustomProperty(
                value=pressure.value,
                unit=pressure.unit,
                symbol="P"
            )

        # NOTE: Calculate rate
        return self.eq(
            xi_converted,
            args,
            self.parameters
        )
