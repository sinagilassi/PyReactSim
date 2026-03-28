# Import libs
import logging
from typing import Any, Dict, List, Literal, Optional, Callable
from pythermodb_settings.models import Component, CustomProperty, Pressure, Temperature, Volume, CustomProp, ComponentKey
from pythermodb_settings.utils import set_component_id
from pyreactlab_core.models.reaction import Reaction
import pycuc
# locals
from .rate_exp_refs import X, rXs, rArgs, rParams, rRet

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
            pressure: Optional[Pressure] = None,
            state_key: Optional[ComponentKey] = None
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

        Notes
        -----
        - State defines for each reaction based on formula-state component key such as CO2-g.
        """
        # NOTE: Build call args from defaults + call overrides
        call_args: rArgs = {}
        if args is not None:
            call_args.update(args)

        # NOTE: Update xi values while preserving per-component metadata (e.g., order)
        xi_converted: rXs = {}

        # NOTE: state key
        state_key = 'Formula-State' if state_key is None else state_key

        for comp in self.components:
            # >>> get state for the component
            # >> create id
            id_ = set_component_id(comp, state_key)
            # >> find state for component
            current_x = self.state.get(id_)
            # >>> check
            if current_x is None:
                current_x = X(component=comp)

            # state unit
            current_x_unit = current_x.unit

            # check if component state is provided in xi
            if id_ in xi.keys():
                # internal state unit and value
                unit_ = xi[id_].unit
                value_ = xi[id_].value

                # ! convert xi value to the same unit as current_x if needed
                if (
                    current_x_unit is not None and
                    current_x_unit != '' and
                    unit_ != current_x_unit
                ):
                    # set
                    value_ = pycuc.convert_from_to(
                        value=value_,
                        from_unit=unit_,
                        to_unit=current_x_unit
                    )

                # upd
                xi_converted[id_] = X(
                    component=comp,
                    order=current_x.order,
                    value=value_,
                    unit=current_x_unit
                )
            else:
                logger.warning(
                    f"Component {id_} not found in xi. Reusing previous/default value."
                )
                xi_converted[id_] = X(
                    component=comp,
                    order=current_x.order,
                    value=current_x.value,
                    unit=current_x.unit
                )

        # NOTE: Add Temperature and Pressure to args if provided
        if temperature is not None:
            # internal temperature unit and value
            value_internal = temperature.value
            unit_internal = temperature.unit

            # check if temperature argument is already provided in args, if not add it
            if 'T' in call_args:
                # external temperature unit and value
                # value_external = call_args['T'].value
                unit_external = call_args['T'].unit

                # ! convert external temperature to the same unit as internal if needed
                if unit_external != unit_internal:
                    value_external_converted = pycuc.convert_from_to(
                        value=value_internal,
                        from_unit=unit_internal,
                        to_unit=unit_external
                    )

                    # update call args with converted temperature
                    call_args['T'] = CustomProperty(
                        value=value_external_converted,
                        unit=unit_external,
                        symbol="T"
                    )
            else:
                call_args['T'] = CustomProperty(
                    value=temperature.value,
                    unit=temperature.unit,
                    symbol="T"
                )

        if pressure is not None:
            # internal pressure unit and value
            value_internal = pressure.value
            unit_internal = pressure.unit

            # check if pressure argument is already provided in args, if not add it
            if 'P' in call_args:
                # external pressure unit and value
                # value_external = call_args['P'].value
                unit_external = call_args['P'].unit

                # ! convert external pressure to the same unit as internal if needed
                if unit_external != unit_internal:
                    value_external_converted = pycuc.convert_from_to(
                        value=value_internal,
                        from_unit=unit_internal,
                        to_unit=unit_external
                    )

                    # update call args with converted pressure
                    call_args['P'] = CustomProperty(
                        value=value_external_converted,
                        unit=unit_external,
                        symbol="P"
                    )
            else:
                call_args['P'] = CustomProperty(
                    value=pressure.value,
                    unit=pressure.unit,
                    symbol="P"
                )

        # NOTE: persist updated state for subsequent calculations
        self.state.update(xi_converted)

        # NOTE: Calculate rate
        # ! call the rate expression function as defined
        return self.eq(
            xi_converted,
            call_args,
            self.params
        )
