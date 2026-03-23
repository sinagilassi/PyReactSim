# import libs
# annotations
from __future__ import annotations
from pydantic import BaseModel, Field, PrivateAttr, model_validator
from typing import Dict, List, Literal, Optional, Callable
from pythermodb_settings.models import Component, CustomProperty, Pressure, Temperature, Volume, CustomProp, ComponentKey
from pyreactlab_core.models.reaction import Reaction
from .rate import ReactionRate
# locals

# NOTE: args
rArgs = Dict[str, CustomProperty]
# NOTE: parameters
rParams = Dict[str, CustomProperty]
# NOTE: return
rRet = Dict[str, CustomProperty]
# NOTE: state


class X(BaseModel):
    """
    Represents the state of a component in a reaction, including the component itself and its order in the reaction.

    - For a reaction A + 2B -> C, the state of component A would have an order of 1, while the state of component B would have an order of 2.

    Attributes
    ----------
    component: Component
        Component object containing information about the component (e.g., name, formula, state).
    order: float | int
        The order of the reaction with respect to this component. This indicates how the rate depends on the concentration or partial pressure of this component.
    value: float | int
        The value of the state variable for this component (e.g., concentration in mol/L or partial pressure in atm).
    unit: str
        The unit of the state variable (e.g., 'mol/L' for concentration or 'atm' for pressure).
    """
    component: Component = Field(
        ...,
        description="The component for which the reaction rate is being calculated."
    )
    order: float | int = Field(
        default=0,
        description="The order of the reaction with respect to this component."
    )
    value: float | int = Field(
        default=0,
        description="The value of the state variable for this component (e.g., concentration or partial pressure)."
    )
    unit: str = Field(
        default="",
        description="The unit of the state variable (e.g., 'mol/L' for concentration or 'atm' for pressure)."
    )


rXs = Dict[str, X]

# SECTION: Rate Expression


class ReactionRateExpression(BaseModel):
    basis: Literal['concentration', 'pressure'] = Field(
        ...,
        description="The basis for the reaction rate expression, either 'concentration' or 'pressure'."
    )
    components: List[Component] = Field(
        ...,
        description="The list of components involved in the reaction for which the rate expression is defined."
    )
    component_key: ComponentKey = Field(
        ...,
        description="The key to use for identifying components in the reaction rate expression. This should match the keys used in the state (Xi) and parameters (rParams) dictionaries."
    )
    reaction: Reaction = Field(
        ...,
        description="The reaction for which the rate expression is defined."
    )
    params: rParams = Field(
        default_factory=dict,
        description="A dictionary of parameters that may be used in the rate expression."
    )
    args: rArgs = Field(
        default_factory=dict,
        description="A dictionary of arguments that may be used in the rate expression, such as temperature and pressure."
    )
    returns: rRet = Field(
        default_factory=dict,
        description="A dictionary defining the expected return values from the rate expression calculation."
    )
    state: rXs = Field(
        default_factory=dict,
        description="Component state template keyed by component id. Use this to define reaction orders and default values."
    )
    eq: Callable[[rXs, rArgs, rParams], rRet] = Field(
        ...,
        description="A callable that takes the state (Xi), arguments (rArgs), parameters (rParams) and returns (rRet)."
    )
    _reaction_rate: ReactionRate = PrivateAttr()

    @model_validator(mode="after")
    def init(self):
        # init
        self._reaction_rate = ReactionRate(
            basis=self.basis,
            components=self.components,
            reaction=self.reaction,
            params=self.params,
            args=self.args,
            returns=self.returns,
            eq=self.eq,
            state=self.state,
            component_key=self.component_key
        )

        return self

    def calc(
            self,
            xi: Dict[str, CustomProperty],
            *,
            args: Optional[rArgs] = None,
            temperature: Optional[Temperature] = None,
            pressure: Optional[Pressure] = None
    ) -> rRet:
        """
        Update the reaction rate based on the provided state (xi) either concentration or pressure.
        """
        # calculate rate
        return self._reaction_rate.calc(
            xi=xi,
            args=args,
            temperature=temperature,
            pressure=pressure
        )
