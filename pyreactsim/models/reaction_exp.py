# import libs
# annotations
from __future__ import annotations
from pydantic import BaseModel, Field, computed_field, model_validator
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, TypeAlias, Callable, Awaitable
from pythermodb_settings.models import Component, CustomProperty, Pressure, Temperature, Volume, CustomProp
from pyreactlab_core.models.reaction import Reaction
from ..core.rate import ReactionRate
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


Xi = Dict[str, X]

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
    reaction: Reaction = Field(
        ...,
        description="The reaction for which the rate expression is defined."
    )
    parameters: rParams = Field(
        default_factory=dict,
        description="A dictionary of parameters that may be used in the rate expression."
    )
    arguments: rArgs = Field(
        default_factory=dict,
        description="A dictionary of arguments that may be used in the rate expression, such as temperature and pressure."
    )
    returns: rRet = Field(
        default_factory=dict,
        description="A dictionary of return values from the rate expression, typically including the calculated reaction rate."
    )
    eq: Callable[[Xi, rParams, rArgs], rRet] = Field(
        ...,
        description="A callable that takes the state (Xi), parameters (rParams), and arguments (rArgs) and return (rRet)."
    )

    @model_validator(mode="after")
    def init(self):
        # init
        self.reactionRate = ReactionRate(
            basis=self.basis,
            components=self.components,
            reaction=self.reaction,
            parameters=self.parameters,
            arguments=self.arguments,
            returns=self.returns,
            eq=self.eq
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
        return self.reactionRate.calc(
            xi=xi,
            args=args,
            temperature=temperature,
            pressure=pressure
        )
