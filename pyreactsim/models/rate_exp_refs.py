# import libs
# annotations
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Dict
from pythermodb_settings.models import Component, CustomProperty
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
