# import packages/modules
import math
from rich import print
from typing import Callable, Dict, Optional, Union, List, Any
import pyThermoDB as ptdb
import pyThermoLinkDB as ptdblink
from pyThermoLinkDB import (
    build_component_model_source,
    build_components_model_source,
    build_model_source
)
from pyThermoLinkDB.models import ComponentModelSource, ModelSource
from pythermodb_settings.models import Component, Pressure, Temperature, CustomProp, Volume, CustomProperty
from pyThermoDB import ComponentThermoDB
from pyThermoDB import build_component_thermodb_from_reference
from pyreactlab_core.models.reaction import Reaction
from pyreactsim_core.models import (
    rArgs,
    rParams,
    rRet,
    X,
    rXs,
    ReactionRateExpression
)
# ! locals

# check version
print(ptdb.__version__)
print(ptdblink.__version__)


# =======================================
# SECTION: Inputs
# =======================================
# ! assumptions: variable pressure, isothermal, ideal gas behavior, single component system
# NOTE: Components
A = Component(
    name="A",
    formula="A",
    state="g",
)

B = Component(
    name="B",
    formula="B",
    state="g",
)

C = Component(
    name="C",
    formula="C",
    state="g",
)

D = Component(
    name="D",
    formula="D",
    state="g",
)

components: list[Component] = [A, B, C, D]

# NOTE: Reaction
reaction = Reaction(
    name="reaction 1",
    reaction="A(g) + B(g) <=> C(g) + 2D(g)",
    components=components
)


# SECTION: Reaction
# ! states
states: rXs = {
    'A-g': X(component=A, unit="atm"),
    'B-g': X(component=B, unit="atm"),
    'C-g': X(component=C, unit="atm"),
    'D-g': X(component=D, unit="atm"),
}

# ! rate expression parameters
rate_params: rParams = {
    'k': CustomProperty(value=0.5, unit="mol/m3.s.atm2", symbol="k"),
}

# ! rate expression arguments
rate_args: rArgs = {}

# ! rate expression return values
rate_return: rRet = CustomProperty(value=0, unit="mol/m3.s", symbol="r1")


def r1(X: Dict[str, X], args: rArgs, params: rParams) -> CustomProperty:
    # rate constant k function of temperature and pressure
    # constant [1/s]
    k = params['k'].value

    # pressure of each component in atm
    P_A = X['A-g'].value
    P_B = X['B-g'].value

    # rate expression: r = k*C[A-g]*C[B-g]
    r1 = k*P_A*P_B

    return CustomProperty(
        name="r1",
        description="Reaction rate for reaction 1",
        value=r1,
        unit="mol/m3.s",
        symbol="r1"
    )


# execute
rate_expression = ReactionRateExpression(
    name="r1",
    basis='pressure',
    components=components,
    reaction=reaction,
    params=rate_params,
    args=rate_args,
    ret=rate_return,
    state=states,
    state_key='Formula-State',
    eq=r1,
    component_key='Name-Formula'
)

reaction_rates: List[ReactionRateExpression] = [rate_expression]

# SECTION: rate expression evaluation

# # NOTE: initial temperature
# initial_temperature = Temperature(
#     value=298.0,
#     unit="K",
# )

# # NOTE: initial pressure
# initial_pressure = Pressure(
#     value=1,
#     unit="atm",
# )

# # NOTE: calculate rate expression value
# result = rate_expression.calc(
#     xi={
#         'A-g': CustomProperty(value=1.0, unit="Pa", symbol="A-g"),
#         'B-g': CustomProperty(value=2.0, unit="Pa", symbol="B-g"),
#     },
#     args={
#         'k': CustomProperty(value=0.5, unit="mol/m3.s.atm2", symbol="k"),
#     },
#     temperature=initial_temperature,
#     pressure=initial_pressure,
#     mode="log"
# )
# print(result)
