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
from pyreactsim.utils import arrhenius_equation
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
    state="l",
)

B = Component(
    name="B",
    formula="B",
    state="l",
)

C = Component(
    name="C",
    formula="C",
    state="l",
)

D = Component(
    name="D",
    formula="D",
    state="l",
)


components: list[Component] = [A, B, C, D]

# NOTE: Reaction
reaction_1 = Reaction(
    name="r1",
    reaction="A(l) + B(l) => C(l) + D(l)",
    components=components
)

# SECTION: Reaction
# ! states
states: rXs = {
    'A-l': X(component=A),
    'B-l': X(component=B),
    'C-l': X(component=C),
    'D-l': X(component=D),
}

# ! rate expression parameters
rate_params: rParams = {
    'k1_ref': CustomProperty(value=4000, unit="m3/mol.s", symbol="k1_ref"),
    'Ea1': CustomProperty(value=25000, unit="J/mol", symbol="Ea1"),
    'T_ref': CustomProperty(value=323, unit="K", symbol="T_ref"),
}

# ! rate expression arguments
rate_args: rArgs = {
    'T': CustomProperty(value=0.0, unit="K", symbol="T"),
}

# ! rate expression return values
rate_1_return: rRet = CustomProperty(value=0, unit="mol/m3.s", symbol="r1")


# NOTE: reaction 1
def r1(X: Dict[str, X], args: rArgs, params: rParams) -> CustomProperty:
    # rate constant k function of temperature and pressure
    # constant [1/s]
    k_ref = params['k1_ref'].value
    Ea = params['Ea1'].value
    T_ref = params['T_ref'].value

    # args
    T = args['T'].value

    # calculate rate constant k using Arrhenius equation
    # ? m3/mol.s for bimolecular reaction in liquid phase
    k = arrhenius_equation(
        k_ref=k_ref,
        Ea=Ea,
        T=T,
        T_ref=T_ref,
    )

    # concentration of reactants in mol/m3
    C_A = X['A-l'].value
    C_B = X['B-l'].value

    # rate expression: r = k*C[A-l]*C[B-l]
    r1 = k*C_A*C_B

    return CustomProperty(
        name="r1",
        description="Reaction rate for reaction 1",
        value=r1,
        unit="mol/m3.s",
        symbol="r1"
    )


# execute
rate_expression_1 = ReactionRateExpression(
    name="reaction rate 1",
    basis='concentration',
    components=components,
    reaction=reaction_1,
    params=rate_params,
    args=rate_args,
    ret=rate_1_return,
    state=states,
    state_key='Formula-State',
    eq=r1,
    component_key='Name-Formula'
)


# SECTION: reaction rates
reaction_rates: List[ReactionRateExpression] = [
    rate_expression_1,
]

# # SECTION: rate expression evaluation

# # NOTE: initial temperature
# initial_temperature = Temperature(
#     value=298.0,
#     unit="K",
# )

# # NOTE: initial pressure
# initial_pressure = Pressure(
#     value=101325.0,
#     unit="Pa",
# )

# # NOTE: calculate rate expression value
# result = rate_expression.calc(
#     xi={
#         'A-l': CustomProperty(value=1.0, unit="mol/m3", symbol="A-l"),
#     },
#     args={
#         'k': CustomProperty(value=5e-2, unit="1/s", symbol="k"),
#     },
#     temperature=initial_temperature,
#     pressure=initial_pressure,
#     mode="log"
# )
# print(result)
