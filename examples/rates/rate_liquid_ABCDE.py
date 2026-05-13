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

E = Component(
    name="E",
    formula="E",
    state="l",
)

components: list[Component] = [A, B, C, D, E]

# NOTE: Reaction
reaction_1 = Reaction(
    name="reaction 1",
    reaction="A(l) + B(l) => C(l)",
    components=components
)

reaction_2 = Reaction(
    name="reaction 2",
    reaction="A(l) + 2C(l) => D(l) + E(l)",
    components=components
)

# SECTION: Reaction
# ! states
states: rXs = {
    'A-l': X(component=A),
    'B-l': X(component=B),
    'C-l': X(component=C),
    'D-l': X(component=D),
    'E-l': X(component=E),
}

# ! rate expression parameters
rate_params: rParams = {
    'k1_ref': CustomProperty(value=5e-5, unit="m3/mol.s", symbol="k1_ref"),
    'Ea1': CustomProperty(value=42500.0, unit="J/mol", symbol="Ea1"),
    'k2_ref': CustomProperty(value=2e-6, unit="m6/mol2.s", symbol="k2_ref"),
    'Ea2': CustomProperty(value=44500.0, unit="J/mol", symbol="Ea2"),
    'T_ref': CustomProperty(value=323, unit="K", symbol="T_ref"),
}

# ! rate expression arguments
rate_args: rArgs = {
    'T': CustomProperty(value=0.0, unit="K", symbol="T"),
}

# ! rate expression return values
rate_1_return: rRet = CustomProperty(value=0, unit="mol/m3.s", symbol="r1")
rate_2_return: rRet = CustomProperty(value=0, unit="mol/m3.s", symbol="r2")


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
    k = arrhenius_equation(
        k_ref=k_ref,
        Ea=Ea,
        T=T,
        T_ref=T_ref
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
    name="r1",
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

# NOTE: reaction 2


def r2(X: Dict[str, X], args: rArgs, params: rParams) -> CustomProperty:
    # rate constant k function of temperature and pressure
    # constant [1/s]
    k_ref = params['k2_ref'].value
    Ea = params['Ea2'].value
    T_ref = params['T_ref'].value

    # args
    T = args['T'].value

    # calculate rate constant k using Arrhenius equation
    k2 = arrhenius_equation(
        k_ref=k_ref,
        Ea=Ea,
        T=T,
        T_ref=T_ref
    )

    C_A = X['A-l'].value
    C_C = X['C-l'].value

    # rate expression: r = k*C[A-l]*C[C-l]^2
    r2 = k2*C_A*C_C**2

    return CustomProperty(
        name="r2",
        description="Reaction rate for reaction 2",
        value=r2,
        unit="mol/m3.s",
        symbol="r2"
    )


rate_expression_2 = ReactionRateExpression(
    name="r2",
    basis='concentration',
    components=components,
    reaction=reaction_2,
    params=rate_params,
    args=rate_args,
    ret=rate_2_return,
    state=states,
    state_key='Formula-State',
    eq=r2,
    component_key='Name-Formula'
)


# SECTION: reaction rates
reaction_rates: List[ReactionRateExpression] = [
    rate_expression_1,
    rate_expression_2,
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
