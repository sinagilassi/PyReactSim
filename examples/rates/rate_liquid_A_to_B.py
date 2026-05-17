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
from pyreactsim.utils import arrhenius_equation

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

components: list[Component] = [A, B]

# NOTE: Reaction
reaction = Reaction(
    name="reaction 1",
    reaction="A(l) => B(l)",
    components=components
)


# SECTION: Reaction
# ! states
states: rXs = {
    'A-l': X(component=A),
    'B-l': X(component=B)
}

# ! rate expression parameters
rate_params: rParams = {
    'k1_ref': CustomProperty(value=1e-6, unit="m3/kg.s", symbol="k1_ref"),
    'Ea1': CustomProperty(value=120000, unit="kJ/mol", symbol="Ea1"),
    'T_ref': CustomProperty(value=340, unit="K", symbol="T_ref"),
    'rho_B': CustomProperty(value=1500, unit="kg/m3", symbol="rho_B"),
}

# ! rate expression arguments
rate_args: rArgs = {
    'T': CustomProperty(value=0.0, unit="K", symbol="T"),
}

# ! rate expression return values
rate_return: rRet = CustomProperty(value=0, unit="mol/m3.s", symbol="r1")


def r1(X: Dict[str, X], args: rArgs, params: rParams) -> CustomProperty:
    # rate constant k function of temperature and pressure

    # constant [1/s]
    k_ref = params['k1_ref'].value
    Ea = params['Ea1'].value
    T_ref = params['T_ref'].value

    # bulk density [kg/m3]
    rho_B = params['rho_B'].value

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

    # rate expression: r = k*C[A-l]
    # ? m3/kg.s * mol/m3 * kg/m3 = mol/m3.s
    r1 = k*C_A*rho_B

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
    basis='concentration',
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
