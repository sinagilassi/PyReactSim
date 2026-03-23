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
# locals
from pyreactsim.models.br import BatchReactorOptions
from pyreactsim.models.reaction_exp import rArgs, rParams, rRet, X, rXs, ReactionRateExpression


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
    state="g"
)

B = Component(
    name="B",
    formula="B",
    state="g"
)

C = Component(
    name="C",
    formula="C",
    state="g"
)

components: list[Component] = [A, B, C]

# NOTE: Reaction
reaction = Reaction(
    name="A_to_B",
    reaction="A + B => C",
    components=components
)


# NOTE: reactor vessel volume in m3
reactor_volume = CustomProp(
    value=1.0,
    unit="m3",
)

# NOTE: Jacket temperature
jacket_temperature = CustomProp(
    value=350.0,
    unit="K",
)

# NOTE: Heat transfer coefficient
heat_transfer_coefficient = CustomProp(
    value=500.0,
    unit="W/m2.K",
)

# NOTE: Heat transfer area
heat_transfer_area = CustomProp(
    value=5.0,
    unit="m2",
)

# ! reactor inputs
reactor_inputs = BatchReactorOptions(
    phase='gas',
    heat_transfer_mode='non-isothermal',
    volume_mode='constant',
    jacket_temperature=jacket_temperature,
    heat_transfer_coefficient=heat_transfer_coefficient,
    heat_transfer_area=heat_transfer_area
)

# NOTE: initial temperature
initial_temperature = Temperature(
    value=300.0,
    unit="K",
)

# NOTE: initial pressure
initial_pressure = Pressure(
    value=101325.0,
    unit="Pa",
)

# ! model inputs
model_inputs = {
    "temperature": initial_temperature,
    "pressure": initial_pressure,
}

# SECTION: Reaction
# ! Reaction Rate Expression
states: rXs = {
    'A': X(component=A, order=1),
    'B': X(component=B, order=1)
}

rate_params: rParams = {
    'k0': CustomProperty(value=0.1, unit="1/s", symbol="k0"),
    'Ea': CustomProperty(value=50000.0, unit="J/mol", symbol="Ea"),
    'R': CustomProperty(value=8.314, unit="J/mol.K", symbol="R"),
    "A": CustomProperty(value=0, unit="mol/m3", symbol="A"),
}

rate_return: rRet = {
    'r1': CustomProperty(value=0, unit="mol/m3.s", symbol="r1")
}

rate_args: rArgs = {
    'T': CustomProperty(value=0, unit="K", symbol="T")
}


def r1(X: Dict[str, X], args: rArgs, params: rParams) -> rRet:
    # rate constant k function of temperature and pressure
    k0 = params['k0'].value
    Ea = params['Ea'].value
    k = k0*math.exp(-Ea/(params['R'].value*args['T'].value))

    # rate expression: r = k*[A]^order_A*[B]^order_B
    rExp = k*(X['A'].value**X['A'].order)*(X['B'].value**X['B'].order)

    return {
        'r1': CustomProperty(value=rExp, unit="mol/m3.s", symbol="r1")
    }


# execute
rate_expression = ReactionRateExpression(
    basis='concentration',
    components=[A, B],
    reaction=reaction,
    params=rate_params,
    args=rate_args,
    returns=rate_return,
    state=states,
    eq=r1,
    component_key='Name-Formula'
)

# cal
result = rate_expression.calc(
    xi={
        'A': CustomProperty(value=2.0, unit="mol/m3", symbol="A"),
        'B': CustomProperty(value=3.0, unit="mol/m3", symbol="B")
    },
    temperature=initial_temperature,
    pressure=initial_pressure
)
print(result)
