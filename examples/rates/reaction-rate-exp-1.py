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
from pyreactsim.models import rArgs, rParams, rRet, X, rXs, ReactionRateExpression

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
    X={
        "name": "mole",
        "value": 100,
        "unit": "mol",
        "symbol": "n"
    }
)

B = Component(
    name="B",
    formula="B",
    state="g",
    X={
        "name": "mole",
        "value": 120,
        "unit": "mol",
        "symbol": "n"
    }
)

C = Component(
    name="C",
    formula="C",
    state="g",
    X={
        "name": "mole",
        "value": 0.0,
        "unit": "mol",
        "symbol": "n"
    }
)

D = Component(
    name="D",
    formula="D",
    state="g",
    X={
        "name": "mole",
        "value": 0.0,
        "unit": "mol",
        "symbol": "n"
    }
)

components: list[Component] = [A, B, C, D]

# NOTE: Reaction
reaction = Reaction(
    name="reaction 1",
    reaction="A(g) + B(g) => C(g) + 2D(g)",
    components=components
)


# NOTE: reactor vessel volume in m3
reactor_volume = Volume(
    value=1.0,
    unit="m3",
)

# NOTE: Jacket temperature
jacket_temperature = Temperature(
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

# NOTE: gas model
gas_model = "ideal"

# ! reactor inputs
reactor_inputs = BatchReactorOptions(
    phase='gas',
    gas_model=gas_model,
    operation_mode='constant_volume',
)

# NOTE: initial temperature
initial_temperature = Temperature(
    value=298.0,
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
states: rXs = {
    'A-g': X(component=A, order=1),
    'B-g': X(component=B, order=1)
}

rate_params: rParams = {
    'k0': CustomProperty(value=1.05e7, unit="1/s", symbol="k0"),
    'Ea': CustomProperty(value=46600.0, unit="J/mol", symbol="Ea"),
    'R': CustomProperty(value=8.314, unit="J/mol.K", symbol="R"),
}

rate_return: rRet = CustomProperty(value=0, unit="mol/m3.s", symbol="r1")

rate_args: rArgs = {
    'T': CustomProperty(value=0, unit="K", symbol="T")
}


def r1(X: Dict[str, X], args: rArgs, params: rParams) -> CustomProperty:
    # rate constant k function of temperature and pressure
    k0 = params['k0'].value
    Ea = params['Ea'].value
    k = k0*math.exp(-Ea/(params['R'].value*args['T'].value))

    # rate expression: r = k*[A]^order_A*[B]^order_B
    rExp = k*(X['A-g'].value**X['A-g'].order)*(X['B-g'].value**X['B-g'].order)

    return CustomProperty(
        name="r1",
        description="Reaction rate for reaction 1",
        value=rExp,
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

# calculate rate expression value
result = rate_expression.calc(
    xi={
        'A-g': CustomProperty(value=1.0, unit="mol/m3", symbol="A-g"),
        'B-g': CustomProperty(value=1.0, unit="mol/m3", symbol="B-g")
    },
    args={
        'T': CustomProperty(
            value=300,
            unit='K',
            symbol='T'
        )
    },
    temperature=initial_temperature,
    pressure=initial_pressure,
    mode="log"
)
print(result)
