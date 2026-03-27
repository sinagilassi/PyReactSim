# import packages/modules
import logging
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
from pyreactsim.models.br import BatchReactorOptions, BatchReactorResult
from pyreactsim.docs.brs import batch_react
# ! model source
from model_source_exp_1 import model_source, components
from rate_exp_3 import rate_expression
from examples.plot.plot_res import plot_batch_reactor_result

# check version
print(ptdb.__version__)
print(ptdblink.__version__)

# NOTE: set logger
logger = logging.getLogger(__name__)
# turn off logging for pyThermoDB and pyThermoLinkDB
logging.getLogger("pyThermoDB").setLevel(logging.WARNING)
logging.getLogger("pyThermoLinkDB").setLevel(logging.WARNING)
logging.getLogger("pyThermoLinkDB").setLevel(logging.ERROR)
logging.getLogger("pythermocalcdb").setLevel(logging.WARNING)
logging.getLogger("pythermocalcdb").setLevel(logging.ERROR)
logging.getLogger("pyreactsim").setLevel(logging.INFO)

# ====================================================
# SECTION: Inputs
# ====================================================
# ! assumptions: variable pressure, isothermal, ideal gas behavior, single component system

# NOTE: reactor vessel volume in m3
reactor_volume = CustomProp(
    value=1.0,
    unit="m3",
)

# NOTE: Jacket temperature
jacket_temperature = CustomProp(
    value=450.0,
    unit="K",
)

# NOTE: Heat transfer coefficient
heat_transfer_coefficient = CustomProp(
    value=100.0,
    unit="W/m2.K",
)

# NOTE: Heat transfer area
heat_transfer_area = CustomProp(
    value=1,
    unit="m2",
)

# ! reactor inputs
reactor_inputs = BatchReactorOptions(
    phase='gas',
    heat_transfer_mode='isothermal',
    volume_mode='constant',
    gas_model='ideal',
    reactor_volume=reactor_volume,
    jacket_temperature=jacket_temperature,
    heat_transfer_coefficient=heat_transfer_coefficient,
    heat_transfer_area=heat_transfer_area,
    heat_capacity_mode='constant'
)

# ====================================================
# SECTION: model inputs
# ====================================================
# NOTE: initial temperature
initial_temperature = Temperature(
    value=523,
    unit="K",
)

# NOTE: initial pressure
initial_pressure = Pressure(
    value=5e6,
    unit="Pa",
)

# ! model inputs
model_inputs = {
    "temperature": initial_temperature,
    "pressure": initial_pressure,
}

# ====================================================
# SECTION: Simulation
# ====================================================
simulation_result: BatchReactorResult | None = batch_react(
    components=components,
    model_inputs=model_inputs,
    reactor_inputs=reactor_inputs,
    reaction_rates={"r1": rate_expression},
    model_source=model_source,
    component_key='Name-Formula',
    solver_options={
        "method": "BDF",
        "time_span": (0, 120),
        "rtol": 1e-6,
        "atol": 1e-9
    }
)
print(simulation_result)

if simulation_result is not None:
    plot_batch_reactor_result(
        result=simulation_result,
        components=components,
    )
