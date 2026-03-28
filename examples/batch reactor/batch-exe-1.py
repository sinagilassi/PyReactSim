# import packages/modules
import logging
import warnings
from rich import print
from typing import Callable, Dict, Optional, Union, List, Any
import pyThermoDB as ptdb
import pyThermoLinkDB as ptdblink
from pythermodb_settings.models import Pressure, Temperature, CustomProp, Volume
from pyThermoDB import ComponentThermoDB
from pyThermoDB import build_component_thermodb_from_reference
from pyreactlab_core.models.reaction import Reaction
# locals
from pyreactsim.models.br import BatchReactorOptions, BatchReactorResult
from pyreactsim.docs.brs import batch_react
# NOTE: for example
# ! model sources
from model_source_exp_1 import model_source
# ! rate expressions & components
from rate_exp_5 import reaction_rates, components
# ! plot function
from examples.plot.plot_res import plot_batch_reactor_result

# check version
print(ptdb.__version__)
print(ptdblink.__version__)

# NOTE: silence library warnings/errors for this example run
warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)
for logger_name in ("pyThermoDB", "pyThermoLinkDB", "pyThermoCalcDB", "pyreactsim", "pyreactlab_core"):
    logging.getLogger(logger_name).setLevel(logging.CRITICAL + 1)

# ====================================================
# SECTION: Inputs
# ====================================================
# ! assumptions: variable pressure, isothermal, ideal gas behavior, single component system

# NOTE: reactor vessel volume in m3
reactor_volume = Volume(
    value=3.0,
    unit="m3",
)

# NOTE: Jacket temperature
jacket_temperature = Temperature(
    value=340,
    unit="K",
)

# NOTE: Heat transfer coefficient
heat_transfer_coefficient = CustomProp(
    value=100.0,
    unit="W/m2.K",
)

# NOTE: Heat transfer area
heat_transfer_area = CustomProp(
    value=2,
    unit="m2",
)

# ! reactor inputs
reactor_inputs = BatchReactorOptions(
    phase='gas',
    heat_transfer_mode='non-isothermal',
    operation_mode='constant_pressure',
    gas_model='ideal',
    reactor_volume=reactor_volume,
    jacket_temperature=jacket_temperature,
    heat_transfer_coefficient=heat_transfer_coefficient,
    heat_transfer_area=heat_transfer_area,
    heat_capacity_mode='temperature-dependent',
)

# ====================================================
# SECTION: model inputs
# ====================================================
# NOTE: initial temperature
initial_temperature = Temperature(
    value=340,
    unit="K",
)

# NOTE: initial pressure
initial_pressure = Pressure(
    value=5,
    unit="atm",
)

# NOTE: constant heat capacity (Cp) for the system in J/mol.K
# constant_heat_capacity = {
#     "CO2-g": CustomProp(value=30.0, unit="J/mol.K"),
#     "H2-g": CustomProp(value=25.0, unit="J/mol.K"),
#     "CH3OH-g": CustomProp(value=40.0, unit="J/mol.K"),
#     "H2O-g": CustomProp(value=35.0, unit="J/mol.K"),
# }

# ! model inputs
model_inputs = {
    "temperature": initial_temperature,
    "pressure": initial_pressure,
    # "heat_capacity": constant_heat_capacity,
}

# ====================================================
# SECTION: Simulation
# ====================================================
simulation_result: BatchReactorResult | None = batch_react(
    components=components,
    model_inputs=model_inputs,
    reactor_inputs=reactor_inputs,
    reaction_rates=reaction_rates,
    model_source=model_source,
    component_key='Name-Formula',
    solver_options={
        "method": "BDF",
        "time_span": (0, 200),
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
