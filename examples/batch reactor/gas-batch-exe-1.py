# import packages/modules
import logging
import warnings
from rich import print
import pyThermoDB as ptdb
import pyThermoLinkDB as ptdblink
from pythermodb_settings.models import Pressure, Temperature, CustomProp, Volume
# locals
from pyreactsim.models import BatchReactorOptions, HeatTransferOptions
from pyreactsim.thermo import build_thermo_source
from pyreactsim import create_batch_reactor, BatchReactor
# NOTE: for example
# ! model sources
# from examples.source.gas_model_source_exp_1 import model_source
# ! rate expressions & components
from examples.rates.rate_exp_1 import reaction_rates, components, model_source
# ! plot function
from examples.plot.plot_res import plot_batch_reactor_result


# check version
print(ptdb.__version__)
print(ptdblink.__version__)

# NOTE: silence library warnings/errors for this example run
# warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)
for logger_name in ("pyThermoDB", "pyThermoLinkDB", "pyThermoCalcDB", "pyreactlab_core"):
    logging.getLogger(logger_name).setLevel(logging.CRITICAL + 1)

# ====================================================
# SECTION: Inputs
# ====================================================
# ! assumptions: variable pressure, isothermal, ideal gas behavior, single component system

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

# ! batch reactor options
batch_reactor_options = BatchReactorOptions(
    modeling_type='scale',
    phase='gas',
    operation_mode='constant_volume',
    gas_model='ideal',
    # gas_heat_capacity_mode="constant",
    # ideal_gas_formation_enthalpy_mode="model_inputs",
)

# ! heat transfer options
heat_transfer_options = HeatTransferOptions(
    heat_transfer_mode='non-isothermal',
    heat_transfer_coefficient=heat_transfer_coefficient,
    heat_transfer_area=heat_transfer_area,
    jacket_temperature=jacket_temperature,
)

# ====================================================
# SECTION: thermo inputs
# ====================================================
# NOTE: optional constant gas heat capacities [J/mol.K]
constant_gas_heat_capacity = {
    "CO2-g": CustomProp(value=30.0, unit="J/mol.K"),
    "H2-g": CustomProp(value=25.0, unit="J/mol.K"),
    "CH3OH-g": CustomProp(value=40.0, unit="J/mol.K"),
    "H2O-g": CustomProp(value=35.0, unit="J/mol.K"),
    "CO-g": CustomProp(value=35.0, unit="J/mol.K"),
}

# NOTE: ideal gas formation enthalpy at 298 K [J/mol]
constant_ideal_gas_formation_enthalpy = {
    "CO2-g": CustomProp(value=-393520.0, unit="J/mol"),
    "H2-g": CustomProp(value=0.0, unit="J/mol"),
    "CH3OH-g": CustomProp(value=-201000.0, unit="J/mol"),
    "H2O-g": CustomProp(value=-241820.0, unit="J/mol"),
    "CO-g": CustomProp(value=-110530.0, unit="J/mol"),
}

# ! thermo inputs
thermo_inputs = {
    "gas_heat_capacity": constant_gas_heat_capacity,
    "ideal_gas_formation_enthalpy": constant_ideal_gas_formation_enthalpy,
}

# ====================================================
# SECTION: model inputs
# ====================================================
# NOTE: reactor vessel volume in m3
reactor_volume = Volume(
    value=3.0,
    unit="m3",
)

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

# NOTE: initial mole feed for the system in mol
initial_mole = {
    "CO2-g": CustomProp(value=1.0, unit="mol"),
    "H2-g": CustomProp(value=3.0, unit="mol"),
    "CH3OH-g": CustomProp(value=0.0, unit="mol"),
    "H2O-g": CustomProp(value=0.0, unit="mol"),
}

# NOTE: constant heat capacity (Cp) for the system in J/mol.K
constant_gas_heat_capacity = {
    "CO2-g": CustomProp(value=30.0, unit="J/mol.K"),
    "H2-g": CustomProp(value=25.0, unit="J/mol.K"),
    "CH3OH-g": CustomProp(value=40.0, unit="J/mol.K"),
    "H2O-g": CustomProp(value=35.0, unit="J/mol.K"),
}

# ! model inputs
model_inputs = {
    "mole": initial_mole,
    "temperature": initial_temperature,
    "pressure": initial_pressure,
    'reactor_volume': reactor_volume,
}

# ====================================================
# SECTION: build thermo source
# ====================================================
# NOTE: build thermo source
thermo_source = build_thermo_source(
    components=components,
    model_source=model_source,
    thermo_inputs=thermo_inputs,
    reactor_options=batch_reactor_options,
    heat_transfer_options=heat_transfer_options,
    reaction_rates=reaction_rates,
    component_key="Name-Formula",
)
print("[bold green]Thermo source successfully built![/bold green]")
print(thermo_source)

# ====================================================
# SECTION: create batch reactor
# ====================================================
batch_reactor: BatchReactor = create_batch_reactor(
    model_inputs=model_inputs,
    thermo_source=thermo_source,
    rhs_log_interval=10,
    rhs_log_enabled=True,
    rhs_log_timing_enabled=True,
)
print("[bold green]Batch reactor successfully created![/bold green]")
print(batch_reactor)


# NOTE: simulate batch reactor
simulation_results = batch_reactor.simulate(
    time_span=(0, 50),
    solver_options={
        "method": "BDF",
        "rtol": 1e-5,
        "atol": 1e-8,
        # "first_step": 1e-8,
        # "max_step": 1e-3,
    },
    mode='log'
)
print("[bold green]Batch reactor simulation completed![/bold green]")
# print(simulation_results)

if simulation_results is not None:
    plot_batch_reactor_result(
        result=simulation_results,
        components=components,
    )
