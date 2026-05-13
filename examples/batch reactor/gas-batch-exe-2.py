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
from pyreactsim.app import evaluate_batch_reactor
# NOTE: for example
# ! model sources
# from examples.source.gas_model_source_exp_1 import model_source
# ! rate expressions & components
from examples.rates.rate_A_B_to_C_D import reaction_rates, components
# ! plot function
from examples.plot.plot_res import plot_batch_reactor_result
from examples.plot.plot_xy import plot_xy


# check version
print(ptdb.__version__)
print(ptdblink.__version__)

# NOTE: silence library warnings/errors for this example run
# warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)
for logger_name in ("pyThermoDB", "pyThermoLinkDB", "pythermocalcdb", "pyreactlab_core"):
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
    operation_mode='constant_volume',
    phase='gas',
    gas_model='ideal',
    # mode
    gas_heat_capacity_mode="constant",
    # source
    gas_heat_capacity_source="model_inputs",
    ideal_gas_formation_enthalpy_source="model_inputs",
)

# ! heat transfer options
heat_transfer_options = HeatTransferOptions(
    heat_transfer_mode='isothermal',
    heat_transfer_coefficient=None,
    heat_transfer_area=None,
    jacket_temperature=None,
)

# ====================================================
# SECTION: thermo inputs
# ====================================================
# NOTE: optional constant gas heat capacities [J/mol.K]
constant_gas_heat_capacity = {
    "A-g": CustomProp(value=30.0, unit="J/mol.K"),
    "B-g": CustomProp(value=25.0, unit="J/mol.K"),
    "C-g": CustomProp(value=40.0, unit="J/mol.K"),
    "D-g": CustomProp(value=35.0, unit="J/mol.K"),
}

# NOTE: ideal gas formation enthalpy at 298 K [J/mol]
constant_ideal_gas_formation_enthalpy = {
    "A-g": CustomProp(value=-393520.0, unit="J/mol"),
    "B-g": CustomProp(value=-100000.0, unit="J/mol"),
    "C-g": CustomProp(value=-200000.0, unit="J/mol"),
    "D-g": CustomProp(value=-150000.0, unit="J/mol"),
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
    "A-g": CustomProp(value=100.0, unit="mol"),
    "B-g": CustomProp(value=120.0, unit="mol"),
    "C-g": CustomProp(value=0.0, unit="mol"),
    "D-g": CustomProp(value=0.0, unit="mol"),
}

# ! model inputs
model_inputs = {
    "mole": initial_mole,
    "temperature": initial_temperature,
    'reactor_volume': reactor_volume,
}

# ====================================================
# SECTION: build thermo source
# ====================================================
# NOTE: build thermo source
thermo_source = build_thermo_source(
    components=components,
    model_source=None,
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
    # rhs_log_interval=10,
    # rhs_log_enabled=True,
    # rhs_log_timing_enabled=True,
)
print("[bold green]Batch reactor successfully created![/bold green]")
print(batch_reactor)


# NOTE: simulate batch reactor
simulation_results = batch_reactor.simulate(
    time_span=(0, 800),
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
    evaluated_results = evaluate_batch_reactor(
        batch_reactor=batch_reactor,
        simulation_results=simulation_results,
    )

    plot_xy(
        x=evaluated_results["time"],
        y=evaluated_results["pressure_total"],
        legends=["total pressure"],
        xlabel="time (s)",
        ylabel="total pressure (Pa)",
        title="Gas Batch Reactor Total Pressure vs Time",
    )

    plot_batch_reactor_result(
        result=simulation_results,
        components=components,
    )
