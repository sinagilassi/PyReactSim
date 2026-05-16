# import packages/modules
import logging
import sys
import warnings
from pathlib import Path

from rich import print

import pyThermoDB as ptdb
import pyThermoLinkDB as ptdblink
from pythermodb_settings.models import CustomProp, Temperature, Volume

# locals
from pyreactsim import PFRReactor, create_pfr_reactor
from pyreactsim.models import HeatTransferOptions, PFRReactorOptions
from pyreactsim.thermo import build_thermo_source

# NOTE: example-specific imports
# ! rate expressions & components
from examples.rates.rate_A_B_to_C_D import reaction_rates, components
from examples.plot.plot_res import plot_pfr_reactor_result

# NOTE: example source and kinetics
# ! add project root and examples root to import path for standalone script execution
PROJECT_DIR = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = Path(__file__).resolve().parents[1]
for path in (PROJECT_DIR, EXAMPLES_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# check version
print(ptdb.__version__)
print(ptdblink.__version__)

# NOTE: silence library warnings/errors for this example run
warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)
for logger_name in ("pyThermoDB", "pyThermoLinkDB", "pythermocalcdb", "pyreactsim", "pyreactlab_core"):
    logging.getLogger(logger_name).setLevel(logging.CRITICAL + 1)

# ====================================================
# SECTION: Inputs
# ====================================================
# ! Case: non-isothermal, constant-pressure, ideal gas (gas-phase PFR)

# NOTE: jacket temperature
jacket_temperature = Temperature(
    value=330,
    unit="K",
)

# NOTE: heat transfer coefficient
heat_transfer_coefficient = CustomProp(
    value=100.0,
    unit="W/m2.K",
)

# NOTE: heat transfer area
heat_transfer_area = CustomProp(
    value=2.0,
    unit="m2",
)

# NOTE: reactor options for thermo/source compatibility
pfr_reactor_options = PFRReactorOptions(
    modeling_type="scale",
    operation_mode="constant_pressure",
    phase="gas",
    gas_model="ideal",
    # mode
    pressure_mode="shortcut",
    # source
)

# NOTE: heat transfer options
heat_transfer_options = HeatTransferOptions(
    heat_transfer_mode="isothermal",
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
    # "gas_heat_capacity": constant_gas_heat_capacity,
    # "ideal_gas_formation_enthalpy": constant_ideal_gas_formation_enthalpy,
}

# ====================================================
# SECTION: model inputs
# ====================================================
# NOTE: reactor volume / integration limit [m3]
reactor_volume = Volume(
    value=1.0,
    unit="m3",
)

# NOTE: pressure
pressure = CustomProp(
    value=8,
    unit="atm",
)

# NOTE: feed stream temperature [K]
inlet_temperature = Temperature(
    value=340,
    unit="K",
)

# NOTE: feed component molar flow rates [mol/s]
feed_mole_flow = {
    "A-g": CustomProp(value=1, unit="mol/s"),
    "B-g": CustomProp(value=1.2, unit="mol/s"),
    "C-g": CustomProp(value=0.0, unit="mol/s"),
    "D-g": CustomProp(value=0.0, unit="mol/s"),
}

# NOTE: model inputs for PFR
model_inputs = {
    "inlet_flows": feed_mole_flow,
    "reactor_volume": reactor_volume,
    "inlet_temperature": inlet_temperature,
    "inlet_pressure": pressure,
}

# ====================================================
# SECTION: build thermo source
# ====================================================
thermo_source = build_thermo_source(
    components=components,
    model_source=None,
    thermo_inputs=thermo_inputs,
    reactor_options=pfr_reactor_options,
    heat_transfer_options=heat_transfer_options,
    reaction_rates=reaction_rates,
    component_key="Name-Formula",
)
print("[bold green]Thermo source successfully built![/bold green]")
print(thermo_source)

# ====================================================
# SECTION: create pfr reactor
# ====================================================
pfr_reactor: PFRReactor = create_pfr_reactor(
    model_inputs=model_inputs,
    thermo_source=thermo_source,
)
print("[bold green]PFR reactor successfully created![/bold green]")
print(pfr_reactor)

# NOTE: simulate PFR along reactor volume
simulation_results = pfr_reactor.simulate(
    volume_span=(0, reactor_volume.value),
    solver_options={
        "method": "Radau",
        "rtol": 1e-6,
        "atol": 1e-9,
        # max step size as fraction of total volume
        # "max_step": reactor_volume.value / 100,
    },
    mode="log"
)
print("[bold green]PFR simulation completed![/bold green]")
# print(simulation_results)

if simulation_results is not None:
    plot_pfr_reactor_result(
        result=simulation_results,
        components=components,
    )
