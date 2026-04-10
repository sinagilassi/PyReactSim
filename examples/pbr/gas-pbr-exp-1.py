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
from pyreactsim import PBRReactor, create_pbr_reactor
from pyreactsim.models import HeatTransferOptions, PBRReactorOptions
from pyreactsim.thermo import build_thermo_source

# NOTE: example-specific imports
# ! create inline
# from examples.source.gas_model_source_exp_1 import model_source
# ! load from file
# from examples.source.gas_load_model_source import model_source
# ! add components & reaction rates
# from examples.rates.rate_exp_7 import components, reaction_rates, model_source
from examples.rates.methanol_1 import components, reaction_rates, model_source
# ! plot
from examples.plot.plot_res import plot_pbr_reactor_result

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
for logger_name in ("pyThermoDB", "pyThermoLinkDB", "pyThermoCalcDB", "pyreactlab_core"):
    logging.getLogger(logger_name).setLevel(logging.CRITICAL + 1)

# NOTE: debug logging for thermo source initialization timings
# logging.basicConfig(
#     level=logging.DEBUG,
#     format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
# )
# logging.getLogger("pyreactsim.sources.thermo_source").setLevel(logging.DEBUG)

# ====================================================
# SECTION: Inputs
# ====================================================
# ! Case: non-isothermal, constant-pressure, ideal gas (gas-phase PFR)

# NOTE: jacket temperature
jacket_temperature = Temperature(
    value=503,
    unit="K",
)

# NOTE: heat transfer coefficient
heat_transfer_coefficient = CustomProp(
    value=100.0,
    unit="W/m2.K",
)

# NOTE: heat transfer area
heat_transfer_area = CustomProp(
    value=2,
    unit="m2",
)

# NOTE: reactor options for thermo/source compatibility
pfr_reactor_options = PBRReactorOptions(
    modeling_type="scale",  # ! configure scale or physical modeling
    phase="gas",
    operation_mode="constant_pressure",
    pressure_mode="shortcut",
    gas_model="ideal",
    # gas_heat_capacity_mode="constant",
    # ideal_gas_formation_enthalpy_mode="model_inputs",
)

# NOTE: heat transfer options
heat_transfer_options = HeatTransferOptions(
    heat_transfer_mode="non-isothermal",
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
# NOTE: reactor volume / integration limit [m3]
reactor_volume = Volume(
    value=0.05,
    unit="m3",
)

# NOTE: bulk density for catalyst mass to volume conversion [kg/m3]
bulk_density = CustomProp(
    value=1200.0,
    unit="kg/m3",
)

# NOTE: pressure
pressure = CustomProp(
    value=50,
    unit="bar",
)

# NOTE: feed stream temperature [K]
inlet_temperature = Temperature(
    value=503,
    unit="K",
)

# NOTE: feed component molar flow rates [mol/s]
feed_mole_flow = {
    "CO2-g": CustomProp(value=2.5, unit="mol/s"),
    "H2-g": CustomProp(value=7.5, unit="mol/s"),
    "CH3OH-g": CustomProp(value=0.001, unit="mol/s"),
    "H2O-g": CustomProp(value=0.001, unit="mol/s"),
    "CO-g": CustomProp(value=0.001, unit="mol/s"),
}

# NOTE: model inputs for PFR
model_inputs = {
    "inlet_flows": feed_mole_flow,
    "reactor_volume": reactor_volume,
    "inlet_temperature": inlet_temperature,
    "inlet_pressure": pressure,
    "bulk_density": bulk_density,
}

# ====================================================
# SECTION: build thermo source
# ====================================================
thermo_source = build_thermo_source(
    components=components,
    model_source=model_source,
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
pfr_reactor: PBRReactor = create_pbr_reactor(
    model_inputs=model_inputs,
    thermo_source=thermo_source,
)
print("[bold green]PFR reactor successfully created![/bold green]")
print(pfr_reactor)

# NOTE: simulate PFR along reactor volume
simulation_results = pfr_reactor.simulate(
    solver_options={
        "method": "BDF",
        "volume_span": (0.0, reactor_volume.value),
        "rtol": 1e-5,
        "atol": [1e-10, 1e-10, 1e-10, 1e-10, 1e-10, 1e-6],
        "first_step": 1e-8,
        "max_step": 1e-3,
    }
)

# NOTE: simulate using diffeqpy
# simulation_results = pfr_reactor.simulate_diffeqpy(
#     solver_options={
#         "method": "Rodas5",
#         "volume_span": (0.0, reactor_volume.value),
#     }
# )


print("[bold green]PFR simulation completed![/bold green]")

# print(simulation_results)
if simulation_results is not None:
    plot_pbr_reactor_result(
        result=simulation_results,
        components=components,
    )
