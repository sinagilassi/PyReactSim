# import packages/modules
from examples.plot.plot_res import plot_cstr_reactor_result
from examples.rates.rate_exp_1 import components, reaction_rates
from examples.source.gas_model_source_exp_1 import model_source
import logging
import sys
import warnings
from pathlib import Path
from rich import print
import pyThermoDB as ptdb
import pyThermoLinkDB as ptdblink
from pythermodb_settings.models import CustomProp, Temperature, Volume
# locals
from pyreactsim import CSTRReactor, create_cstr_reactor
from pyreactsim.models import BatchReactorOptions, CSTRReactorOptions, HeatTransferOptions
from pyreactsim.thermo import build_thermo_source

# NOTE: example source and kinetics
# ! add project root and examples root to import path for standalone script execution
PROJECT_DIR = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = Path(__file__).resolve().parents[1]
for path in (PROJECT_DIR, EXAMPLES_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# NOTE: CSTR plotting helper


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
# ! Case 6: non-isothermal, constant-volume, ideal gas (gas-phase CSTR)

# NOTE: jacket temperature
jacket_temperature = Temperature(
    value=340,
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
cstr_reactor_options = CSTRReactorOptions(
    phase="gas",
    operation_mode="constant_volume",
    holdup_volume_mode="fixed",
    outlet_flow_mode="calculated",
    gas_model="ideal",
    gas_heat_capacity_mode="temperature-dependent",
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
}

# ! thermo inputs
thermo_inputs = {
    "gas_heat_capacity": constant_gas_heat_capacity,
}

# ====================================================
# SECTION: model inputs
# ====================================================
# NOTE: fixed reactor holdup volume [m3]
reactor_volume = Volume(
    value=3.0,
    unit="m3",
)

# NOTE: initial reactor temperature [K]
initial_temperature = Temperature(
    value=340,
    unit="K",
)

# NOTE: feed stream temperature [K]
feed_temperature = Temperature(
    value=330,
    unit="K",
)

# NOTE: initial reactor holdup moles [mol]
initial_mole = {
    "CO2-g": CustomProp(value=1.0, unit="mol"),
    "H2-g": CustomProp(value=3.0, unit="mol"),
    "CH3OH-g": CustomProp(value=0.0, unit="mol"),
    "H2O-g": CustomProp(value=0.0, unit="mol"),
}

# NOTE: feed component molar flow rates [mol/s]
feed_mole_flow = {
    "CO2-g": CustomProp(value=0.05, unit="mol/s"),
    "H2-g": CustomProp(value=0.15, unit="mol/s"),
    "CH3OH-g": CustomProp(value=0.0, unit="mol/s"),
    "H2O-g": CustomProp(value=0.0, unit="mol/s"),
}

# NOTE: outlet total molar flow [mol/s]
outlet_mole_flow_total = CustomProp(
    value=0.20,
    unit="mol/s",
)


# NOTE: model inputs for CSTR
# ! constant volume
model_inputs = {
    "initial_mole": initial_mole,
    "inlet_flows": feed_mole_flow,
    "reactor_volume": reactor_volume,
    "initial_temperature": initial_temperature,
    "inlet_temperature": feed_temperature,
    "outlet_flow": outlet_mole_flow_total,
}

# ====================================================
# SECTION: build thermo source
# ====================================================
thermo_source = build_thermo_source(
    components=components,
    model_source=model_source,
    thermo_inputs=thermo_inputs,
    reactor_options=cstr_reactor_options,
    heat_transfer_options=heat_transfer_options,
    reaction_rates=reaction_rates,
    component_key="Name-Formula",
)
print("[bold green]Thermo source successfully built![/bold green]")
print(thermo_source)

# ====================================================
# SECTION: create cstr reactor
# ====================================================
cstr_reactor: CSTRReactor = create_cstr_reactor(
    model_inputs=model_inputs,
    thermo_source=thermo_source,
    cstr_reactor_options=cstr_reactor_options,
)
print("[bold green]CSTR reactor successfully created![/bold green]")
print(cstr_reactor)

# NOTE: simulate CSTR
simulation_results = cstr_reactor.simulate(
    solver_options={
        "method": "BDF",
        "time_span": (0, 200.0),
        "rtol": 1e-6,
        "atol": 1e-9,
    }
)
print("[bold green]CSTR simulation completed![/bold green]")
print(simulation_results)

# NOTE: plot CSTR results
if simulation_results is not None:
    plot_cstr_reactor_result(
        result=simulation_results,
        components=components,
    )
