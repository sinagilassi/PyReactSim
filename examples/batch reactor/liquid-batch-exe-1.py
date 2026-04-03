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

# NOTE: Batch reactor settings
# ! model sources for liquid phase batch reactor
from examples.source.liquid_model_source_exp_1 import model_source
# ! rate expressions & components
from examples.rates.rate_exp_6 import reaction_rates, components
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

# NOTE: Jacket temperature
jacket_temperature = Temperature(
    value=350,
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
    phase='liquid',
    operation_mode='variable_volume',
    gas_model='ideal',
    gas_heat_capacity_mode='temperature-dependent',
    liquid_heat_capacity_mode='temperature-dependent',
    liquid_density_mode='constant',
)

# ! heat transfer options
heat_transfer_options = HeatTransferOptions(
    heat_transfer_mode='non-isothermal',
    heat_transfer_coefficient=None,
    heat_transfer_area=None,
    jacket_temperature=None,
)

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
    value=1,
    unit="bar",
)

# NOTE: initial mole feed for the system in mol
initial_mole = {
    "CH3COOH-l": CustomProp(value=100.0, unit="mol"),  # acetic acid
    "CH3OH-l": CustomProp(value=100.0, unit="mol"),  # methanol
    "C3H6O2-l": CustomProp(value=0.0, unit="mol"),  # methyl acetate
    "H2O-l": CustomProp(value=0.0, unit="mol"),  # water
}

# NOTE: constant heat capacity (Cp) for the system in J/mol.K
constant_gas_heat_capacity = {}

# NOTE: constant liquid density (rho_LIQ) for the system in kg/m3
constant_liquid_density = {
    "CH3OH-l": CustomProp(value=786.6, unit="kg/m3"),  # methanol
    "H2O-l": CustomProp(value=997.0, unit="kg/m3"),   # water
    "CH3COOH-l": CustomProp(value=1049.0, unit="kg/m3"),  # acetic acid
    "C3H6O2-l": CustomProp(value=932.0, unit="kg/m3"),  # methyl acetate
}

# ! thermo inputs
thermo_inputs = {
    "liquid_density": constant_liquid_density,
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
    components=components,
    model_inputs=model_inputs,
    thermo_source=thermo_source,
)
print("[bold green]Batch reactor successfully created![/bold green]")
print(batch_reactor)


# NOTE: simulate batch reactor
simulation_results = batch_reactor.simulate()
print("[bold green]Batch reactor simulation completed![/bold green]")
print(simulation_results)

if simulation_results is not None:
    plot_batch_reactor_result(
        result=simulation_results,
        components=components,
    )
