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
from pyreactsim.models import PFRReactorOptions, HeatTransferOptions
from pyreactsim.thermo import build_thermo_source
# NOTE: example-specific imports
# from examples.source.liquid_model_source_exp_1 import model_source
# ! rate expressions & components
from examples.rates.rate_liquid_ABCDE_2 import reaction_rates, components
from examples.plot.plot_res import plot_pfr_reactor_result

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
# warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)
for logger_name in ("pyThermoDB", "pyThermoLinkDB", "pythermocalcdb", "pyreactlab_core"):
    logging.getLogger(logger_name).setLevel(logging.CRITICAL + 1)

# ====================================================
# SECTION: Inputs
# ====================================================
# ! Case 6: non-isothermal, constant-volume, ideal gas (gas-phase CSTR)

# NOTE: jacket temperature
jacket_temperature = Temperature(
    value=300,
    unit="K",
)

# NOTE: heat transfer coefficient
heat_transfer_coefficient = CustomProp(
    value=1200,
    unit="W/m2.K",
)

# NOTE: heat transfer area
heat_transfer_area = CustomProp(
    value=2.0,
    unit="m2",
)

# NOTE: reactor options for thermo/source compatibility
cstr_reactor_options = PFRReactorOptions(
    modeling_type="scale",
    phase="liquid",
    operation_mode="constant_volume",
    # radius
    reactor_radius=CustomProp(value=0.03, unit="m"),
    # mode
    # gas_heat_capacity_mode="temperature-dependent",
    # liquid_heat_capacity_mode='temperature-dependent',
    use_liquid_mixture_volumetric_heat_capacity=True,
    reaction_enthalpy_mode='reaction',
    # source
    liquid_density_mode='constant',
    # source
    liquid_density_source="model_inputs",
    molecular_weight_source="model_inputs",
    liquid_mixture_volumetric_heat_capacity_source="model_inputs",
    reaction_enthalpy_source='model_inputs',
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
# NOTE: molecular weight (MW) for the components in g/mol
molecular_weight = {
    "A-l": CustomProp(value=0.028, unit="kg/mol"),
    "B-l": CustomProp(value=0.018, unit="kg/mol"),
    "C-l": CustomProp(value=0.046, unit="kg/mol"),
    "D-l": CustomProp(value=0.032, unit="kg/mol"),
}

# NOTE: optional constant gas heat capacities [J/mol.K]
constant_gas_heat_capacity = {
    "A-l": CustomProp(value=75.3, unit="J/mol.K"),
    "B-l": CustomProp(value=75.3, unit="J/mol.K"),
    "C-l": CustomProp(value=75.3, unit="J/mol.K"),
    "D-l": CustomProp(value=75.3, unit="J/mol.K"),
}

# NOTE: optional constant liquid heat capacities [J/mol.K]
constant_liquid_heat_capacity = {
    "A-l": CustomProp(value=81.1, unit="J/mol.K"),
    "B-l": CustomProp(value=75.3, unit="J/mol.K"),
    "C-l": CustomProp(value=120.5, unit="J/mol.K"),
    "D-l": CustomProp(value=90.2, unit="J/mol.K"),
}

# NOTE: constant liquid density (rho_LIQ) for the system in kg/m3
constant_liquid_density = {
    "A-l": CustomProp(value=570, unit="kg/m3"),
    "B-l": CustomProp(value=1000, unit="kg/m3"),
    "C-l": CustomProp(value=789, unit="kg/m3"),
    "D-l": CustomProp(value=850, unit="kg/m3"),
}

# NOTE: reaction enthalpy (dH_rxn) for the reactions in J/mol
reaction_enthalpies = {
    "r1": CustomProp(value=-50000, unit="J/mol"),
}

# NOTE: volumetric heat capacity of liquid mixture in J/m3.K
liquid_mixture_volumetric_heat_capacity = CustomProp(
    value=1100*4000,  # J/m3.K
    unit="J/m3.K",
)

# ! thermo inputs
thermo_inputs = {
    # "gas_heat_capacity": constant_gas_heat_capacity,
    # "liquid_heat_capacity": constant_liquid_heat_capacity,
    "molecular_weight": molecular_weight,
    "liquid_density": constant_liquid_density,
    "reaction_enthalpy": reaction_enthalpies,
    "liquid_mixture_volumetric_heat_capacity": liquid_mixture_volumetric_heat_capacity,
}

# ====================================================
# SECTION: model inputs
# ====================================================
# NOTE: fixed reactor holdup volume [m3]
reactor_volume = Volume(
    value=3.0,
    unit="m3",
)

# NOTE: feed stream temperature [K]
inlet_temperature = Temperature(
    value=310,
    unit="K",
)

# NOTE: feed component molar flow rates [mol/s]
feed_mole_flow = {
    "A-l": CustomProp(value=100, unit="mol/s"),
    "B-l": CustomProp(value=90, unit="mol/s"),
    "C-l": CustomProp(value=0.0, unit="mol/s"),
    "D-l": CustomProp(value=0.0, unit="mol/s"),
}

# NOTE: model inputs for CSTR
# ! constant volume
model_inputs = {
    "inlet_flows": feed_mole_flow,
    "reactor_volume": reactor_volume,
    "inlet_temperature": inlet_temperature,
}

# NOTE: volume span for PFR simulation in m3
volume_span = (0, reactor_volume.value)

print("[bold green]Model inputs successfully defined![/bold green]")

# ====================================================
# SECTION: build thermo source
# ====================================================
thermo_source = build_thermo_source(
    components=components,
    model_source=None,
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
_reactor: PFRReactor = create_pfr_reactor(
    model_inputs=model_inputs,
    thermo_source=thermo_source,
)
print("[bold green]reactor successfully created![/bold green]")
print(_reactor)

# NOTE: simulate CSTR
simulation_results = _reactor.simulate(
    volume_span=volume_span,
    solver_options={
        "method": "Radau",
        "rtol": 1e-6,
        "atol": 1e-9,
        # "max_step": 0.001,
    },
    mode="log"
)
print("[bold green]simulation completed![/bold green]")
# print(simulation_results)

# NOTE: plot results
if simulation_results is not None:
    plot_pfr_reactor_result(
        result=simulation_results,
        components=components,
    )
