import math
from rich import print
from typing import Dict, Tuple

from pythermodb_settings.models import CustomProperty, Temperature, Pressure, CustomProp
from pyreactlab_core.models.reaction import Reaction

from pyreactsim.models import rArgs, rParams, rRet, X, rXs, ReactionRateExpression
from examples.source.gas_load_model_source import CO2, H2, CH3OH, H2O, CO, model_source


# ====================================================
# SECTION: Shared configuration
# ====================================================

components = [CO2, H2, CO, CH3OH, H2O]

# Paper basis:
# - fugacity is approximated by partial pressure in bar
# - intrinsic rates are mol/kgcat.s
# - final project output must be mol/m3.s

shared_states: rXs = {
    "CO-g": X(component=CO, order=1, unit="bar"),
    "CO2-g": X(component=CO2, order=1, unit="bar"),
    "H2-g": X(component=H2, order=1, unit="bar"),
    "CH3OH-g": X(component=CH3OH, order=1, unit="bar"),
    "H2O-g": X(component=H2O, order=1, unit="bar"),
}

shared_rate_args: rArgs = {
    "T": CustomProperty(value=0.0, unit="K", symbol="T"),
    "rho_B": CustomProperty(value=1770.0, unit="kgcat/m3", symbol="rho_B"),
    "a": CustomProperty(value=1.0, unit="-", symbol="a"),
}

shared_rate_params: rParams = {
    # Gas constant
    "R": CustomProperty(value=8.314462618, unit="J/mol.K", symbol="R"),

    # k = A * exp(B/RT), rates from Graaf constants cited by the paper
    "A_k1": CustomProperty(value=4.89e7, unit="mol/kgcat.s.bar", symbol="A_k1"),
    "B_k1": CustomProperty(value=-63000.0, unit="J/mol", symbol="B_k1"),
    "A_k2": CustomProperty(value=1.09e5, unit="mol/kgcat.s.bar", symbol="A_k2"),
    "B_k2": CustomProperty(value=-87500.0, unit="J/mol", symbol="B_k2"),
    "A_k3": CustomProperty(value=9.64e6, unit="mol/kgcat.s.bar", symbol="A_k3"),
    "B_k3": CustomProperty(value=-152900.0, unit="J/mol", symbol="B_k3"),

    # K = A * exp(B/RT)
    "A_KCO": CustomProperty(value=2.16e-5, unit="1/bar", symbol="A_KCO"),
    "B_KCO": CustomProperty(value=46800.0, unit="J/mol", symbol="B_KCO"),
    "A_KCO2": CustomProperty(value=7.05e-7, unit="1/bar", symbol="A_KCO2"),
    "B_KCO2": CustomProperty(value=61700.0, unit="J/mol", symbol="B_KCO2"),
    "A_KH2O_over_sqrt_KH2": CustomProperty(
        value=6.37e-9, unit="1/bar^0.5", symbol="A_KH2O_over_sqrt_KH2"
    ),
    "B_KH2O_over_sqrt_KH2": CustomProperty(
        value=84000.0, unit="J/mol", symbol="B_KH2O_over_sqrt_KH2"
    ),

    # Kp = 10^(A/T - B)
    "A_KP1": CustomProperty(value=5139.0, unit="K", symbol="A_KP1"),
    "B_KP1": CustomProperty(value=12.621, unit="-", symbol="B_KP1"),
    "A_KP2": CustomProperty(value=3066.0, unit="K", symbol="A_KP2"),
    "B_KP2": CustomProperty(value=10.592, unit="-", symbol="B_KP2"),
    "A_KP3": CustomProperty(value=-2073.0, unit="K", symbol="A_KP3"),
    "B_KP3": CustomProperty(value=-2.029, unit="-", symbol="B_KP3"),
}


def _calc_temperature_dependent_terms(T: float, params: rParams) -> Dict[str, float]:
    R = params["R"].value
    inv_RT = 1.0 / (R * T)

    k1 = params["A_k1"].value * math.exp(params["B_k1"].value * inv_RT)
    k2 = params["A_k2"].value * math.exp(params["B_k2"].value * inv_RT)
    k3 = params["A_k3"].value * math.exp(params["B_k3"].value * inv_RT)

    KCO = params["A_KCO"].value * math.exp(params["B_KCO"].value * inv_RT)
    KCO2 = params["A_KCO2"].value * math.exp(params["B_KCO2"].value * inv_RT)
    KH2O_over_sqrt_KH2 = params["A_KH2O_over_sqrt_KH2"].value * math.exp(
        params["B_KH2O_over_sqrt_KH2"].value * inv_RT
    )

    KP1 = 10.0 ** (params["A_KP1"].value / T - params["B_KP1"].value)
    KP2 = 10.0 ** (params["A_KP2"].value / T - params["B_KP2"].value)
    KP3 = 10.0 ** (params["A_KP3"].value / T - params["B_KP3"].value)

    return {
        "k1": k1,
        "k2": k2,
        "k3": k3,
        "KCO": KCO,
        "KCO2": KCO2,
        "KH2O_over_sqrt_KH2": KH2O_over_sqrt_KH2,
        "KP1": KP1,
        "KP2": KP2,
        "KP3": KP3,
    }


def _common_terms(Xs: Dict[str, X], args: rArgs, params: rParams) -> Tuple[Dict[str, float], Dict[str, float], float]:
    T = args["T"].value
    if T <= 0.0:
        raise ValueError("Temperature T must be > 0 K.")

    p = {
        "CO": Xs["CO-g"].value,
        "CO2": Xs["CO2-g"].value,
        "H2": Xs["H2-g"].value,
        "CH3OH": Xs["CH3OH-g"].value,
        "H2O": Xs["H2O-g"].value,
    }

    tdep = _calc_temperature_dependent_terms(T, params)

    eps = 1.0e-30
    p_h2 = max(p["H2"], eps)
    adsorption_term = (
        1.0
        + tdep["KCO"] * p["CO"]
        + tdep["KCO2"] * p["CO2"]
    )
    hydrogen_water_term = math.sqrt(
        p_h2) + tdep["KH2O_over_sqrt_KH2"] * p["H2O"]
    denom = max(adsorption_term * max(hydrogen_water_term, eps), eps)

    return p, tdep, denom


# ====================================================
# SECTION: Reaction 1
# CO + 2H2 <=> CH3OH
# ====================================================

reaction_1 = Reaction(
    name="reaction 1",
    reaction="CO(g) + 2H2(g) <=> CH3OH(g)",
    components=components,
)

params_1: rParams = dict(shared_rate_params)
args_1: rArgs = dict(shared_rate_args)
states_1: rXs = dict(shared_states)
ret_1: rRet = CustomProperty(value=0.0, unit="mol/m3.s", symbol="r1")


def r1(Xs: Dict[str, X], args: rArgs, params: rParams) -> CustomProperty:
    """
    Graaf-form CO hydrogenation rate from the cited methanol paper.

    Input interpretation:
    - X[...].value are partial pressures in bar (used in place of fugacities)
    - intrinsic rate is mol/kgcat.s
    - converted to reactor-volume basis with rho_B and activity a
    """

    rho_B = args["rho_B"].value
    a = args["a"].value

    p, tdep, denom = _common_terms(Xs, args, params)
    p_h2 = max(p["H2"], 1.0e-30)
    kp1 = max(tdep["KP1"], 1.0e-30)

    driving = p["CO"] * (p_h2 ** 1.5) - p["CH3OH"] / (math.sqrt(p_h2) * kp1)
    r_mass = tdep["k1"] * tdep["KCO"] * driving / denom
    r_volume = a * rho_B * r_mass

    return CustomProperty(
        name="r1",
        description="CO hydrogenation rate with Graaf-form denominator (partial pressure basis)",
        value=r_volume,
        unit="mol/m3.s",
        symbol="r1",
    )


rate_expression_1 = ReactionRateExpression(
    name="reaction 1",
    basis="pressure",
    components=components,
    reaction=reaction_1,
    params=params_1,
    args=args_1,
    ret=ret_1,
    state=states_1,
    state_key="Formula-State",
    eq=r1,
    component_key="Name-Formula",
)


# ====================================================
# SECTION: Reaction 2
# CO2 + 3H2 <=> CH3OH + H2O
# ====================================================

reaction_2 = Reaction(
    name="reaction 2",
    reaction="CO2(g) + 3H2(g) <=> CH3OH(g) + H2O(g)",
    components=components,
)

params_2: rParams = dict(shared_rate_params)
args_2: rArgs = dict(shared_rate_args)
states_2: rXs = dict(shared_states)
ret_2: rRet = CustomProperty(value=0.0, unit="mol/m3.s", symbol="r2")


def r2(Xs: Dict[str, X], args: rArgs, params: rParams) -> CustomProperty:
    """
    Graaf-form CO2 hydrogenation rate from the cited methanol paper.

    Input interpretation:
    - X[...].value are partial pressures in bar (used in place of fugacities)
    - intrinsic rate is mol/kgcat.s
    - converted to reactor-volume basis with rho_B and activity a
    """

    rho_B = args["rho_B"].value
    a = args["a"].value

    p, tdep, denom = _common_terms(Xs, args, params)
    p_h2 = max(p["H2"], 1.0e-30)
    kp2 = max(tdep["KP2"], 1.0e-30)

    driving = (
        p["CO2"] * (p_h2 ** 1.5)
        - (p["CH3OH"] * p["H2O"]) / ((p_h2 ** 1.5) * kp2)
    )
    r_mass = tdep["k2"] * tdep["KCO2"] * driving / denom
    r_volume = a * rho_B * r_mass

    return CustomProperty(
        name="r2",
        description="CO2 hydrogenation rate with Graaf-form denominator (partial pressure basis)",
        value=r_volume,
        unit="mol/m3.s",
        symbol="r2",
    )


rate_expression_2 = ReactionRateExpression(
    name="reaction 2",
    basis="pressure",
    components=components,
    reaction=reaction_2,
    params=params_2,
    args=args_2,
    ret=ret_2,
    state=states_2,
    state_key="Formula-State",
    eq=r2,
    component_key="Name-Formula",
)


# ====================================================
# SECTION: Reaction 3
# CO2 + H2 <=> CO + H2O
# ====================================================

reaction_3 = Reaction(
    name="reaction 3",
    reaction="CO2(g) + H2(g) <=> CO(g) + H2O(g)",
    components=components,
)

params_3: rParams = dict(shared_rate_params)
args_3: rArgs = dict(shared_rate_args)
states_3: rXs = dict(shared_states)
ret_3: rRet = CustomProperty(value=0.0, unit="mol/m3.s", symbol="r3")


def r3(Xs: Dict[str, X], args: rArgs, params: rParams) -> CustomProperty:
    """
    Graaf-form RWGS rate from the cited methanol paper.

    Input interpretation:
    - X[...].value are partial pressures in bar (used in place of fugacities)
    - intrinsic rate is mol/kgcat.s
    - converted to reactor-volume basis with rho_B and activity a
    """

    rho_B = args["rho_B"].value
    a = args["a"].value

    p, tdep, denom = _common_terms(Xs, args, params)
    kp3 = max(tdep["KP3"], 1.0e-30)

    driving = p["CO2"] * p["H2"] - (p["H2O"] * p["CO"]) / kp3
    r_mass = tdep["k3"] * tdep["KCO2"] * driving / denom
    r_volume = a * rho_B * r_mass

    return CustomProperty(
        name="r3",
        description="RWGS rate with Graaf-form denominator (partial pressure basis)",
        value=r_volume,
        unit="mol/m3.s",
        symbol="r3",
    )


rate_expression_3 = ReactionRateExpression(
    name="reaction 3",
    basis="pressure",
    components=components,
    reaction=reaction_3,
    params=params_3,
    args=args_3,
    ret=ret_3,
    state=states_3,
    state_key="Formula-State",
    eq=r3,
    component_key="Name-Formula",
)


reaction_rates = [
    rate_expression_1,
    rate_expression_2,
    rate_expression_3,
]

# ---------------------------------------
# SECTION: Evaluate Reactions
# ---------------------------------------
# NOTE: temperature
temperature = Temperature(
    value=503,
    unit="K"
)

# NOTE: pressure
pressure = Pressure(
    value=74.98,
    unit="bar"
)

# NOTE: args
args = {
    'rho_B': CustomProperty(value=1770.0, unit="kgcat/m3", symbol="rho_B"),
}

# calculate rate expression value
result = rate_expression_1.calc(
    xi={
        'CH3OH-g': CustomProperty(value=0.2, unit="Pa", symbol="CH3OH-g"),
        'CO-g': CustomProperty(value=0.4, unit="Pa", symbol="CO-g")
    },
    args=args,
    temperature=temperature,
    pressure=pressure,
    mode="log"
)
print(result)
