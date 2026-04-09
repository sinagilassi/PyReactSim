import math
from typing import Dict

from pythermodb_settings.models import CustomProperty
from pyreactlab_core.models.reaction import Reaction

from pyreactsim.models import rArgs, rParams, rRet, X, rXs, ReactionRateExpression
from examples.source.gas_load_model_source import CO2, H2, CH3OH, H2O, CO, model_source


# ====================================================
# SECTION: Components and shared states
# ====================================================

components = [CO2, H2, CO, CH3OH, H2O]

# Paper uses fugacity (bar). Here we use partial pressure in bar as requested.
states: rXs = {
    "CO-g": X(component=CO, order=1, unit="bar"),
    "CO2-g": X(component=CO2, order=1, unit="bar"),
    "H2-g": X(component=H2, order=1, unit="bar"),
    "CH3OH-g": X(component=CH3OH, order=1, unit="bar"),
    "H2O-g": X(component=H2O, order=1, unit="bar"),
}

rate_args: rArgs = {
    "T": CustomProperty(value=503.0, unit="K", symbol="T"),
    "rho_B": CustomProperty(value=1770.0, unit="kgcat/m3", symbol="rho_B"),
    "a": CustomProperty(value=1.0, unit="-", symbol="a"),
}

rate_params: rParams = {
    "R": CustomProperty(value=8.314462618, unit="J/mol.K", symbol="R"),
}


# ====================================================
# SECTION: Reaction 1
# CO + 2H2 <=> CH3OH
# Eq. (4) in the paper
# ====================================================

reaction_1 = Reaction(
    name="reaction 1",
    reaction="CO(g) + 2H2(g) <=> CH3OH(g)",
    components=components,
)

ret_1: rRet = CustomProperty(value=0.0, unit="mol/m3.s", symbol="r1")


def r1(Xs: Dict[str, X], args: rArgs, params: rParams) -> CustomProperty:
    eps = 1.0e-30

    P_CO = Xs["CO-g"].value
    P_CO2 = Xs["CO2-g"].value
    P_H2 = max(Xs["H2-g"].value, eps)
    P_CH3OH = Xs["CH3OH-g"].value
    P_H2O = Xs["H2O-g"].value

    T = args["T"].value
    rho_B = args["rho_B"].value
    a = args["a"].value
    R = params["R"].value

    k1 = 4.89 * math.exp(-63000.0 / (R * T))
    KCO = 2.16 * math.exp(46800.0 / (R * T))
    KCO2 = 7.05 * math.exp(61700.0 / (R * T))
    KH2O_over_sqrt_KH2 = 6.37 * math.exp(84000.0 / (R * T))
    KP1 = 10.0 ** (5139.0 / T - 12.621)

    denominator = (
        (1.0 + KCO * P_CO + KCO2 * P_CO2)
        * (math.sqrt(P_H2) + KH2O_over_sqrt_KH2 * P_H2O)
    )
    denominator = max(denominator, eps)

    # r1 = k1*KCO*(P_CO*P_H2^(3/2) - P_CH3OH/(P_H2^(1/2)*KP1))/den
    r1_mass = k1 * KCO * (
        P_CO * (P_H2 ** 1.5) - P_CH3OH / (math.sqrt(P_H2) * max(KP1, eps))
    ) / denominator

    r1_volume = a * rho_B * r1_mass

    return CustomProperty(
        name="r1",
        description="CO hydrogenation rate (Graaf form, partial pressure in bar)",
        value=r1_volume,
        unit="mol/m3.s",
        symbol="r1",
    )


rate_expression_1 = ReactionRateExpression(
    name="reaction 1",
    basis="pressure",
    components=components,
    reaction=reaction_1,
    params=rate_params,
    args=rate_args,
    ret=ret_1,
    state=states,
    state_key="Formula-State",
    eq=r1,
    component_key="Name-Formula",
)


# ====================================================
# SECTION: Reaction 2
# CO2 + 3H2 <=> CH3OH + H2O
# Eq. (5) in the paper
# ====================================================

reaction_2 = Reaction(
    name="reaction 2",
    reaction="CO2(g) + 3H2(g) <=> CH3OH(g) + H2O(g)",
    components=components,
)

ret_2: rRet = CustomProperty(value=0.0, unit="mol/m3.s", symbol="r2")


def r2(Xs: Dict[str, X], args: rArgs, params: rParams) -> CustomProperty:
    eps = 1.0e-30

    P_CO = Xs["CO-g"].value
    P_CO2 = Xs["CO2-g"].value
    P_H2 = max(Xs["H2-g"].value, eps)
    P_CH3OH = Xs["CH3OH-g"].value
    P_H2O = Xs["H2O-g"].value

    T = args["T"].value
    rho_B = args["rho_B"].value
    a = args["a"].value
    R = params["R"].value

    k2 = 1.09 * math.exp(-87500.0 / (R * T))
    KCO = 2.16 * math.exp(46800.0 / (R * T))
    KCO2 = 7.05 * math.exp(61700.0 / (R * T))
    KH2O_over_sqrt_KH2 = 6.37 * math.exp(84000.0 / (R * T))
    KP2 = 10.0 ** (3066.0 / T - 10.592)

    denominator = (
        (1.0 + KCO * P_CO + KCO2 * P_CO2)
        * (math.sqrt(P_H2) + KH2O_over_sqrt_KH2 * P_H2O)
    )
    denominator = max(denominator, eps)

    # r2 = k2*KCO2*(P_CO2*P_H2^(3/2) - (P_H2O*P_CH3OH)/(P_H2^(3/2)*KP2))/den
    r2_mass = k2 * KCO2 * (
        P_CO2 * (P_H2 ** 1.5)
        - (P_H2O * P_CH3OH) / ((P_H2 ** 1.5) * max(KP2, eps))
    ) / denominator

    r2_volume = a * rho_B * r2_mass

    return CustomProperty(
        name="r2",
        description="CO2 hydrogenation rate (Graaf form, partial pressure in bar)",
        value=r2_volume,
        unit="mol/m3.s",
        symbol="r2",
    )


rate_expression_2 = ReactionRateExpression(
    name="reaction 2",
    basis="pressure",
    components=components,
    reaction=reaction_2,
    params=rate_params,
    args=rate_args,
    ret=ret_2,
    state=states,
    state_key="Formula-State",
    eq=r2,
    component_key="Name-Formula",
)


# ====================================================
# SECTION: Reaction 3
# CO2 + H2 <=> CO + H2O
# Eq. (6) in the paper
# ====================================================

reaction_3 = Reaction(
    name="reaction 3",
    reaction="CO2(g) + H2(g) <=> CO(g) + H2O(g)",
    components=components,
)

ret_3: rRet = CustomProperty(value=0.0, unit="mol/m3.s", symbol="r3")


def r3(Xs: Dict[str, X], args: rArgs, params: rParams) -> CustomProperty:
    eps = 1.0e-30

    P_CO = Xs["CO-g"].value
    P_CO2 = Xs["CO2-g"].value
    P_H2 = max(Xs["H2-g"].value, eps)
    P_H2O = Xs["H2O-g"].value

    T = args["T"].value
    rho_B = args["rho_B"].value
    a = args["a"].value
    R = params["R"].value

    k3 = 9.64 * math.exp(-152900.0 / (R * T))
    KCO = 2.16 * math.exp(46800.0 / (R * T))
    KCO2 = 7.05 * math.exp(61700.0 / (R * T))
    KH2O_over_sqrt_KH2 = 6.37 * math.exp(84000.0 / (R * T))
    KP3 = 10.0 ** (-2073.0 / T + 2.029)

    denominator = (
        (1.0 + KCO * P_CO + KCO2 * P_CO2)
        * (math.sqrt(P_H2) + KH2O_over_sqrt_KH2 * P_H2O)
    )
    denominator = max(denominator, eps)

    # r3 = k3*KCO2*(P_CO2*P_H2 - (P_H2O*P_CO)/KP3)/den
    r3_mass = k3 * KCO2 * (
        P_CO2 * P_H2 - (P_H2O * P_CO) / max(KP3, eps)
    ) / denominator

    r3_volume = a * rho_B * r3_mass

    return CustomProperty(
        name="r3",
        description="RWGS rate (Graaf form, partial pressure in bar)",
        value=r3_volume,
        unit="mol/m3.s",
        symbol="r3",
    )


rate_expression_3 = ReactionRateExpression(
    name="reaction 3",
    basis="pressure",
    components=components,
    reaction=reaction_3,
    params=rate_params,
    args=rate_args,
    ret=ret_3,
    state=states,
    state_key="Formula-State",
    eq=r3,
    component_key="Name-Formula",
)


reaction_rates = [rate_expression_1, rate_expression_2, rate_expression_3]
