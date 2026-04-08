# import packages/modules
import math
from typing import Dict

from pythermodb_settings.models import CustomProperty
from pyreactlab_core.models.reaction import Reaction

# locals
from pyreactsim.models import rArgs, rParams, rRet, X, rXs, ReactionRateExpression

# ! model source
# from examples.source.gas_model_source_exp_1 import components, CO2, H2, CH3OH, H2O
from examples.source.gas_load_model_source import model_source, CO2, H2, CH3OH, H2O


# ====================================================
# SECTION: Reaction Rate Expression
# ====================================================

components = [CO2, H2, CH3OH, H2O]

reaction = Reaction(
    name="reaction 1",
    reaction="CO2(g) + 3H2(g) => CH3OH(g) + H2O(g)",
    components=components
)

# NOTE:
# basis='concentration'
# all X[...].value must be concentrations in mol/m3
states: rXs = {
    'CO2-g': X(component=CO2, order=1),
    'H2-g': X(component=H2, order=1),
    'CH3OH-g': X(component=CH3OH, order=1),
    'H2O-g': X(component=H2O, order=1),
}

rate_params: rParams = {
    # kinetic model is internally evaluated using partial pressures in bar

    'k_meoh': CustomProperty(
        value=1.0,
        unit="mol/m3.s.bar2",
        symbol="k_meoh"
    ),

    'K_eq': CustomProperty(
        value=1.0,
        unit="-",
        symbol="K_eq"
    ),

    'K_redox': CustomProperty(
        value=1.0,
        unit="-",
        symbol="K_redox"
    ),
    'K_H2': CustomProperty(
        value=1.0,
        unit="1/bar",
        symbol="K_H2"
    ),
    'K_H2O': CustomProperty(
        value=1.0,
        unit="1/bar",
        symbol="K_H2O"
    ),
    'K_CO2': CustomProperty(
        value=1.0,
        unit="1/bar^1.5",
        symbol="K_CO2"
    ),
    'R': CustomProperty(value=8.314462618, unit="Pa.m3/mol.K", symbol="R")
}

rate_return: rRet = CustomProperty(
    value=0,
    unit="mol/m3.s",
    symbol="r1"
)

rate_args: rArgs = {
    'T': CustomProperty(value=0, unit="K", symbol="T"),
}


def r1(X: Dict[str, X], args: rArgs, params: rParams) -> CustomProperty:
    """
    Concentration-based methanol synthesis rate expression.

    Reaction:
        CO2(g) + 3H2(g) <=> CH3OH(g) + H2O(g)

    External input units:
        C_i : mol/m3
        T   : K
        R   : Pa.m3/mol.K

    Internal steps:
        C_i -> P_i(Pa) -> P_i(bar)

    Output:
        mol/m3.s
    """

    # ------------------------------------------
    # states: concentrations in mol/m3
    # ------------------------------------------
    C_CO2 = X['CO2-g'].value
    C_H2 = X['H2-g'].value
    C_CH3OH = X['CH3OH-g'].value
    C_H2O = X['H2O-g'].value

    # ------------------------------------------
    # arguments
    # ------------------------------------------
    T = args['T'].value

    # ------------------------------------------
    # parameters
    # ------------------------------------------
    R = params['R'].value
    k_meoh = params['k_meoh'].value
    K_eq = params['K_eq'].value
    K_redox = params['K_redox'].value
    K_H2 = params['K_H2'].value
    K_H2O = params['K_H2O'].value
    K_CO2 = params['K_CO2'].value

    # ------------------------------------------
    # convert concentrations -> partial pressures
    # P_i = C_i * R * T
    # result in Pa
    # ------------------------------------------
    P_CO2_pa = C_CO2 * R * T
    P_H2_pa = C_H2 * R * T
    P_CH3OH_pa = C_CH3OH * R * T
    P_H2O_pa = C_H2O * R * T

    # ------------------------------------------
    # convert Pa -> bar
    # ------------------------------------------
    PA_TO_BAR = 1.0e-5

    P_CO2 = P_CO2_pa * PA_TO_BAR
    P_H2 = P_H2_pa * PA_TO_BAR
    P_CH3OH = P_CH3OH_pa * PA_TO_BAR
    P_H2O = P_H2O_pa * PA_TO_BAR

    # ------------------------------------------
    # protection against division by zero
    # ------------------------------------------
    eps = 1.0e-30
    P_CO2_eff = max(P_CO2, eps)
    P_H2_eff = max(P_H2, eps)

    # ------------------------------------------
    # driving force
    # ------------------------------------------
    driving_force = 1.0 - (
        (P_CH3OH * P_H2O) /
        (max(K_eq, eps) * P_CO2_eff * (P_H2_eff ** 3))
    )

    # ------------------------------------------
    # denominator term
    # ------------------------------------------
    denom = (
        1.0
        + K_redox * (P_H2O / P_H2_eff)
        + math.sqrt(max(K_H2 * P_H2, 0.0))
        + K_H2O * P_H2O
        + K_CO2 * P_CO2 * math.sqrt(max(P_H2, 0.0))
    ) ** 3

    denom = max(denom, eps)

    # ------------------------------------------
    # reaction rate
    # ------------------------------------------
    r_exp = k_meoh * P_CO2 * P_H2 * driving_force / denom

    return CustomProperty(
        name="r1",
        description="Reaction rate for methanol synthesis from CO2 and H2 (concentration basis, mol/m3 input)",
        value=r_exp,
        unit="mol/m3.s",
        symbol="r1"
    )


rate_expression = ReactionRateExpression(
    name="reaction 1",
    basis='concentration',
    components=components,
    reaction=reaction,
    params=rate_params,
    args=rate_args,
    ret=rate_return,
    state=states,
    state_key='Formula-State',
    eq=r1,
    component_key='Name-Formula'
)

reaction_rates = [rate_expression]
