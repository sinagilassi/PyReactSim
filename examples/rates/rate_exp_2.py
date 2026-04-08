# import packages/modules
import math
from typing import Dict

from pythermodb_settings.models import CustomProperty
from pyreactlab_core.models.reaction import Reaction

# locals
from pyreactsim.models import rArgs, rParams, rRet, X, rXs, ReactionRateExpression

# ! model source
# from examples.source.gas_model_source_exp_1 import components, CO2, H2, CH3OH, H2O
# ! load from file
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
# basis='pressure'
# all X[...].value must be partial pressures in Pa
states: rXs = {
    'CO2-g': X(component=CO2, order=1),
    'H2-g': X(component=H2, order=1),
    'CH3OH-g': X(component=CH3OH, order=1),
    'H2O-g': X(component=H2O, order=1),
}

rate_params: rParams = {
    # user-facing pressure is Pa, but kinetic expression is evaluated internally in bar

    # main kinetic coefficient
    'k_meoh': CustomProperty(
        value=1.0,
        unit="mol/m3.s.bar2",
        symbol="k_meoh"
    ),

    # equilibrium constant of the overall reaction
    'K_eq': CustomProperty(
        value=1.0,
        unit="-",
        symbol="K_eq"
    ),

    # adsorption / inhibition parameters
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
}

rate_return: rRet = CustomProperty(
    value=0,
    unit="mol/m3.s",
    symbol="r1"
)

rate_args: rArgs = {
    'T': CustomProperty(value=0, unit="K", symbol="T")
}


def r1(X: Dict[str, X], args: rArgs, params: rParams) -> CustomProperty:
    """
    Pressure-based methanol synthesis rate expression.

    Reaction:
        CO2(g) + 3H2(g) <=> CH3OH(g) + H2O(g)

    External input units:
        P_i : Pa

    Internal evaluation:
        bar

    Output:
        mol/m3.s
    """

    # ------------------------------------------
    # states: partial pressures in Pa
    # ------------------------------------------
    P_CO2_pa = X['CO2-g'].value
    P_H2_pa = X['H2-g'].value
    P_CH3OH_pa = X['CH3OH-g'].value
    P_H2O_pa = X['H2O-g'].value

    # ------------------------------------------
    # convert Pa -> bar
    # ------------------------------------------
    PA_TO_BAR = 1.0e-5

    P_CO2 = P_CO2_pa * PA_TO_BAR
    P_H2 = P_H2_pa * PA_TO_BAR
    P_CH3OH = P_CH3OH_pa * PA_TO_BAR
    P_H2O = P_H2O_pa * PA_TO_BAR

    # ------------------------------------------
    # parameters
    # ------------------------------------------
    k_meoh = params['k_meoh'].value
    K_eq = params['K_eq'].value
    K_redox = params['K_redox'].value
    K_H2 = params['K_H2'].value
    K_H2O = params['K_H2O'].value
    K_CO2 = params['K_CO2'].value

    # ------------------------------------------
    # protection against division by zero
    # ------------------------------------------
    eps = 1.0e-30
    P_CO2_eff = max(P_CO2, eps)
    P_H2_eff = max(P_H2, eps)

    # ------------------------------------------
    # driving force
    # 1 - (P_CH3OH * P_H2O) / (K_eq * P_CO2 * P_H2^3)
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
        description="Reaction rate for methanol synthesis from CO2 and H2 (pressure basis, Pa input)",
        value=r_exp,
        unit="mol/m3.s",
        symbol="r1"
    )


rate_expression = ReactionRateExpression(
    name="reaction 1",
    basis='pressure',
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
