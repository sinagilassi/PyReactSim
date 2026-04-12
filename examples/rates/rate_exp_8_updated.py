# import packages/modules
import logging
import math
from rich import print
from typing import Callable, Dict, Optional, Union, List, Any
from pythermodb_settings.models import CustomProperty
from pyreactlab_core.models.reaction import Reaction

# locals
from pyreactsim.models import rArgs, rParams, rRet, X, rXs, ReactionRateExpression

# NOTE: Replace these imports with the exact component objects from your liquid reference module.
from examples.source.liquid_load_model_source import CH3COOH, CH3OH, C3H6O2, H2O, model_source


# ====================================================
# SECTION: Reaction Rate Expression
# ====================================================

# Components
components = [CH3COOH, CH3OH, C3H6O2, H2O]

# Reaction definition
reaction = Reaction(
    name="reaction 1",
    reaction="CH3COOH(l) + CH3OH(l) <=> C3H6O2(l) + H2O(l)",
    components=components
)

# States (concentration-based, liquid phase)
states: rXs = {
    'CH3COOH-l': X(component=CH3COOH, order=1, unit="mol/m3"),
    'CH3OH-l': X(component=CH3OH, order=1, unit="mol/m3"),
    'C3H6O2-l': X(component=C3H6O2, order=1, unit="mol/m3"),
    'H2O-l': X(component=H2O, order=1, unit="mol/m3"),
}

# Rate parameters (Arrhenius form)
rate_params: rParams = {
    # Forward Arrhenius parameters
    'A_f': CustomProperty(value=1.0e8, unit="m3/mol.s", symbol="A_f"),
    'Ea_f': CustomProperty(value=55000, unit="J/mol", symbol="Ea_f"),

    # Reverse Arrhenius parameters
    'A_r': CustomProperty(value=2.0e7, unit="m3/mol.s", symbol="A_r"),
    'Ea_r': CustomProperty(value=48000, unit="J/mol", symbol="Ea_r"),
}

rate_return: rRet = CustomProperty(value=0, unit="mol/m3.s", symbol="r1")

rate_args: rArgs = {
    'T': CustomProperty(value=0, unit="K", symbol="T")
}


def r1(X: Dict[str, X], args: rArgs, params: rParams) -> CustomProperty:
    """
    Reversible liquid-phase esterification rate expression.

    Reaction
    --------
    CH3COOH(l) + CH3OH(l) <=> C3H6O2(l) + H2O(l)

    Project states
    --------------
    X[...].value are liquid-phase concentrations in mol/m3.

    Internal model
    --------------
    Forward and reverse rate constants are evaluated with Arrhenius form:
        kf = A_f * exp(-Ea_f / (R * T))
        kr = A_r * exp(-Ea_r / (R * T))

    The reversible rate is written in thermodynamically consistent form:
        r = kf * (C_acid * C_meoh - (C_ester * C_h2o) / K_eq)

    with:
        K_eq = kf / kr

    Output
    ------
    Reaction rate returned on reactor-volume basis in mol/m3.s.
    """

    R = 8.314  # J/mol.K

    # Arrhenius parameters
    A_f = params['A_f'].value
    Ea_f = params['Ea_f'].value
    A_r = params['A_r'].value
    Ea_r = params['Ea_r'].value
    T = args['T'].value

    # Safety guard for temperature
    if T <= 0:
        raise ValueError("Temperature must be greater than 0 K for Arrhenius evaluation.")

    # Calculate forward and reverse rate constants
    kf = A_f * math.exp(-Ea_f / (R * T))
    kr = A_r * math.exp(-Ea_r / (R * T))

    # Thermodynamic consistency from kinetic ratio
    if kr <= 0:
        raise ValueError("Reverse rate constant kr must be greater than 0.")
    K_eq = kf / kr

    # Concentrations
    c_acid = X['CH3COOH-l'].value
    c_meoh = X['CH3OH-l'].value
    c_meac = X['C3H6O2-l'].value
    c_h2o = X['H2O-l'].value

    # Forward and reverse concentration products
    forward_term = (
        (c_acid ** X['CH3COOH-l'].order) *
        (c_meoh ** X['CH3OH-l'].order)
    )
    reverse_term = (
        (c_meac ** X['C3H6O2-l'].order) *
        (c_h2o ** X['H2O-l'].order)
    )

    # Net reversible rate
    rExp = kf * (forward_term - reverse_term / K_eq)

    return CustomProperty(
        name="r1",
        description="Thermodynamically consistent reversible rate for liquid-phase esterification of acetic acid with methanol",
        value=rExp,
        unit="mol/m3.s",
        symbol="r1"
    )


# Reaction rate expression object
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

reaction_rates: List[ReactionRateExpression] = [rate_expression]
