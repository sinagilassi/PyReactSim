# import packages/modules
import logging
import math
from rich import print
from typing import Callable, Dict, Optional, Union, List, Any
from pythermodb_settings.models import CustomProperty
from pyreactlab_core.models.reaction import Reaction
from pyreactsim_core.models import rArgs, rParams, rRet, X, rXs, ReactionRateExpression

# locals

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

# Rate parameters (now using Arrhenius form)
rate_params: rParams = {
    # Pre-exponential factor forward
    'A_f': CustomProperty(value=1.0e8, unit="m3/mol.s", symbol="A_f"),
    # Activation energy forward ~55 kJ/mol
    'Ea_f': CustomProperty(value=55000, unit="J/mol", symbol="Ea_f"),
    # Pre-exponential factor reverse
    'A_r': CustomProperty(value=2.0e7, unit="m3/mol.s", symbol="A_r"),
    # Activation energy reverse
    'Ea_r': CustomProperty(value=48000, unit="J/mol", symbol="Ea_r"),
}

rate_return: rRet = CustomProperty(value=0, unit="mol/m3.s", symbol="r1")

rate_args: rArgs = {
    # Temperature in Kelvin
    'T': CustomProperty(value=0, unit="K", symbol="T")
}


def r1(X: Dict[str, X], args: rArgs, params: rParams) -> CustomProperty:
    """
    Reversible second-order liquid-phase esterification rate:

    r = kf * C_CH3COOH * C_CH3OH - kr * C_C3H6O2 * C_H2O

    where kf and kr follow Arrhenius law:
    kf = A_f * exp(-Ea_f / (R * T))
    kr = A_r * exp(-Ea_r / (R * T))

    Units:
        C_i : mol/m³
        r   : mol/(m³·s)
        kf, kr : m³/(mol·s)
    """

    R = 8.314  # J/(mol·K)

    A_f = params['A_f'].value
    Ea_f = params['Ea_f'].value
    A_r = params['A_r'].value
    Ea_r = params['Ea_r'].value
    T = args['T'].value

    # Calculate rate constants
    kf = A_f * math.exp(-Ea_f / (R * T))
    kr = A_r * math.exp(-Ea_r / (R * T))

    # Concentrations
    c_acid = X['CH3COOH-l'].value
    c_meoh = X['CH3OH-l'].value
    c_meac = X['C3H6O2-l'].value
    c_h2o = X['H2O-l'].value

    # Net rate
    rExp = (
        kf * (c_acid ** X['CH3COOH-l'].order) * (c_meoh ** X['CH3OH-l'].order)
        - kr * (c_meac ** X['C3H6O2-l'].order) * (c_h2o ** X['H2O-l'].order)
    )

    return CustomProperty(
        name="r1",
        description="Reversible second-order rate for liquid-phase Fischer esterification of acetic acid with methanol (concentration basis)",
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
