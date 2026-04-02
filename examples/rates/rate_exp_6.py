# import packages/modules
import logging
import math
from rich import print
from typing import Callable, Dict, Optional, Union, List, Any
from pythermodb_settings.models import CustomProperty
from pyreactlab_core.models.reaction import Reaction

# locals
from pyreactsim.models import rArgs, rParams, rRet, X, rXs, ReactionRateExpression

# NOTE:
# Replace these imports with the exact component objects from your liquid reference module.
# Example names are illustrative and should match your actual exported variables.
from examples.source.liquid_model_source_exp_1 import CH3COOH, CH3OH, C3H6O2, H2O


# ====================================================
# SECTION: Reaction Rate Expression
# ====================================================

# NOTE: Components
components = [CH3COOH, CH3OH, C3H6O2, H2O]

# NOTE: Reaction
reaction = Reaction(
    name="reaction 1",
    reaction="CH3COOH(l) + CH3OH(l) <=> C3H6O2(l) + H2O(l)",
    components=components
)

# NOTE:
# Concentration-based, liquid-phase, reversible rate expression:
#
# r = kf * C_AA * C_MeOH - kr * C_MeAc * C_H2O
#
# Units:
#   C_i : mol/m3
#   r   : mol/m3.s
#   kf  : m3/mol.s
#   kr  : m3/mol.s
#
# Since each term is second-order overall:
#   (m3/mol.s) * (mol/m3) * (mol/m3) = mol/m3.s

states: rXs = {
    'CH3COOH-l': X(component=CH3COOH, order=1, unit="mol/m3"),
    'CH3OH-l': X(component=CH3OH, order=1, unit="mol/m3"),
    'C3H6O2-l': X(component=C3H6O2, order=1, unit="mol/m3"),
    'H2O-l': X(component=H2O, order=1, unit="mol/m3"),
}

rate_params: rParams = {
    'kf': CustomProperty(value=1.0e-6, unit="m3/mol.s", symbol="k_f"),
    'kr': CustomProperty(value=2.0e-7, unit="m3/mol.s", symbol="k_r"),
}

rate_return: rRet = CustomProperty(value=0, unit="mol/m3.s", symbol="r1")

rate_args: rArgs = {
    'T': CustomProperty(value=0, unit="K", symbol="T")
}


def r1(X: Dict[str, X], args: rArgs, params: rParams) -> CustomProperty:
    kf = params['kf'].value
    kr = params['kr'].value

    c_acid = X['CH3COOH-l'].value
    c_meoh = X['CH3OH-l'].value
    c_meac = X['C3H6O2-l'].value
    c_h2o = X['H2O-l'].value

    # Reversible liquid-phase rate:
    # r = kf*C_acid*C_meoh - kr*C_meac*C_h2o
    rExp = (
        kf * (c_acid ** X['CH3COOH-l'].order) * (c_meoh ** X['CH3OH-l'].order)
        - kr * (c_meac ** X['C3H6O2-l'].order) * (c_h2o ** X['H2O-l'].order)
    )

    return CustomProperty(
        name="r1",
        description="Reaction rate for liquid-phase esterification of acetic acid with methanol",
        value=rExp,
        unit="mol/m3.s",
        symbol="r1"
    )


# ! reaction rate expression
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
