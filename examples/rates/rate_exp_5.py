# import packages/modules
import logging
import math
from rich import print
from typing import Callable, Dict, Optional, Union, List, Any
from pythermodb_settings.models import CustomProperty
from pyreactlab_core.models.reaction import Reaction
# locals
from pyreactsim.models import rArgs, rParams, rRet, X, rXs, ReactionRateExpression
# ! model source
from examples.source.gas_model_source_exp_1 import C2H4, H2, C2H6


# ====================================================
# SECTION: Reaction Rate Expression
# ====================================================

# NOTE: Components
components = [C2H4, H2, C2H6]

# NOTE: Reaction
reaction = Reaction(
    name="reaction 1",
    reaction="C2H4(g) + H2(g) => C2H6(g)",
    components=components
)

# NOTE:
# Pressure-based, gas-phase, template-consistent rate expression:
# r = k * P_C2H4 * P_H2
#
# Units:
#   P_i : atm
#   r   : mol/m3.s
#   k   : mol/m3.s.atm2

states: rXs = {
    'C2H4-g': X(component=C2H4, order=1, unit="atm"),
    'H2-g': X(component=H2, order=1, unit="atm"),
}

rate_params: rParams = {
    'k': CustomProperty(value=0.5, unit="mol/m3.s.atm2", symbol="k"),
}

rate_return: rRet = CustomProperty(value=0, unit="mol/m3.s", symbol="r1")

rate_args: rArgs = {
    'T': CustomProperty(value=0, unit="K", symbol="T")
}


def r1(X: Dict[str, X], args: rArgs, params: rParams) -> CustomProperty:
    # NOTE: k may later be made temperature dependent
    k = params['k'].value

    # r = k * P_C2H4 * P_H2
    rExp = k * (X['C2H4-g'].value**X['C2H4-g'].order) * \
        (X['H2-g'].value**X['H2-g'].order)

    return CustomProperty(
        name="r1",
        description="Reaction rate for ethylene hydrogenation",
        value=rExp,
        unit="mol/m3.s",
        symbol="r1"
    )


# ! reaction rate expression
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

reaction_rates: List[ReactionRateExpression] = [rate_expression]
