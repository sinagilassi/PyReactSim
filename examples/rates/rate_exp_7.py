# import packages/modules
import logging
import math
from rich import print
from typing import Callable, Dict, Optional, Union, List, Any
from pythermodb_settings.models import CustomProperty
from pyreactlab_core.models.reaction import Reaction
from pyreactsim_core.models import rArgs, rParams, rRet, X, rXs, ReactionRateExpression
# locals
# ! model source
# ! create inline
# from examples.source.gas_model_source_exp_1 import model_source
# ! load from file
from examples.source.gas_load_model_source import model_source, CO2, H2, CH3OH, H2O

# ====================================================
# SECTION: Reaction Rate Expression
# ====================================================

# NOTE: Components
components = [CO2, H2, CH3OH, H2O]

# NOTE: Reaction
reaction = Reaction(
    name="reaction 1",
    reaction="CO2(g) + 3H2(g) => CH3OH(g) + H2O(g)",
    components=components
)


states: rXs = {
    'CO2-g': X(component=CO2, order=1, unit="Pa"),
    'H2-g': X(component=H2, order=2, unit="Pa"),
}

rate_params: rParams = {
    'k': CustomProperty(value=2e-11, unit="mol/m3.s.Pa2", symbol="k"),
}

rate_return: rRet = CustomProperty(value=0, unit="mol/m3.s", symbol="r1")

rate_args: rArgs = {
    'T': CustomProperty(value=0, unit="K", symbol="T"),
    'rho_B': CustomProperty(value=0, unit="kg/m3", symbol="rho_B"),
}


def r1(X: Dict[str, X], args: rArgs, params: rParams) -> CustomProperty:
    # rate constant k function of temperature and pressure
    k = params['k'].value

    # bulk density conversion for PBR: r_V = rho_B * r'
    rho_B = args["rho_B"].value

    # ??? rate expression: r = k*[A]^order_A*[B]^order_B
    rExp = k*(X['CO2-g'].value**X['CO2-g'].order) * \
        (X['H2-g'].value**X['H2-g'].order)

    # >> convert to reactor-volume basis by bulk density for PBR
    rExp = rho_B * rExp

    return CustomProperty(
        name="r1",
        description="Reaction rate for reaction 1",
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

reaction_rates = [rate_expression]
