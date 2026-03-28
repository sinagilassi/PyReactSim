# import packages/modules
from typing import Dict, List
from pythermodb_settings.models import CustomProperty
from pyreactlab_core.models.reaction import Reaction

from pyreactsim.models import (
    rArgs, rParams, rRet, X, rXs, ReactionRateExpression
)

# ! model source
# assumes these are defined in your component source file
from model_source_exp_1 import components, CO2, H2, CO, CH3OH, H2O


# ====================================================
# SECTION: REACTION 1
# CO2 + 3H2 => CH3OH + H2O
# ====================================================

reaction_1 = Reaction(
    name="reaction 1",
    reaction="CO2(g) + 3H2(g) => CH3OH(g) + H2O(g)",
    components=components
)

states_1: rXs = {
    "CO2-g": X(component=CO2, order=1),
    "H2-g": X(component=H2, order=2),   # kept same style as your example
}

params_1: rParams = {
    "k1": CustomProperty(value=2.0e-11, unit="mol/m3.s.Pa3", symbol="k1"),
}

ret_1: rRet = CustomProperty(value=0.0, unit="mol/m3.s", symbol="r1")

args_1: rArgs = {
    "T": CustomProperty(value=0.0, unit="K", symbol="T")
}


def r1(Xs: Dict[str, X], args: rArgs, params: rParams) -> CustomProperty:
    k1 = params["k1"].value

    p_co2 = Xs["CO2-g"].value
    p_h2 = Xs["H2-g"].value

    # simple pressure-based power-law rate
    rate = k1 * (p_co2 ** 1) * (p_h2 ** 2)

    return CustomProperty(
        name="r1",
        description="Rate of reaction 1: CO2 + 3H2 => CH3OH + H2O",
        value=rate,
        unit="mol/m3.s",
        symbol="r1"
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
    component_key="Name-Formula"
)


# ====================================================
# SECTION: REACTION 2
# CO2 + H2 => CO + H2O
# ====================================================

reaction_2 = Reaction(
    name="reaction 2",
    reaction="CO2(g) + H2(g) => CO(g) + H2O(g)",
    components=components
)

states_2: rXs = {
    "CO2-g": X(component=CO2, order=1),
    "H2-g": X(component=H2, order=1),
}

params_2: rParams = {
    "k2": CustomProperty(value=1.0e-10, unit="mol/m3.s.Pa2", symbol="k2"),
}

ret_2: rRet = CustomProperty(value=0.0, unit="mol/m3.s", symbol="r2")

args_2: rArgs = {
    "T": CustomProperty(value=0.0, unit="K", symbol="T")
}


def r2(Xs: Dict[str, X], args: rArgs, params: rParams) -> CustomProperty:
    k2 = params["k2"].value

    p_co2 = Xs["CO2-g"].value
    p_h2 = Xs["H2-g"].value

    rate = k2 * (p_co2 ** 1) * (p_h2 ** 1)

    return CustomProperty(
        name="r2",
        description="Rate of reaction 2: CO2 + H2 => CO + H2O",
        value=rate,
        unit="mol/m3.s",
        symbol="r2"
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
    component_key="Name-Formula"
)


# ====================================================
# SECTION: REACTION 3
# CO + 2H2 => CH3OH
# ====================================================

reaction_3 = Reaction(
    name="reaction 3",
    reaction="CO(g) + 2H2(g) => CH3OH(g)",
    components=components
)

states_3: rXs = {
    "CO-g": X(component=CO, order=1),
    "H2-g": X(component=H2, order=2),
}

params_3: rParams = {
    "k3": CustomProperty(value=5.0e-11, unit="mol/m3.s.Pa3", symbol="k3"),
}

ret_3: rRet = CustomProperty(value=0.0, unit="mol/m3.s", symbol="r3")

args_3: rArgs = {
    "T": CustomProperty(value=0.0, unit="K", symbol="T")
}


def r3(Xs: Dict[str, X], args: rArgs, params: rParams) -> CustomProperty:
    k3 = params["k3"].value

    p_co = Xs["CO-g"].value
    p_h2 = Xs["H2-g"].value

    rate = k3 * (p_co ** 1) * (p_h2 ** 2)

    return CustomProperty(
        name="r3",
        description="Rate of reaction 3: CO + 2H2 => CH3OH",
        value=rate,
        unit="mol/m3.s",
        symbol="r3"
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
    component_key="Name-Formula"
)


# ====================================================
# SECTION: COLLECT ALL RATE EXPRESSIONS
# ====================================================

rate_expressions: List[ReactionRateExpression] = [
    rate_expression_1,
    rate_expression_2,
    rate_expression_3,
]
