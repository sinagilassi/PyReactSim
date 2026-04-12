# import packages/modules
from typing import Dict, List
from pythermodb_settings.models import CustomProperty
from pyreactlab_core.models.reaction import Reaction

from pyreactsim.models import rArgs, rParams, rRet, X, rXs, ReactionRateExpression

# NOTE: replace with your actual liquid source
from examples.source.liquid_load_model_source import CH3COOH, CH3OH, C3H6O2, H2O, model_source


# ====================================================
# SECTION: Reaction Rate Expression
# ====================================================

components = [CH3COOH, CH3OH, C3H6O2, H2O]

reaction = Reaction(
    name="reaction 1",
    reaction="CH3COOH(l) + CH3OH(l) <=> C3H6O2(l) + H2O(l)",
    components=components
)

states: rXs = {
    "CH3COOH-l": X(component=CH3COOH, order=1, unit="mol/m3"),
    "CH3OH-l": X(component=CH3OH, order=1, unit="mol/m3"),
    "C3H6O2-l": X(component=C3H6O2, order=1, unit="mol/m3"),
    "H2O-l": X(component=H2O, order=1, unit="mol/m3"),
}

rate_params: rParams = {
    # Artificial mild reversible kinetics for debugging
    "kf": CustomProperty(value=1.0e-6, unit="m3/mol.s", symbol="k_f"),
    "kr": CustomProperty(value=2.0e-7, unit="m3/mol.s", symbol="k_r"),
}

rate_return: rRet = CustomProperty(
    value=0.0,
    unit="mol/m3.s",
    symbol="r1"
)

rate_args: rArgs = {
    "T": CustomProperty(value=0.0, unit="K", symbol="T"),
    "rho_B": CustomProperty(value=0.0, unit="kg/m3", symbol="rho_B"),
}


def r1(Xs: Dict[str, X], args: rArgs, params: rParams) -> CustomProperty:
    """
    Artificial reversible liquid-phase esterification rate for debugging.

    Reaction:
        CH3COOH(l) + CH3OH(l) <=> C3H6O2(l) + H2O(l)

    Incoming states:
        Xs[...] are concentrations in mol/m3

    Intrinsic artificial rate:
        r' = kf*C_acid*C_meoh - kr*C_meac*C_h2o

    Reactor-volume rate:
        r = rho_B * r'

    Returned unit:
        mol/m3.s
    """

    kf = params["kf"].value
    kr = params["kr"].value
    rho_B = args["rho_B"].value

    c_acid = Xs["CH3COOH-l"].value
    c_meoh = Xs["CH3OH-l"].value
    c_meac = Xs["C3H6O2-l"].value
    c_h2o = Xs["H2O-l"].value

    forward = (
        (c_acid ** Xs["CH3COOH-l"].order) *
        (c_meoh ** Xs["CH3OH-l"].order)
    )

    reverse = (
        (c_meac ** Xs["C3H6O2-l"].order) *
        (c_h2o ** Xs["H2O-l"].order)
    )

    r_mass = kf * forward - kr * reverse
    r_volume = rho_B * r_mass

    return CustomProperty(
        name="r1",
        description="Artificial mild reversible rate for debugging liquid PBR behavior",
        value=r_volume,
        unit="mol/m3.s",
        symbol="r1"
    )


rate_expression = ReactionRateExpression(
    name="reaction 1",
    basis="concentration",
    components=components,
    reaction=reaction,
    params=rate_params,
    args=rate_args,
    ret=rate_return,
    state=states,
    state_key="Formula-State",
    eq=r1,
    component_key="Name-Formula"
)

reaction_rates: List[ReactionRateExpression] = [rate_expression]
