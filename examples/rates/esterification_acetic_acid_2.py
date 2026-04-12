# import packages/modules
import logging
import math
from typing import Dict, List
from pythermodb_settings.models import CustomProperty
from pyreactlab_core.models.reaction import Reaction

# locals
from pyreactsim.models import rArgs, rParams, rRet, X, rXs, ReactionRateExpression

# NOTE: Replace with your actual liquid source
from examples.source.liquid_load_model_source import CH3COOH, CH3OH, C3H6O2, H2O, model_source


# ====================================================
# SECTION: Reaction Rate Expression
# ====================================================

# Components
components = [CH3COOH, CH3OH, C3H6O2, H2O]

# Reaction
reaction = Reaction(
    name="reaction 1",
    reaction="CH3COOH(l) + CH3OH(l) <=> C3H6O2(l) + H2O(l)",
    components=components
)

# ====================================================
# STATES (activity approximated by concentration)
# ====================================================

states: rXs = {
    'CH3COOH-l': X(component=CH3COOH, order=1, unit="mol/m3"),
    'CH3OH-l': X(component=CH3OH, order=1, unit="mol/m3"),
    'C3H6O2-l': X(component=C3H6O2, order=1, unit="mol/m3"),
    'H2O-l': X(component=H2O, order=1, unit="mol/m3"),
}


# ====================================================
# KINETIC PARAMETERS (FROM ARTICLE STRUCTURE)
# ====================================================

rate_params: rParams = {

    # Forward Arrhenius (esterification)
    'k1_0': CustomProperty(value=1.2e6, unit="m3/mol.s", symbol="k1_0"),
    'Ea1': CustomProperty(value=52000, unit="J/mol", symbol="Ea1"),

    # Backward Arrhenius (hydrolysis)
    'k_1_0': CustomProperty(value=8.0e5, unit="m3/mol.s", symbol="k_1_0"),
    'Ea_1': CustomProperty(value=50000, unit="J/mol", symbol="Ea_1"),

    # catalytic exponent (from paper: 0.5–1 → use 1)
    'R_exp': CustomProperty(value=1.0, unit="-", symbol="R"),

}

rate_return: rRet = CustomProperty(value=0, unit="mol/m3.s", symbol="r1")

rate_args: rArgs = {
    'T': CustomProperty(value=0, unit="K", symbol="T"),
    'rho_B': CustomProperty(value=0, unit="kg/m3", symbol="rho_B"),
}


def r1(X: Dict[str, X], args: rArgs, params: rParams) -> CustomProperty:
    """
    Esterification kinetics based on literature (activity-based model simplified)

    Original form:
        r = a_HOAc^R * (k1*a_HOAc*a_MeOH - k-1*a_MeAc*a_H2O)

    Simplified (activities ≈ concentrations):
        r' = C_acid^R * (k1*C_acid*C_meoh - k-1*C_meac*C_h2o)

    Reactor-scale:
        r = rho_B * r'

    Units:
        C_i : mol/m3
        r'  : mol/kgcat.s
        r   : mol/m3.s
    """

    Rg = 8.314  # J/mol.K

    # parameters
    k1_0 = params['k1_0'].value
    Ea1 = params['Ea1'].value

    k_1_0 = params['k_1_0'].value
    Ea_1 = params['Ea_1'].value

    R_exp = params['R_exp'].value

    T = args['T'].value
    rho_B = args['rho_B'].value

    # Arrhenius expressions
    k1 = k1_0 * math.exp(-Ea1 / (Rg * T))
    k_1 = k_1_0 * math.exp(-Ea_1 / (Rg * T))

    # concentrations
    c_acid = X['CH3COOH-l'].value
    c_meoh = X['CH3OH-l'].value
    c_meac = X['C3H6O2-l'].value
    c_h2o = X['H2O-l'].value

    # catalytic term (VERY IMPORTANT from paper)
    catalytic = c_acid ** R_exp

    # forward / reverse
    forward = k1 * (c_acid * c_meoh)
    reverse = k_1 * (c_meac * c_h2o)

    # intrinsic rate
    r_mass = catalytic * (forward - reverse)

    # convert to reactor volume basis (PBR requirement)
    rExp = rho_B * r_mass

    return CustomProperty(
        name="r1",
        description="Literature-based esterification kinetics (activity-based, reversible, catalytic)",
        value=rExp,
        unit="mol/m3.s",
        symbol="r1"
    )


# ====================================================
# ReactionRateExpression
# ====================================================

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
