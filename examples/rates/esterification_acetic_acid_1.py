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

# States (concentration-based)
states: rXs = {
    'CH3COOH-l': X(component=CH3COOH, order=1, unit="mol/m3"),
    'CH3OH-l': X(component=CH3OH, order=1, unit="mol/m3"),
    'C3H6O2-l': X(component=C3H6O2, order=1, unit="mol/m3"),
    'H2O-l': X(component=H2O, order=1, unit="mol/m3"),
}

# ====================================================
# KINETIC PARAMETERS (literature-style)
# ====================================================

rate_params: rParams = {
    # Arrhenius forward
    'A': CustomProperty(value=5.0e6, unit="m3/mol.s", symbol="A"),
    'Ea': CustomProperty(value=55000, unit="J/mol", symbol="Ea"),

    # Equilibrium constant parameters (empirical)
    # ln(K_eq) = a + b/T
    'a': CustomProperty(value=3.5, unit="-", symbol="a"),
    'b': CustomProperty(value=-1200, unit="K", symbol="b"),
}

rate_return: rRet = CustomProperty(value=0, unit="mol/m3.s", symbol="r1")

rate_args: rArgs = {
    'T': CustomProperty(value=0, unit="K", symbol="T"),
    # catalyst bulk density
    'rho_B': CustomProperty(value=0, unit="kg/m3", symbol="rho_B"),
}


def r1(X: Dict[str, X], args: rArgs, params: rParams) -> CustomProperty:
    """
    Heterogeneous catalytic esterification (Amberlyst-type model)

    Intrinsic rate:
        r' = k * (C_A * C_B - (C_C * C_D) / K_eq)

    Reactor-scale rate:
        r = rho_B * r'

    where:
        k = A * exp(-Ea / RT)
        ln(K_eq) = a + b/T

    Units:
        C_i : mol/m3
        r'  : mol/kgcat.s
        r   : mol/m3.s
    """

    R = 8.314  # J/mol.K

    # parameters
    A = params['A'].value
    Ea = params['Ea'].value
    a = params['a'].value
    b = params['b'].value

    T = args['T'].value
    rho_B = args['rho_B'].value

    # Arrhenius rate constant
    k = A * math.exp(-Ea / (R * T))

    # Equilibrium constant
    K_eq = math.exp(a + b / T)

    # concentrations
    c_acid = X['CH3COOH-l'].value
    c_meoh = X['CH3OH-l'].value
    c_meac = X['C3H6O2-l'].value
    c_h2o = X['H2O-l'].value

    # driving force
    forward = (c_acid ** X['CH3COOH-l'].order) * (c_meoh ** X['CH3OH-l'].order)
    reverse = (c_meac ** X['C3H6O2-l'].order) * (c_h2o ** X['H2O-l'].order)

    r_mass = k * (forward - reverse / K_eq)

    # convert to reactor-volume basis
    rExp = rho_B * r_mass

    return CustomProperty(
        name="r1",
        description="Catalytic esterification of acetic acid with methanol (thermodynamic + Arrhenius + bulk density)",
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
