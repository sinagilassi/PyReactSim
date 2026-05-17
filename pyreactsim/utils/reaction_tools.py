# import libs
import logging
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union, TypeAlias
from pythermodb_settings.models import Component, ComponentKey
from pyreactlab_core.models.reaction import Reaction
from pyreactlab_core import build_rxns_stoichiometry
from pyreactsim_core.models import ReactionRateExpression
# locals
# from ..models.rate_exp import ReactionRateExpression
from ..models import ReactorPhase

# NOTE: set logger
logger = logging.getLogger(__name__)


# SECTION: Stoichiometry matrix generator
def stoichiometry_mat(
        reactions: List[Reaction],
        components: List[Component],
        component_key: ComponentKey
):
    """
    Generate the stoichiometry matrix for a given list of reactions and components.

    Parameters
    ----------
    reactions : List[Reaction]
        A list of Reaction objects representing the chemical reactions.
    components : List[Component]
        A list of Component objects representing the chemical components involved in the reactions.
    component_key : ComponentKey
        A ComponentKey object representing the key to be used for the components in the reactions.

    Returns
    -------
    np.ndarray
        A stoichiometry matrix where rows correspond to reactions and columns correspond to components.
    """
    try:
        res_src = build_rxns_stoichiometry(
            reactions,
            components,
            component_key
        )

        # >> check
        if res_src is None:
            raise ValueError(
                "Failed to build stoichiometry matrix. Check reactions and components.")

        res = res_src['matrix']
        if isinstance(res, list):
            res = np.array(res)
        else:
            raise ValueError(
                "Unexpected format for stoichiometry matrix. Expected a list or numpy array.")

        return res
    except Exception as e:
        logger.error(f"Error in generating stoichiometry matrix: {e}")
        raise


# SECTION: stoichiometry matrix based on component key
def stoichiometry_mat_key(
        reactions: List[Reaction],
        component_key: ComponentKey
) -> List[Dict[str, float]]:
    """
    Generate the stoichiometry matrix for a given list of reactions and components based on a specified component key.

    Parameters
    ----------
    reactions : List[Reaction]
        A list of Reaction objects representing the chemical reactions.
    component_key : ComponentKey
        A ComponentKey object representing the key to be used for the components in the reactions.

    Returns
    -------
    List[Dict[str, float]]
        A list of dictionaries where each dictionary represents the stoichiometry of a reaction with component names as keys and their corresponding stoichiometric coefficients as values.
    """
    try:
        # NOTE: res
        res: List[Dict[str, float]] = []

        # iterate over reactions
        for reaction in reactions:
            # >> stoichiometry
            stoichiometry_: Dict[str, float] | None = reaction.reaction_stoichiometry_source.get(
                component_key,
                None
            )

            # >> check
            if stoichiometry_ is None:
                raise ValueError(
                    f"Stoichiometry not found for reaction {reaction.name} with component key {component_key}"
                )

            # add
            res.append(stoichiometry_)

        # res
        return res
    except Exception as e:
        logger.error(f"Error in generating stoichiometry matrix: {e}")
        raise


# SECTION: Calculate residence time (space time)
def calc_residence_time(
        volume: float,
        volumetric_flow_rate: float
) -> float:
    """
    Calculate the residence time (space time) for a reactor given its volume and volumetric flow rate.

    Parameters
    ----------
    volume : float
        The volume of the reactor (in cubic meters).
    volumetric_flow_rate : float
        The volumetric flow rate of the feed (in cubic meters per second).

    Returns
    -------
    float
        The residence time (space time) in seconds.
    """
    try:
        if volumetric_flow_rate <= 0:
            raise ValueError("Volumetric flow rate must be greater than zero.")
        residence_time = volume / volumetric_flow_rate
        return residence_time
    except Exception as e:
        logger.error(f"Error in calculating residence time: {e}")
        raise

# NOTE: Arrhenius equation


def arrhenius_equation(
        k_ref: float,
        Ea: float,
        T: float,
        T_ref: float,
        R: float = 8.314
):
    """
    Calculate the rate constant using the Arrhenius equation.

    Parameters
    ----------
    k_ref : float
        The reference rate constant at the reference temperature (in appropriate units).
    Ea : float
        The activation energy (in J/mol).
    T : float
        The temperature at which to calculate the rate constant (in K).
    T_ref : float
        The reference temperature (in K).
    R : float, optional
        The universal gas constant (default is 8.314 J/mol.K).

    Returns
    -------
    float
        The calculated rate constant at temperature T.
    """
    try:
        return k_ref * np.exp((-Ea / R) * (1/T - 1/T_ref))
    except Exception as e:
        logger.error(
            f"Error in calculating rate constant using Arrhenius equation: {e}")
        raise
