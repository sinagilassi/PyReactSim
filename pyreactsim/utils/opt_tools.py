# import libs
import logging
from typing import Any, Dict, List, Optional, Tuple, cast
from pythermodb_settings.models import Component, ComponentKey, CustomProp, Pressure, Temperature, CustomProperty
from pythermodb_settings.utils import set_component_id
from pyreactlab_core.models.reaction import Reaction
# locals
from ..models.rate_exp import ReactionRateExpression
from ..models.rate_exp_refs import rArgs, rParams, rRet, rXs

# NOTE: logger setup
logger = logging.getLogger(__name__)


def _create_id(formula: str, state: str) -> str:
    """
    Create a component id based on the formula and state of the component.
    """
    return f"{formula}-{state}"


def retrieve_reaction_stoichiometry(
        components: List[Component],
        reaction_rates: ReactionRateExpression,
):
    """
    Retrieve details from a ReactionRateExpression object
    """
    # SECTION: Analyze reaction rate expression
    # NOTE: name
    reaction_name = reaction_rates.name

    # NOTE: reaction
    reaction = reaction_rates.reaction
    # >> reaction coefficients
    nu = reaction.reaction_stoichiometry_dict
    nu_keys = list(nu.keys())
    # >> nu comp
    nu_comp = {}
    nu_vector = []

    # iterate through components and set ids
    for comp in components:
        comp_id = set_component_id(comp, 'Name-Formula')
        if comp_id in nu_keys:
            nu_comp[comp_id] = nu.get(comp_id, 0)
            nu_vector.append(nu.get(comp_id, 0))
        else:
            logger.warning(
                f"Component {comp_id} not found in reaction stoichiometry. Setting nu to 0."
            )
            nu_comp[comp_id] = 0
            nu_vector.append(0)

    return nu_comp, nu_vector
