# import libs
import logging
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union, TypeAlias
from pythermodb_settings.models import Component, ComponentKey
from pyreactlab_core.models.reaction import Reaction
from pyreactlab_core import build_rxns_stoichiometry
# locals
from ..models.rate_exp import ReactionRateExpression
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


# SECTION: State key checker
