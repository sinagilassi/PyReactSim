# Import libs
import logging
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, TypeAlias, Callable
from pythermodb_settings.models import Component, CustomProperty
from pyreactlab_core.models.reaction import Reaction
# locals
from ..models.reaction_exp import rArgs, rParams, rRet, X, Xi, ReactionRateExpression

# NOTE: logger setup
logger = logging.getLogger(__name__)


class ReactionRate:
    def __init__(
        self,
        basis: Literal['concentration', 'pressure'],
        components: List[Component],
        reaction: Reaction,
        eq: Callable[[Xi, rArgs, rParams], rRet]
    ):
        self.basis = basis
        self.components = components
        self.reaction = reaction
        self.eq = eq

    def calc(self, xi: Dict[str, CustomProperty]):
        pass
