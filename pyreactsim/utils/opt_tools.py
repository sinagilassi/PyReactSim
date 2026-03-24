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

# SECTION:
