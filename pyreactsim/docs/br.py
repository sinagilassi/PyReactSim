# import libs
import logging
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, TypeAlias
from pythermodb_settings.models import Component
from pyThermoLinkDB.models import ModelSource
# locals
from ..models.br import BatchReactorOptions

# NOTE: set logger
logger = logging.getLogger(__name__)


def batch_react(
        components: List[Component],
        model_inputs: Dict[str, Any],
        reactor_inputs: BatchReactorOptions,
        model_source: ModelSource,
        **kwargs
):
    pass
