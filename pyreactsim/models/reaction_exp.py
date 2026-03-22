# import libs
# annotations
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, TypeAlias, Callable, Awaitable
from pythermodb_settings.models import Component, CustomProperty
# locals

# NOTE: args
rArgs = Dict[str, CustomProperty]
# NOTE: parameters
rParams = Dict[str, CustomProperty]
# NOTE: return
rRet = Dict[str, CustomProperty]
# NOTE: state
X = Dict[str, float]
# ReactionRateExpression = Callable[[X, Params], r]

# SECTION: Irreversible Reaction Rate Expression


class IrreversibleReactionRateExpression(BaseModel):
    basis: Literal['concentration', 'pressure']
