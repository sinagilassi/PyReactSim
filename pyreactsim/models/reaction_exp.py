# import libs
# annotations
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, TypeAlias, Callable, Awaitable
from pythermodb_settings.models import Component
# locals

ReactionRateExpression = Callable[[PMap, AMap], RetMap | Awaitable[RetMap]]
