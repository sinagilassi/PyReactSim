# import libs
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, TypeAlias
from pythermodb_settings.models import Component, CustomProp, Volume, Temperature

# SECTION: Heat exchange models


class HeatExchanger(BaseModel):
    """Model for heat exchanger in the reactor.

    Attributes
    ----------
    heat_transfer_coefficient : CustomProp
        Heat transfer coefficient (required if mode is non-isothermal).
    heat_transfer_area : CustomProp
        Heat transfer area (required if mode is non-isothermal).
    jacket_temperature : Temperature
        Temperature of the jacket (required if mode is non-isothermal).
    """
    heat_transfer_coefficient: Optional[CustomProp] = Field(
        default=None,
        description="Heat transfer coefficient (required if mode is non-isothermal)."
    )
    heat_transfer_area: Optional[CustomProp] = Field(
        default=None,
        description="Heat transfer area (required if mode is non-isothermal)."
    )
    jacket_temperature: Optional[CustomProp] = Field(
        default=None,
        description="Temperature of the jacket (required if mode is non-isothermal)."
    )
