# import libs
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, TypeAlias
from pythermodb_settings.models import Component, CustomProp, Volume, Temperature
# locals
from .ref import ReactorPhase, OperationMode, GasModel


class CSTRReactorOptions(BaseModel):
    phase: ReactorPhase = Field(..., description="Phase of the CSTR reactor.")
    pressure_mode: Literal['constant', 'variable'] = Field(
        ...,
        description="Pressure mode for the CSTR reactor (constant or variable)."
    )
    Volume_mode: Literal['constant', 'variable'] = Field(
        ...,
        description="Volume mode for the CSTR reactor (constant or variable)."
    )
    gas_model: GasModel = Field(
        default='ideal',
        description="Gas model to use (required if gas is involved)."
    )
    gas_heat_capacity_mode: Optional[Literal['constant', 'temperature-dependent', 'differential']] = Field(
        default='temperature-dependent',
        description="Gas heat capacity mode."
    )
    liquid_heat_capacity_mode: Optional[Literal['constant', 'temperature-dependent', 'differential']] = Field(
        default='temperature-dependent',
        description="Liquid heat capacity mode."
    )
    liquid_density_mode: Optional[Literal['constant', 'temperature-dependent']] = Field(
        default=None,
        description="Liquid density mode."
    )
