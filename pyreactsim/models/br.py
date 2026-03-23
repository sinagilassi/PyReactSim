# import libs
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, TypeAlias
from pythermodb_settings.models import Component, CustomProp

# SECTION: Batch Reactor Model
# NOTE: Reactor types
Phase = Literal['gas', 'liquid']
# NOTE: Isothermal non-isothermal models
HeatTransferMode = Literal['isothermal', 'non-isothermal']
# NOTE: Volume mode
VolumeMode = Literal['constant', 'variable']
# NOTE: Gas Model
GasModel = Literal['ideal', 'real']


class BatchReactorOptions(BaseModel):
    phase: Phase = Field(
        ...,
        description="Phase of the batch reactor (gas or liquid)."
    )
    heat_transfer_mode: HeatTransferMode = Field(
        ...,
        description="Heat transfer mode (isothermal or non-isothermal)."
    )
    volume_mode: VolumeMode = Field(
        ...,
        description="Volume mode (constant or variable)."
    )
    gas_model: Optional[GasModel] = Field(
        default=None,
        description="Gas model to use (required if phase is gas)."
    )
    reactor_volume: Optional[CustomProp] = Field(
        default=None,
        description="Volume of the reactor (required if volume mode is constant)."
    )
    jacket_temperature: Optional[CustomProp] = Field(
        default=None,
        description="Temperature of the jacket (required if heat transfer mode is non-isothermal)."
    )
    heat_transfer_coefficient: Optional[CustomProp] = Field(
        default=None,
        description="Heat transfer coefficient (required if heat transfer mode is non-isothermal)."
    )
    heat_transfer_area: Optional[CustomProp] = Field(
        default=None,
        description="Heat transfer area (required if heat transfer mode is non-isothermal)."
    )
