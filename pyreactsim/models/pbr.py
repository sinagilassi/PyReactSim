# import libs
from pydantic import BaseModel, Field
from typing import Any, Dict, Literal, Optional
# locals
from .ref import GasModel, ReactorPhase


class PBRReactorOptions(BaseModel):
    """Options for configuring the PBR reactor model."""

    phase: ReactorPhase = Field(
        ...,
        description="Phase of the PBR reactor (gas or liquid)."
    )
    operation_mode: Literal[
        'constant_pressure', 'constant_volume'
    ] = Field(
        ...,
        description="Operating condition of the reactor (constant volume or constant pressure)."
    )
    pressure_mode: Optional[Literal["constant", "shortcut", "state_variable"]] = Field(
        default="constant",
        description="Pressure mode as constant, shortcut, and state_variable. The shortcut uses ideal-gas formulation and state_variable considers pressure as a variable computes the pressure drop along the reactor."
    )
    gas_model: GasModel = Field(
        default="ideal",
        description="Gas model to use (required if phase is gas)."
    )
    gas_heat_capacity_mode: Optional[Literal["constant", "temperature-dependent", "differential"]] = Field(
        default="temperature-dependent",
        description="Gas heat capacity mode as constant, temperature-dependent, and differential."
    )
    liquid_heat_capacity_mode: Optional[Literal["constant", "temperature-dependent", "differential"]] = Field(
        default="temperature-dependent",
        description="Liquid heat capacity mode as constant, temperature-dependent, and differential."
    )
    liquid_density_mode: Optional[Literal["constant", "temperature-dependent"]] = Field(
        default=None,
        description="Density mode as constant or temperature-dependent."
    )
