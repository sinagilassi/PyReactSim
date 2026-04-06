# import libs
from pydantic import BaseModel, Field
from typing import Any, Dict, Literal, Optional
# locals
from .ref import GasModel, ReactorPhase


class PFRReactorOptions(BaseModel):
    """Options for configuring the PFR reactor model."""

    phase: ReactorPhase = Field(
        ...,
        description="Phase of the PFR reactor (gas or liquid)."
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


class PFRReactorResult(BaseModel):
    """Container for PFR reactor simulation outputs."""

    volume: Any = Field(
        ...,
        description="Reactor-volume coordinate points returned by the ODE solver."
    )
    state: Any = Field(
        ...,
        description="State matrix returned by the ODE solver (n_states, n_points)."
    )
    success: bool = Field(
        ...,
        description="Whether the ODE solver finished successfully."
    )
    message: str = Field(
        default="",
        description="Solver status message."
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "volume": self.volume,
            "state": self.state,
            "success": self.success,
            "message": self.message,
        }

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_dict().get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]
