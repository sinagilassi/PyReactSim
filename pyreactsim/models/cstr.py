# import libs
from pydantic import BaseModel, Field
from typing import Any, Dict, Literal, Optional
# locals
from .ref import GasModel, ReactorPhase


class CSTRReactorOptions(BaseModel):
    """Options for configuring the CSTR reactor model."""

    phase: ReactorPhase = Field(
        ...,
        description="Phase of the CSTR reactor (gas or liquid)."
    )
    operation_mode: Literal[
        'constant_pressure', 'constant_volume'
    ] = Field(
        ...,
        description="Operating condition of the reactor (constant volume or constant pressure)."
    )
    holdup_volume_mode: Literal["fixed", "dynamic"] = Field(
        ...,
        description="Holdup volume mode as fixed or dynamic (required if operation_mode is constant pressure)."
    )
    outlet_flow_mode: Literal["calculated", "fixed"] = Field(
        ...,
        description="Outlet flow mode as calculated or fixed (optional, default is calculated)."
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


class CSTRReactorResult(BaseModel):
    """Container for CSTR reactor simulation outputs."""

    time: Any = Field(
        ...,
        description="Time points returned by the ODE solver."
    )
    state: Any = Field(
        ...,
        description="State matrix returned by the ODE solver (n_states, n_time)."
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
            "time": self.time,
            "state": self.state,
            "success": self.success,
            "message": self.message,
        }

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_dict().get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]
