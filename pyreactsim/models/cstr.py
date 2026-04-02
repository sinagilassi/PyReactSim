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
    case: Literal[1, 2, 3, 4, 5, 6, 7] = Field(
        ...,
        description="Gas-phase CSTR case number based on predefined mass/heat-balance closures."
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

    @property
    def is_isothermal(self) -> bool:
        return self.case in (1, 2, 3, 4)

    @property
    def is_non_isothermal(self) -> bool:
        return not self.is_isothermal

    @property
    def is_constant_pressure(self) -> bool:
        return self.case in (1, 3, 4, 5, 7)

    @property
    def is_variable_pressure(self) -> bool:
        return not self.is_constant_pressure

    @property
    def is_constant_volume(self) -> bool:
        return self.case in (2, 6)

    @property
    def is_variable_volume(self) -> bool:
        return self.case in (4, 7)

    @property
    def uses_energy_balance(self) -> bool:
        return self.is_non_isothermal


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
