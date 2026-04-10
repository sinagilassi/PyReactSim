# import libs
from pydantic import BaseModel, Field
from typing import Any, Dict, Literal, Optional
# locals
from .ref import GasModel, ReactorPhase, ReactorOptions


class PFRReactorOptions(ReactorOptions):
    """
    Options for configuring the PFR reactor model.


    Attributes
    ----------
    operation_mode : Literal['constant_pressure', 'constant_volume']
        Operating condition of the reactor (constant volume or constant pressure).
    pressure_mode : Optional[Literal["constant", "shortcut", "state_variable"]]
        Pressure mode as constant, shortcut, and state_variable. The shortcut uses ideal-gas formulation and state_variable considers pressure as a variable computes the pressure drop along the reactor.
    phase : ReactorPhase
        Phase of the reactor (gas or liquid).
    gas_model : GasModel
        Gas model to use (required if phase is gas).
    gas_heat_capacity_mode : Optional[Literal['constant', 'temperature-dependent', 'differential']]
        Gas heat capacity mode as constant, temperature-dependent, and differential.
    liquid_heat_capacity_mode : Optional[Literal['constant', 'temperature-dependent', 'differential']]
        Liquid heat capacity mode as constant, temperature-dependent, and differential.
    liquid_density_mode : Optional[Literal['constant', 'temperature-dependent']]
        Liquid density mode as constant or temperature-dependent.
    ideal_gas_formation_enthalpy_mode : Optional[Literal['model_inputs', 'model_source']]
        Source of gas formation enthalpy as model_inputs or model_source.
    molecular_weight_mode : Optional[Literal['model_inputs', 'model_source']]
        Source of molecular weight as model_inputs or model_source.
    """
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
