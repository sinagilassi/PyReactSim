# import libs
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, TypeAlias
# locals
from .ref import OperationMode, ReactorOptions


class BatchReactorOptions(ReactorOptions):
    """
    Options for configuring the batch reactor model.

    Attributes
    ----------
    modeling_type : Literal['physical', 'scale']
        Modeling type as physical or scale. The physical model solves the ODE system in physical units, while the scale model solves an equivalent scaled state vector.
    operation_mode : OperationMode
        Operating condition of the reactor (constant volume or constant pressure).
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
    ideal_gas_formation_enthalpy_source : Optional[Literal['model_inputs', 'model_source']]
        Source of gas formation enthalpy as model_inputs or model_source.
    molecular_weight_source : Optional[Literal['model_inputs', 'model_source']]
        Source of molecular weight as model_inputs or model_source.
    """
    modeling_type: Literal['physical', 'scale'] = Field(
        default="physical",
        description="Modeling type as physical or scale. The physical model solves the ODE system in physical units, while the scale model solves an equivalent scaled state vector."
    )
    operation_mode: OperationMode = Field(
        ...,
        description="Operating condition of the reactor (constant volume or constant pressure)."
    )


class BatchReactorResult(BaseModel):
    """Container for batch reactor simulation outputs."""

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
        """Return a plain dictionary representation."""
        return {
            "time": self.time,
            "state": self.state,
            "success": self.success,
            "message": self.message,
        }

    def get(self, key: str, default: Any = None) -> Any:
        """Dictionary-like accessor for backward compatibility."""
        return self.to_dict().get(key, default)

    def __getitem__(self, key: str) -> Any:
        """Support bracket access such as result['time']."""
        return self.to_dict()[key]
