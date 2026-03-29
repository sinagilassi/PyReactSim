# import libs
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, TypeAlias
from pythermodb_settings.models import Component, CustomProp, Volume, Temperature
# locals
from .ref import HeatTransferMode, ReactorPhase, OperationMode, GasModel


class BatchReactorOptions(BaseModel):
    """Options for configuring the batch reactor model.

    Attributes
    ----------
    phase : ReactorPhase
        Phase of the batch reactor (gas or liquid).
    heat_transfer_mode : HeatTransferMode
        Heat transfer mode (isothermal or non-isothermal).
    operation_mode : OperationMode
        Operating condition of the reactor (constant volume or constant pressure).
    gas_model : GasModel
        Gas model to use (required if phase is gas).
    reactor_volume : Optional[CustomProp]
        Volume of the reactor (required if volume mode is constant).
    jacket_temperature : Optional[CustomProp]
        Temperature of the jacket (required if heat transfer mode is non-isothermal).
    heat_transfer_coefficient : Optional[CustomProp]
        Heat transfer coefficient (required if heat transfer mode is non-isothermal).
    heat_transfer_area : Optional[CustomProp]
        Heat transfer area (required if heat transfer mode is non-isothermal).
    heat_capacity_mode : Optional[Literal['constant', 'variable']]
        Heat capacity mode (constant or variable).
    """
    phase: ReactorPhase = Field(
        ...,
        description="Phase of the batch reactor (gas or liquid)."
    )
    heat_transfer_mode: HeatTransferMode = Field(
        ...,
        description="Heat transfer mode (isothermal or non-isothermal)."
    )
    operation_mode: OperationMode = Field(
        ...,
        description="Operating condition of the reactor (constant volume or constant pressure)."
    )
    gas_model: GasModel = Field(
        default='ideal',
        description="Gas model to use (required if phase is gas)."
    )
    reactor_volume: Optional[Volume] = Field(
        default=None,
        description="Volume of the reactor (required if volume mode is constant)."
    )
    jacket_temperature: Optional[Temperature] = Field(
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
    gas_heat_capacity_mode: Optional[Literal['constant', 'temperature-dependent', 'differential']] = Field(
        default='temperature-dependent',
        description="Heat capacity mode as constant, temperature-dependant, and differential."
    )
    liquid_heat_capacity_mode: Optional[Literal['constant', 'temperature-dependent', 'differential']] = Field(
        default='temperature-dependent',
        description="Heat capacity mode as constant, temperature-dependant."
    )
    liquid_density_mode: Optional[Literal['constant', 'temperature-dependent']] = Field(
        default=None,
        description="Density mode as constant, temperature-dependant."
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
