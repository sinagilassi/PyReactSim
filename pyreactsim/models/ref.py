# import libs
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, TypeAlias

# SECTION: Batch Reactor Model
# NOTE: Reactor types
ReactorPhase = Literal['gas', 'liquid']
# NOTE: Isothermal non-isothermal models
HeatTransferMode = Literal['isothermal', 'non-isothermal']
# NOTE: Operation modes
OperationMode = Literal[
    'constant_volume',
    'constant_pressure',
    'variable_volume'
]
# NOTE: Gas Model
GasModel = Literal['ideal', 'real']


# SECTION: General Reference Models
class ReactorOptions(BaseModel):
    """
    Base class for reactor options.

    Attributes
    ----------
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
    phase: ReactorPhase = Field(
        ...,
        description="Phase of the batch reactor (gas or liquid)."
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
    reaction_enthalpy_mode: Optional[Literal['ideal_gas', 'liquid']] = Field(
        default="ideal_gas",
        description="Mode for reaction enthalpy calculation as ideal_gas or liquid."
    )
    gas_heat_capacity_source: Optional[Literal['model_inputs', 'model_source']] = Field(
        default="model_source",
        description="Source of gas heat capacity as model_inputs or model_source."
    )
    liquid_heat_capacity_source: Optional[Literal['model_inputs', 'model_source']] = Field(
        default="model_source",
        description="Source of liquid heat capacity as model_inputs or model_source."
    )
    liquid_density_source: Optional[Literal['model_inputs', 'model_source']] = Field(
        default="model_source",
        description="Source of liquid density as model_inputs or model_source."
    )
    ideal_gas_formation_enthalpy_source: Optional[Literal['model_inputs', 'model_source']] = Field(
        default="model_source",
        description="Source of gas formation enthalpy as model_inputs or model_source."
    )
    molecular_weight_source: Optional[Literal['model_inputs', 'model_source']] = Field(
        default="model_source",
        description="Source of molecular weight as model_inputs or model_source."
    )
