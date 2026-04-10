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
    """Base class for reactor options."""
    ideal_gas_formation_enthalpy_mode: Optional[Literal['model_inputs', 'model_source']] = Field(
        default="model_source",
        description="Source of gas formation enthalpy as model_inputs or model_source."
    )
    molecular_weight_mode: Optional[Literal['model_inputs', 'model_source']] = Field(
        default="model_source",
        description="Source of molecular weight as model_inputs or model_source."
    )
