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
