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
        description="Gas heat capacity mode as constant, temperature-dependent, differential, and mixture."
    )
    liquid_heat_capacity_mode: Optional[Literal["constant", "temperature-dependent", "differential"]] = Field(
        default="temperature-dependent",
        description="Liquid heat capacity mode as constant, temperature-dependent, differential, and mixture."
    )
    liquid_density_mode: Optional[Literal["constant", "temperature-dependent", "mixture"]] = Field(
        default=None,
        description="Density mode as constant, temperature-dependent, and mixture."
    )
    reaction_enthalpy_mode: Optional[Literal['ideal_gas', 'liquid', 'reaction']] = Field(
        default="ideal_gas",
        description="Mode for reaction enthalpy calculation as ideal_gas, liquid, and reaction."
    )
    use_gas_mixture_total_heat_capacity: bool = Field(
        default=False,
        description=(
            "If True, use gas mixture total heat capacity directly (J/K). "
            "If False, calculate it from species contributions: "
            "Cp_IG_MIX_TOTAL = sum(n_i * Cp_i), where Cp_i is in J/mol.K and n_i is in mol."
        )
    )
    use_liquid_mixture_volumetric_heat_capacity: bool = Field(
        default=False,
        description=(
            "If True, use liquid mixture volumetric heat capacity directly (J/m3/K). "
            "If False, calculate it from species contributions: "
            "Cp_LIQ_MIX_VOL = sum(C_i * Cp_i), where Cp_i is in J/mol.K and C_i is in mol/m3."
        )
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
    reaction_enthalpy_source: Optional[Literal['model_inputs', 'model_source']] = Field(
        default="model_source",
        description="Source of reaction enthalpy as model_inputs or model_source."
    )
    gas_mixture_total_heat_capacity_source: Optional[Literal['model_inputs', 'model_source']] = Field(
        default="model_inputs",
        description="Source of gas mixture total heat capacity as model_inputs or model_source."
    )
    liquid_mixture_volumetric_heat_capacity_source: Optional[Literal['model_inputs', 'model_source']] = Field(
        default="model_inputs",
        description="Source of liquid mixture volumetric heat capacity as model_inputs or model_source."
    )
