# import libs
import logging
import numpy as np
from typing import Dict, Any, Literal, Tuple, List, Optional
from pyThermoLinkDB.models.component_models import ComponentEquationSource
# locals
from .thermo_custom_inputs import ThermoCustomInputs
from .thermo_model_source import ThermoModelSource

# NOTE: logger setup
logger = logging.getLogger(__name__)


class SourceUtils:

    def __init__(
            self,
            thermo_custom_inputs: ThermoCustomInputs,
            thermo_model_source: ThermoModelSource,
    ):
        # set
        self.thermo_custom_inputs = thermo_custom_inputs
        self.thermo_model_source = thermo_model_source

    # SECTION: Data assigner method
    def data_assigner(
            self,
            symbol: str,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        # NOTE: create variable names
        val = f"{symbol}"
        val_comp = f"{symbol}_comp"

        # >>> init res
        prop_value = np.array([])
        prop_value_comp = {}

        # NOTE: assign property values based on source
        # ! check attribute in custom inputs
        if hasattr(self.thermo_custom_inputs, symbol):
            # > assign values from custom inputs
            prop_value = getattr(self.thermo_custom_inputs, val)
            prop_value_comp = getattr(self.thermo_custom_inputs, val_comp)

            # check empty values
            if (
                prop_value is not None and
                prop_value_comp is not None
            ):
                return prop_value, prop_value_comp

        # ! check attribute in model source
        if hasattr(self.thermo_model_source, symbol):
            # > assign values from model source
            prop_value = getattr(self.thermo_model_source, val)
            prop_value_comp = getattr(self.thermo_model_source, val_comp)

            # check empty values
            if (
                prop_value is not None and
                prop_value_comp is not None
            ):
                return prop_value, prop_value_comp

        # ! if no source found, return empty values
        return prop_value, prop_value_comp

    # SECTION: Equation assigner method
    def equation_source_assigner(
        self,
        symbol: str,
    ) -> Optional[
        Tuple[
            np.ndarray,
            Dict[str, float],
            Dict[str, ComponentEquationSource]
        ]
    ]:
        # NOTE: create variable name
        val = f"{symbol}"
        val_comp = f"{symbol}_comp"
        val_src = f"{symbol}_src"

        # >>> init res
        _values = np.array([])
        _values_comp = {}
        _source = {}

        # NOTE: assign property equations based on source
        # ! check attribute in model source
        if hasattr(self.thermo_model_source, val_src):
            # > assign equations from model source
            eq_src = getattr(self.thermo_model_source, val_src)

            # check empty values
            if eq_src is not None:
                return _values, _values_comp, eq_src

        # NOTE: assign property values based on source
        # ! check attribute in custom inputs (constant values only, no equations)
        if hasattr(self.thermo_custom_inputs, symbol):
            # > assign values from custom inputs
            _values = getattr(self.thermo_custom_inputs, val)
            _values_comp = getattr(self.thermo_custom_inputs, val_comp)

            # check empty values
            if (
                _values is not None and
                _values_comp is not None
            ):
                return _values, _values_comp, _source

        # ! if no source found, return empty values
        # log
        logger.warning(
            f"No source found for {symbol}. Returning empty values and source."
        )

        # res
        return _values, _values_comp, _source
