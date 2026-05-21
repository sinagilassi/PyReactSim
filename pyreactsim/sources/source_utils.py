# import libs
import logging
import numpy as np
from typing import Dict, Any, Literal, Tuple, List, Optional, TypeAlias
from pythermodb_settings.models import CustomProp
from pyThermoLinkDB.models.component_models import ComponentEquationSource
# locals
from .thermo_custom_inputs import ThermoCustomInputs
from .thermo_model_source import ThermoModelSource

# NOTE: logger setup
logger = logging.getLogger(__name__)


# NOTE: Models
# ! data source assigner
DataSourceAssignerResult: TypeAlias = Tuple[
    np.ndarray,
    Dict[str, float],
    Dict[str, Any]
]
# ! equation source assigner
EquationSourceAssignerResult: TypeAlias = Tuple[
    np.ndarray,
    Dict[str, float],
    Dict[str, ComponentEquationSource]
]
# ! property constant assigner
PropertyConstantAssignerResult: TypeAlias = Optional[CustomProp]
# ! properties constant assigner
PropertiesConstantAssignerResult: TypeAlias = Optional[Dict[str, Dict[str, Any]]]


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
    def data_source_assigner(
            self,
            symbol: str,
    ) -> DataSourceAssignerResult:
        """
        Assigns property values based on the source, checking custom inputs first, then model source. Returns the property values and component-specific values if available.

        Parameters
        ----------
        symbol : str
            The symbol representing the property to assign (e.g., 'EnFo_IG_298').

        Returns
        -------
        DataSourceAssignerResult
            A tuple containing the property values (if any) and component-specific values (if any).

        Notes
        -----
        - The method first checks for the property in the custom inputs, and if not found or empty, it checks the model source.
        - If the property is not found in either source, it returns empty values.
        - The component-specific values are expected to be in a dictionary format, where the keys are component names and the values are the corresponding property values for those components.
        - The properties which are usually constant (e.g., standard enthalpy of formation at 298 K) are extracted using this method.
        """
        # NOTE: create variable names
        val = f"{symbol}"
        val_comp = f"{symbol}_comp"
        val_src = f"{symbol}_src"

        # >>> init res
        prop_value = np.array([])
        prop_value_comp = {}
        prop_src = {}

        # NOTE: assign property values based on source
        # ! check attribute in custom inputs
        if hasattr(self.thermo_custom_inputs, symbol):
            # > assign values from custom inputs
            prop_value = getattr(self.thermo_custom_inputs, val)
            prop_value_comp = getattr(self.thermo_custom_inputs, val_comp)
            prop_src = getattr(self.thermo_custom_inputs, val_src)

            # check empty values
            if (
                prop_value is not None
                and prop_value.size > 0
                and isinstance(prop_value_comp, dict) and
                prop_value_comp
            ):
                return prop_value, prop_value_comp, prop_src

        # ! check attribute in model source
        if hasattr(self.thermo_model_source, symbol):
            # > assign values from model source
            prop_value = getattr(self.thermo_model_source, val)
            prop_value_comp = getattr(self.thermo_model_source, val_comp)
            prop_src = getattr(self.thermo_model_source, val_src)

            # check empty values
            if (
                prop_value is not None and
                prop_value.size > 0 and
                isinstance(prop_value_comp, dict) and
                prop_value_comp
            ):
                return prop_value, prop_value_comp, prop_src

        # ! if no source found, return empty values
        return prop_value, prop_value_comp, prop_src

    # SECTION: Equation assigner method
    def equation_source_assigner(
        self,
        symbol: str,
    ) -> EquationSourceAssignerResult:
        """
        Assigns property equations based on the source, checking model source first, then custom inputs. Returns the property values, component-specific values, and equation sources if available.

        Parameters
        ----------
        symbol : str
            The symbol representing the property to assign (e.g., 'Cp_IG').

        Returns
        -------
        EquationSourceAssignerResult
            A tuple containing the property values (if any), component-specific values (if any), and equation sources (if any).

        Notes
        -----
        - The method first checks for the property equations in the model source, and if not found or empty, it checks the custom inputs for constant values.
        - If the property equations are not found in the model source, it will attempt to assign constant values from the custom inputs.
        - If neither source provides the property equations or values, it returns empty values and sources.
        - The component-specific values are expected to be in a dictionary format, where the keys are component names and the values are the corresponding property values for those components.
        - The properties which are typically represented by equations (e.g., ideal gas heat capacity) are extracted using this method.
        """
        # NOTE: create variable name
        val = f"{symbol}"
        val_comp = f"{symbol}_comp"
        val_src = f"{symbol}_src"

        # >>> init res
        _values = np.array([])
        _values_comp = {}
        _source: Dict[str, ComponentEquationSource] = {}

        # NOTE: assign property equations based on source
        # ! check attribute in model source
        if hasattr(self.thermo_model_source, val_src):
            # > assign equations from model source (a dictionary of component to equation source)
            eq_src = getattr(self.thermo_model_source, val_src)

            # check empty values
            if isinstance(eq_src, dict) and eq_src:
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
                _values.size > 0 and
                isinstance(_values_comp, dict) and
                _values_comp
            ):
                return _values, _values_comp, _source

        # ! if no source found, return empty values
        # log
        logger.warning(
            f"No source found for {symbol}. Returning empty values and source."
        )

        # res
        return _values, _values_comp, _source

    # SECTION: data source assigner method
    def property_constant_assigner(
        self,
        symbol: str,
    ) -> PropertyConstantAssignerResult:
        """
        Assigns property source information based on the source, checking custom inputs first, then model source. Returns the property source information if available. The symbol is used to directly access the corresponding attribute in the custom inputs or model source.

        Parameters
        ----------
        symbol : str
            The symbol representing the property to assign (e.g., 'EnFo_IG_298').

        Returns
        -------
        PropertyConstantAssignerResult
            A dictionary containing the property source information if available, or None.
        Notes
        -----
        - The method first checks for the property source information in the custom inputs, and if not found or empty, it checks the model source.
        - If the property source information is not found in either source, it returns an empty dictionary.
        - The property source information is expected to be in a dictionary format, where the keys are component names and the values are the corresponding source information for those components.
        - The properties which are usually constant (e.g., standard enthalpy of formation at 298 K) are extracted using this method.
        """
        # >>> init res
        prop_symbol: CustomProp | None = None

        # NOTE: assign property source information based on source
        # ! check attribute in custom inputs
        if hasattr(self.thermo_custom_inputs, symbol):
            # > assign values from custom inputs
            prop_symbol = getattr(self.thermo_custom_inputs, symbol)

            # check empty values
            if prop_symbol and isinstance(prop_symbol, CustomProp):
                return prop_symbol

        # ! check attribute in model source
        if hasattr(self.thermo_model_source, symbol):
            # > assign values from model source
            prop_symbol = getattr(self.thermo_model_source, symbol)

            # check empty values
            if prop_symbol and isinstance(prop_symbol, CustomProp):
                return prop_symbol

        # ! if no source found, return empty values
        return prop_symbol

    def properties_constant_assigner(
            self,
            symbol: str
    ) -> PropertiesConstantAssignerResult:
        """
        Assigns component-specific property source information based on the source, checking custom inputs first, then model source. Returns a dictionary containing the component-specific property source information if available.

        Parameters
        ----------
        symbol : str
            The symbol representing the property to assign (e.g., 'EnFo_IG_298_comp').

        Returns
        -------
        PropertiesConstantAssignerResult
            A dictionary containing the component-specific property source information if available, or None.
        """
        # >>> init res
        prop_symbol: Dict[str, Dict[str, Any]] | None = None

        # NOTE: assign component-specific property source information based on source
        # ! check attribute in custom inputs
        if hasattr(self.thermo_custom_inputs, symbol):
            # > assign values from custom inputs
            prop_symbol = getattr(self.thermo_custom_inputs, symbol)

            # check empty values
            if prop_symbol and isinstance(prop_symbol, dict):
                return prop_symbol

        # ! check attribute in model source
        if hasattr(self.thermo_model_source, symbol):
            # > assign values from model source
            prop_symbol = getattr(self.thermo_model_source, symbol)

            # check empty values
            if prop_symbol and isinstance(prop_symbol, dict):
                return prop_symbol

        # ! if no source found, return empty values
        return prop_symbol

    # SECTION: Intelligent property assigner method
    def assigner(
            self,
            symbol: str,
            mode: Literal[
                'data',
                'equation',
                'constant',
                'constants',
            ]
    ) -> DataSourceAssignerResult | EquationSourceAssignerResult | PropertyConstantAssignerResult | PropertiesConstantAssignerResult:
        """
        Assigns property values, equations, or source information based on the specified mode, checking custom inputs first, then model source. Returns the assigned values, equations, or source information based on the mode.

        Parameters
        ----------
        symbol : str
            The symbol representing the property to assign (e.g., 'EnFo_IG_298').
        mode : Literal['data', 'equation', 'constant', 'constants']
            The mode of assignment, which determines whether to assign property values ('data'), equations ('equation'), a single property source ('constant'), or component-specific property sources ('constants').

        Returns
        -------
        DataSourceAssignerResult | EquationSourceAssignerResult | PropertyConstantAssignerResult | PropertiesConstantAssignerResult
            The assigned property values, equations, or source information based on the specified mode.
        """
        if mode == 'data':
            return self.data_source_assigner(symbol)
        elif mode == 'equation':
            return self.equation_source_assigner(symbol)
        elif mode == 'constant':
            return self.property_constant_assigner(symbol)
        elif mode == 'constants':
            return self.properties_constant_assigner(symbol)
        else:
            raise ValueError(
                f"Invalid mode: {mode}. Must be one of 'data', 'equation', 'property_constant', or 'properties_constant'.")
