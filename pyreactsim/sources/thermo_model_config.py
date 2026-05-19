# import libs
import logging
from typing import Callable, Tuple, Dict, List, Any
from pythermodb_settings.models import Component, ComponentKey, CustomProp
# locals
from ..utils.tools import config_components_property
from ..models.br import BatchReactorOptions
from ..models.cstr import CSTRReactorOptions
from ..models.pfr import PFRReactorOptions
from ..models.pbr import PBRReactorOptions
from ..models.heat import HeatTransferOptions

# NOTE: logger setup
logger = logging.getLogger(__name__)


class ThermoModelConfig:
    def __init__(
        self,
        components: List[Component],
        thermo_inputs: Dict[str, Any],
        reactor_options: BatchReactorOptions | CSTRReactorOptions | PFRReactorOptions | PBRReactorOptions,
        heat_transfer_options: HeatTransferOptions,
        component_refs: Dict[str, Any],
        component_key: ComponentKey,
    ):
        # NOTE: Set attributes
        self.components = components
        self.thermo_inputs = thermo_inputs
        self.reactor_options = reactor_options
        self.heat_transfer_options = heat_transfer_options
        self.component_refs = component_refs
        self.component_key = component_key

        # SECTION: component reference
        # ! component refs
        self.component_ids = component_refs['component_ids']
        self.component_formula_state = component_refs['component_formula_state']
        self.component_mapper = component_refs['component_mapper']

        # ! model inputs keys
        self.thermo_inputs_keys = list(self.thermo_inputs.keys())

    # SECTION: Criteria matching method
    def match_criteria(self, criteria: Dict[str, Dict[str, List[Any]]]) -> bool:
        all_block = criteria.get("all", {})
        any_block = criteria.get("any", {})
        not_block = criteria.get("not", {})

        def _get_attr_value(attr: str) -> Any:
            if hasattr(self.heat_transfer_options, attr):
                return getattr(self.heat_transfer_options, attr)
            if hasattr(self.reactor_options, attr):
                return getattr(self.reactor_options, attr)
            return None

        all_ok = all(
            _get_attr_value(attr) in allowed_values
            for attr, allowed_values in all_block.items()
        ) if all_block else True

        any_ok = any(
            _get_attr_value(attr) in allowed_values
            for attr, allowed_values in any_block.items()
        ) if any_block else True

        not_ok = all(
            _get_attr_value(attr) not in disallowed_values
            for attr, disallowed_values in not_block.items()
        ) if not_block else True

        return all_ok and any_ok and not_ok

    def _should_configure(
        self,
        prop_criteria: Dict[str, Dict[str, List[Any]]],
        phase_criteria: Dict[str, Dict[str, List[Any]]],
        heat_transfer_mode_criteria: Dict[str, Dict[str, List[Any]]],
    ) -> bool:
        if phase_criteria and not self.match_criteria(phase_criteria):
            return False
        if heat_transfer_mode_criteria and not self.match_criteria(heat_transfer_mode_criteria):
            return False
        if prop_criteria and not self.match_criteria(prop_criteria):
            return False
        return True

    # SECTION: Universal methods for property retrieval and configuration
    # NOTE: configure property values and component mapping based on criteria matching
    def _config_property(
        self,
        prop_name: str,
        unit_conversion_func: Callable[[float, str], float],
        expected_unit: str,
        prop_criteria: Dict[str, Dict[str, List[Any]]],
        phase_criteria: Dict[str, Dict[str, List[Any]]],
        heat_transfer_mode_criteria: Dict[str, Dict[str, List[Any]]],
        strict_unit_check: bool = True,
    ) -> Dict[str, float] | None:
        if not self._should_configure(prop_criteria, phase_criteria, heat_transfer_mode_criteria):
            return None

        # NOTE: if all checks pass, proceed to configure the property
        if prop_name not in self.thermo_inputs_keys:
            raise ValueError(
                f"{prop_name} must be provided in model_inputs for {prop_name} configuration."
            )

        # get property source
        prop_: Dict[str, CustomProp] = self.thermo_inputs[prop_name]

        # set property values and component mapping
        # >> component-wise property values
        prop_comp = {}

        # NOTE: prepare array and component mapping dict for property source
        # iterate through components
        for id in prop_.keys():
            # > value
            value = prop_[id].value
            # > unit
            unit = prop_[id].unit

            # >> check expected unit
            if strict_unit_check:
                if unit != expected_unit:
                    # convert to expected unit
                    value = unit_conversion_func(
                        value,
                        unit
                    )

            # add to component mapping
            prop_comp[id] = value

        return prop_comp

    # NOTE: configure property source dict based on criteria matching
    def _config_property_source(
        self,
        prop_name: str,
        unit_conversion_func: Callable[[float, str], float],
        expected_unit: str,
        prop_criteria: Dict[str, Dict[str, List[Any]]],
        phase_criteria: Dict[str, Dict[str, List[Any]]],
        heat_transfer_mode_criteria: Dict[str, Dict[str, List[Any]]],
        strict_unit_check: bool = True,
    ) -> Tuple[Dict[str, Dict[str, Any]], Any, Dict[str, float]] | None:
        if not self._should_configure(prop_criteria, phase_criteria, heat_transfer_mode_criteria):
            return None

        # NOTE: if all checks pass, proceed to configure the property
        if prop_name not in self.thermo_inputs_keys:
            raise ValueError(
                f"{prop_name} must be provided in model_inputs for {prop_name} configuration."
            )

        # get property source
        prop_: Dict[str, CustomProp] = self.thermo_inputs[prop_name]

        # init property source dict
        prop_src = {}

        # iterate through components
        for formula_state, component_id in zip(self.component_formula_state, self.component_ids):
            if formula_state in prop_.keys():
                # value
                value = prop_[formula_state].value
                # unit
                unit = prop_[formula_state].unit

                # add to property source dict with name-formula key
                prop_src[component_id] = {
                    "value": value,
                    "unit": unit,
                }
            else:
                raise ValueError(
                    f"{prop_name} value for component '{formula_state}' not found in model_inputs."
                )

        # NOTE: prepare array and component mapping dict for property source
        prop_values, prop_comp = config_components_property(
            component_ids=self.component_ids,
            prop_source=prop_src,
            unit_conversion_func=unit_conversion_func,
        )

        return prop_src, prop_values, prop_comp

    # NOTE: configure property constant based on criteria matching
    def _config_property_constant(
        self,
        prop_name: str,
        unit_conversion_func: Callable[[float, str], float],
        expected_unit: str,
        prop_criteria: Dict[str, Dict[str, List[Any]]],
        phase_criteria: Dict[str, Dict[str, List[Any]]],
        heat_transfer_mode_criteria: Dict[str, Dict[str, List[Any]]],
        strict_unit_check: bool = True,
    ) -> CustomProp | None:
        if not self._should_configure(prop_criteria, phase_criteria, heat_transfer_mode_criteria):
            return None

        # NOTE: if all checks pass, proceed to configure the property
        if prop_name not in self.thermo_inputs_keys:
            raise ValueError(
                f"{prop_name} must be provided in model_inputs for {prop_name} configuration."
            )

        # get property source
        prop_: CustomProp = self.thermo_inputs[prop_name]

        # check unit
        if strict_unit_check:
            if prop_.unit != expected_unit:
                # convert to expected unit
                value = unit_conversion_func(
                    prop_.value,
                    prop_.unit
                )
            else:
                value = prop_.value
        else:
            value = prop_.value

        # create property constant
        prop_constant = CustomProp(
            value=value,
            unit=expected_unit,
        )

        return prop_constant
