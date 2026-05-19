# import libs
import logging
from typing import Callable, Tuple, Dict, List, Any
from pythermodb_settings.models import Component, ComponentKey, CustomProp
# locals

from ..models.br import BatchReactorOptions
from ..models.cstr import CSTRReactorOptions
from ..models.pfr import PFRReactorOptions
from ..models.pbr import PBRReactorOptions
from ..models.heat import HeatTransferOptions

# NOTE: logger setup
logger = logging.getLogger(__name__)


class ThermoSourceConfig:
    def __init__(
        self,
        components: List[Component],
        custom_inputs: Dict[str, Any] | None,
        reactor_options: BatchReactorOptions | CSTRReactorOptions | PFRReactorOptions | PBRReactorOptions,
        heat_transfer_options: HeatTransferOptions,
        component_refs: Dict[str, Any],
        component_key: ComponentKey,
    ):
        # NOTE: Set attributes
        self.components = components
        self.custom_inputs = custom_inputs
        self.reactor_options = reactor_options
        self.heat_transfer_options = heat_transfer_options
        self.component_refs = component_refs
        self.component_key = component_key

        # SECTION: component reference
        # ! component refs
        self.component_ids = component_refs['component_ids']
        self.component_formula_state = component_refs['component_formula_state']
        self.component_mapper = component_refs['component_mapper']

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
