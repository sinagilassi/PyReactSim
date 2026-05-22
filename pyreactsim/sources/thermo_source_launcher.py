# import libs
import logging
import numpy as np
from typing import Any, Dict, Tuple, List, cast
# locals
from .thermo_custom_inputs import ThermoCustomInputs
from .thermo_model_source import ThermoModelSource
from ..models.br import BatchReactorOptions
from ..models.cstr import CSTRReactorOptions
from ..models.pfr import PFRReactorOptions
from ..models.pbr import PBRReactorOptions
from .thermo_config import CUSTOM_INPUTS_ATTR_CONFIG, MODEL_SOURCE_ATTR_CONFIG, AVAILABLE_VARIABLES
from .thermo_property_fields import ThermoPropertyFields
from .source_utils import SourceUtils

# NOTE: logger setup
logger = logging.getLogger(__name__)


class ThermoSourceLauncher(ThermoPropertyFields, SourceUtils):
    """
    Class responsible for launching the thermo source construction process based on the provided configuration.
    """

    def __init__(
            self,
            thermo_model_source: ThermoModelSource,
            thermo_custom_inputs: ThermoCustomInputs,
            reactor_options: BatchReactorOptions | CSTRReactorOptions | PFRReactorOptions | PBRReactorOptions,
    ):
        # NOTE: set
        self.reactor_options = reactor_options

        # LINK: ThermoPropertyFields for typed attributes of thermodynamic properties
        ThermoPropertyFields.__init__(self)

        # LINK: SourceUtils for property assignment based on source configuration
        SourceUtils.__init__(
            self,
            thermo_custom_inputs=thermo_custom_inputs,
            thermo_model_source=thermo_model_source,
        )

    # NOTE: placeholder for launch method to be implemented in ThermoSource
    def launch(
            self,
    ):
        # deterministic order for reproducibility
        for prop_symbol in sorted(AVAILABLE_VARIABLES):
            custom_cfg = CUSTOM_INPUTS_ATTR_CONFIG.get(prop_symbol)
            model_cfg = MODEL_SOURCE_ATTR_CONFIG.get(prop_symbol)

            # derive source selector field
            prop_source_field = None
            if custom_cfg is not None:
                prop_source_field = custom_cfg.get("prop_source")
            if prop_source_field is None and model_cfg is not None:
                prop_source_field = model_cfg.get("prop_source")
            if prop_source_field is None:
                continue

            reactor_option_source = getattr(
                self.reactor_options,
                cast(str, prop_source_field),
                None
            )
            if reactor_option_source is None:
                logger.info(
                    "No source configured for property '%s' in reactor options. Skipping assignment.",
                    prop_symbol,
                )
                continue

            # choose configuration that matches selected source
            if reactor_option_source == "custom_inputs":
                prop_details = custom_cfg
            elif reactor_option_source == "model_source":
                prop_details = model_cfg
            else:
                prop_details = None

            if prop_details is None:
                logger.info(
                    "No matching config for property '%s' with source '%s'. Skipping.",
                    prop_symbol,
                    reactor_option_source,
                )
                continue

            assigner_mode = prop_details.get("assigner_mode")
            if assigner_mode is None:
                continue

            # compatibility aliases
            mode_map = {
                "property": "constant",
                "properties": "constants",
            }
            assigner_mode = mode_map.get(
                cast(str, assigner_mode), assigner_mode)

            if assigner_mode not in {"data", "equation", "constant", "constants"}:
                logger.warning(
                    "Unknown assigner mode '%s' for symbol '%s'. Skipping.",
                    assigner_mode,
                    prop_symbol,
                )
                continue

            res = self.assigner(
                symbol=prop_symbol,
                mode=cast(Any, assigner_mode),
            )

            if assigner_mode in {"data", "equation"}:
                values, values_comp, source = cast(
                    Tuple[np.ndarray, Dict[str, float], Dict[str, Any]],
                    res
                )
                setattr(self, prop_symbol, values)
                setattr(self, f"{prop_symbol}_comp", values_comp)
                setattr(self, f"{prop_symbol}_src", source)
            else:
                # constant / constants
                setattr(self, prop_symbol, res)
