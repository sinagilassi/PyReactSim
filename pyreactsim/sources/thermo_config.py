# import libs
from ..utils.unit_tools import (
    to_J_per_mol_K,
    to_g_per_m3,
    to_J_per_mol,
    to_g_per_mol
)

# SECTION: Model Source
# NOTE: configuration
MODEL_SOURCE_ATTR_CONFIG = {
    "Cp_IG": {
        "description": "Ideal gas heat capacity equation source for each component.",
        "method": "equation-source",
        "prop_symbol": "Cp_IG",
        "prop_name": "Cp_IG",
        "prop_source": "gas_heat_capacity_source",
        "unit_conversion_func": None,
        "expected_unit": None,
        "strict_unit_check": None,
        "phase": {
            "any": {
                "phase": ["gas", "liquid"],
            },
        },
        "heat_transfer_mode": {
            "all": {
                "heat_transfer_mode": ["non-isothermal"],
            }
        },
        "assigner_mode": "equation",
    },
    "EnFo_IG_298": {
        "description": "Ideal gas formation enthalpy at 298 K data source for each component.",
        "method": "data-source",
        "prop_symbol": "EnFo_IG_298",
        "prop_name": "EnFo_IG",  # ! used to retrieved from model source
        "prop_source": "ideal_gas_formation_enthalpy_source",
        "unit_conversion_func": to_J_per_mol,
        "expected_unit": "J/mol",
        "strict_unit_check": True,
        "phase": {
            "any": {
                "phase": ["gas", "liquid"],
            },
        },
        "heat_transfer_mode": {
            "all": {
                "heat_transfer_mode": ["non-isothermal"],
            }
        },
        "assigner_mode": "data",
        "executer": "calc_dH_rxns_298",
    },
    "MW": {
        "description": "Molecular weight data source for each component.",
        "method": "data-source",
        "prop_symbol": "MW",
        "prop_name": "MW",
        "prop_source": "molecular_weight_source",
        "unit_conversion_func": to_g_per_mol,
        "expected_unit": "g/mol",
        "strict_unit_check": True,
        "phase": {
            "all": {
                "phase": ["liquid"],
            },
        },
        "heat_transfer_mode": {
            "any": {
                "heat_transfer_mode": ["non-isothermal", "isothermal"],
            }
        },
        "assigner_mode": "data",
    },
    "rho_LIQ": {
        "description": "Liquid density equation source for each component.",
        "method": "equation-source",
        "prop_symbol": "rho_LIQ",
        "prop_name": "rho_LIQ",
        "prop_source": "liquid_density_source",
        "unit_conversion_func": None,
        "expected_unit": None,
        "strict_unit_check": None,
        "phase": {
            "all": {
                "phase": ["liquid"],
            },
        },
        "heat_transfer_mode": {
            "any": {
                "heat_transfer_mode": ["non-isothermal", "isothermal"],
            }
        },
        "assigner_mode": "equation",
    },
    "Cp_LIQ": {
        "description": "Liquid heat capacity equation source for each component.",
        "method": "equation-source",
        "prop_symbol": "Cp_LIQ",
        "prop_name": "Cp_LIQ",
        "prop_source": "liquid_heat_capacity_source",
        "unit_conversion_func": None,
        "expected_unit": None,
        "strict_unit_check": None,
        "phase": {
            "all": {
                "phase": ["liquid"],
            },
        },
        "heat_transfer_mode": {
            "all": {
                "heat_transfer_mode": ["non-isothermal"],
            }
        },
        "assigner_mode": "equation",
    },
    "EnFo_LIQ_298": {
        "description": "Liquid formation enthalpy at 298 K data source for each component.",
        "method": "data-source",
        "prop_symbol": "EnFo_LIQ_298",
        "prop_name": "EnFo_LIQ",
        "prop_source": "liquid_formation_enthalpy_source",
        "unit_conversion_func": to_J_per_mol,
        "expected_unit": "J/mol",
        "strict_unit_check": True,
        "phase": {
            "any": {
                "phase": ["liquid"],
            },
        },
        "heat_transfer_mode": {
            "all": {
                "heat_transfer_mode": ["non-isothermal"],
            }
        },
        "assigner_mode": "data",
    },
}

# NOTE: criteria for model source
MODEL_SOURCE_CRITERIA = {
    "Cp_IG": {
        "all": {
            "gas_heat_capacity_mode": ["temperature-dependent"],
            "gas_heat_capacity_source": ["model_source"],
        }
    },
    "EnFo_IG_298": {
        "all": {
            "ideal_gas_formation_enthalpy_mode": ["temperature-dependent"],
            "ideal_gas_formation_enthalpy_source": ["model_source"],
        },
        "not": {
            "reaction_enthalpy_mode": ["reaction"],
        }
    },
    "MW": {
        "all": {
            "molecular_weight_source": ["model_source"],
        }
    },
    "rho_LIQ": {
        "all": {
            "liquid_density_mode": ["temperature-dependent"],
            "liquid_density_source": ["model_source"],
        }
    },
    "Cp_LIQ": {
        "all": {
            "liquid_heat_capacity_mode": ["temperature-dependent"],
            "liquid_heat_capacity_source": ["model_source"],
        }
    },
    "EnFo_LIQ_298": {
        "all": {
            "liquid_formation_enthalpy_mode": ["constant"],
            "liquid_formation_enthalpy_source": ["model_source"],
        },
        "not": {
            "reaction_enthalpy_mode": ["reaction"],
        }
    },
}

# SECTION: Model Inputs
# NOTE: configurations
CUSTOM_INPUTS_ATTR_CONFIG = {
    "Cp_IG": {
        "description": "Ideal gas heat capacity for each component.",
        "method": "property-source",
        "prop_symbol": "Cp_IG",
        "prop_name": "gas_heat_capacity",
        "prop_source": "gas_heat_capacity_source",
        "unit_conversion_func": to_J_per_mol_K,
        "expected_unit": "J/mol.K",
        "strict_unit_check": True,
        "phase": {
            "any": {
                "phase": ["gas", "liquid"],
            },
        },
        "heat_transfer_mode": {
            "all": {
                "heat_transfer_mode": ["non-isothermal"],
            }
        },
        "assigner_mode": "data"
    },
    "Cp_LIQ": {
        "description": "Liquid heat capacity for each component.",
        "method": "property-source",
        "prop_symbol": "Cp_LIQ",
        "prop_name": "liquid_heat_capacity",
        "prop_source": "liquid_heat_capacity_source",
        "unit_conversion_func": to_J_per_mol_K,
        "expected_unit": "J/mol.K",
        "strict_unit_check": True,
        "phase": {
            "all": {
                "phase": ["liquid"],
            },
        },
        "heat_transfer_mode": {
            "all": {
                "heat_transfer_mode": ["non-isothermal"],
            }
        },
        "assigner_mode": "data"
    },
    "rho_LIQ": {
        "description": "Liquid density for each component.",
        "method": "property-source",
        "prop_symbol": "rho_LIQ",
        "prop_name": "liquid_density",
        "prop_source": "liquid_density_source",
        "unit_conversion_func": to_g_per_m3,
        "expected_unit": "g/m3",
        "strict_unit_check": True,
        "phase": {
            "all": {
                "phase": ["liquid"],
            },
        },
        "heat_transfer_mode": {
            "any": {
                "heat_transfer_mode": ["non-isothermal", "isothermal"],
            }
        },
        "assigner_mode": "data"
    },
    "rho_LIQ_MIX": {
        "description": "Mixture liquid density.",
        "method": "property-constant",
        "prop_symbol": "rho_LIQ_MIX",
        "prop_name": "liquid_density_mixture",
        "prop_source": "liquid_density_mixture_source",
        "unit_conversion_func": to_g_per_m3,
        "expected_unit": "g/m3",
        "strict_unit_check": True,
        "phase": {
            "all": {
                "phase": ["liquid"],
            },
        },
        "heat_transfer_mode": {
            "any": {
                "heat_transfer_mode": ["non-isothermal", "isothermal"],
            }
        },
        "assigner_mode": "property"
    },
    "MW": {
        "description": "Molecular weight for each component.",
        "method": "property-source",
        "prop_symbol": "MW",
        "prop_name": "molecular_weight",
        "prop_source": "molecular_weight_source",
        "unit_conversion_func": to_g_per_mol,
        "expected_unit": "g/mol",
        "strict_unit_check": True,
        "phase": {
            "all": {
                "phase": ["liquid"],
            },
        },
        "heat_transfer_mode": {
            "any": {
                "heat_transfer_mode": ["non-isothermal", "isothermal"],
            }
        },
        "assigner_mode": "data"
    },
    "EnFo_IG_298": {
        "description": "Ideal gas formation enthalpy at 298 K for each component.",
        "method": "property-source",
        "prop_symbol": "EnFo_IG_298",
        "prop_name": "ideal_gas_formation_enthalpy",
        "prop_source": "ideal_gas_formation_enthalpy_source",
        "unit_conversion_func": to_J_per_mol,
        "expected_unit": "J/mol",
        "strict_unit_check": True,
        "phase": {
            "any": {
                "phase": ["gas", "liquid"],
            },
        },
        "heat_transfer_mode": {
            "all": {
                "heat_transfer_mode": ["non-isothermal"],
            }
        },
        "assigner_mode": "data",
        "executer": "calc_dH_rxns_298",
    },
    "dH_rxn": {
        "description": "Reaction enthalpy for each reaction in the system.",
        "method": "property-constants",
        "prop_symbol": "dH_rxn",
        "prop_name": "reaction_enthalpy",
        "prop_source": "reaction_enthalpy_source",
        "unit_conversion_func": to_J_per_mol,
        "expected_unit": "J/mol",
        "strict_unit_check": True,
        "phase": {
            "any": {
                "phase": ["gas", "liquid"],
            },
        },
        "heat_transfer_mode": {
            "all": {
                "heat_transfer_mode": ["non-isothermal"],
            }
        },
        "assigner_mode": "constants",
    },
    "Cp_IG_MIX_TOTAL": {
        "description": "Total heat capacity of gas mixture.",
        "method": "property-constant",
        "prop_symbol": "Cp_IG_MIX_TOTAL",
        "prop_name": "gas_mixture_total_heat_capacity",
        "prop_source": "gas_mixture_total_heat_capacity_source",
        "unit_conversion_func": None,
        "expected_unit": "J/K",
        "strict_unit_check": False,
        "phase": {
            "all": {
                "phase": ["gas"],
            },
        },
        "heat_transfer_mode": {
            "all": {
                "heat_transfer_mode": ["non-isothermal"],
            }
        },
        "assigner_mode": "constant"
    },
    "Cp_LIQ_MIX_TOTAL": {
        "description": "Total heat capacity of liquid mixture.",
        "method": "property-constant",
        "prop_symbol": "Cp_LIQ_MIX_TOTAL",
        "prop_name": "liquid_mixture_total_heat_capacity",
        "prop_source": "liquid_mixture_total_heat_capacity_source",
        "unit_conversion_func": None,
        "expected_unit": "J/K",
        "strict_unit_check": False,
        "phase": {
            "all": {
                "phase": ["liquid"],
            },
        },
        "heat_transfer_mode": {
            "all": {
                "heat_transfer_mode": ["non-isothermal"],
            }
        },
        "assigner_mode": "constant"
    },
    "Cp_LIQ_MIX_VOLUMETRIC": {
        "description": "Volumetric heat capacity of liquid mixture.",
        "method": "property-constant",
        "prop_symbol": "Cp_LIQ_MIX_VOLUMETRIC",
        "prop_name": "liquid_mixture_volumetric_heat_capacity",
        "prop_source": "liquid_mixture_volumetric_heat_capacity_source",
        "unit_conversion_func": None,
        "expected_unit": "J/m3.K",
        "strict_unit_check": False,
        "phase": {
            "all": {
                "phase": ["liquid"],
            },
        },
        "heat_transfer_mode": {
            "all": {
                "heat_transfer_mode": ["non-isothermal"],
            }
        },
        "assigner_mode": "constant"
    },
}

# NOTE: criteria for model inputs
CUSTOM_INPUTS_CRITERIA = {
    "Cp_IG": {
        "all": {
            "gas_heat_capacity_mode": ["constant"],
            "gas_heat_capacity_source": ["custom_inputs"],
        }
    },
    "EnFo_IG_298": {
        "all": {
            "ideal_gas_formation_enthalpy_mode": ["constant"],
            "ideal_gas_formation_enthalpy_source": ["custom_inputs"],
        },
        "not": {
            "reaction_enthalpy_mode": ["reaction"],
        }
    },
    "dH_rxn": {
        "all": {
            "reaction_enthalpy_mode": ["reaction"],
            "reaction_enthalpy_source": ["custom_inputs"],
        }
    },
    "Cp_IG_MIX_TOTAL": {
        "all": {
            "use_gas_mixture_total_heat_capacity": [True],
            "gas_mixture_total_heat_capacity_source": ["custom_inputs"],
        }
    },
    "Cp_LIQ": {
        "all": {
            "liquid_heat_capacity_mode": ["constant"],
            "liquid_heat_capacity_source": ["custom_inputs"],
        }
    },
    "rho_LIQ": {
        "all": {
            "liquid_density_mode": ["constant"],
            "liquid_density_source": ["custom_inputs"],
        }
    },
    "rho_LIQ_MIX": {
        "all": {
            "liquid_density_mode": ["mixture"],
            "liquid_density_source": ["custom_inputs"],
        }
    },
    "MW": {
        "all": {
            "molecular_weight_source": ["custom_inputs"],
        },
        "any": {
            "operation_mode": ["variable_volume", "constant_pressure"],
        }
    },
    "Cp_LIQ_MIX_TOTAL": {
        "all": {
            "use_liquid_mixture_total_heat_capacity": [True],
            "liquid_mixture_total_heat_capacity_source": ["custom_inputs"],
        }
    },
    "Cp_LIQ_MIX_VOLUMETRIC": {
        "all": {
            "use_liquid_mixture_volumetric_heat_capacity": [True],
            "liquid_mixture_volumetric_heat_capacity_source": ["custom_inputs"],
        }
    },
}


# NOTE: available variables for all models
AVAILABLE_VARIABLES = set(
    list(CUSTOM_INPUTS_ATTR_CONFIG.keys()) +
    list(MODEL_SOURCE_ATTR_CONFIG.keys())
)
