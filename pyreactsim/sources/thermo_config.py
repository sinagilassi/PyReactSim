# import libs
from ..utils.unit_tools import (
    to_J_per_mol_K,
    to_g_per_m3,
    to_J_per_mol,
    to_g_per_mol
)

# SECTION: Model Inputs
# NOTE: configurations
MODEL_INPUTS_ATTR_CONFIG = {
    "Cp_IG": {
        "description": "Ideal gas heat capacity for each component.",
        "method": "property-source",
        "prop_name": "gas_heat_capacity",
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
    },
    "Cp_LIQ": {
        "description": "Liquid heat capacity for each component.",
        "method": "property-source",
        "prop_name": "liquid_heat_capacity",
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
    },
    "rho_LIQ": {
        "description": "Liquid density for each component.",
        "method": "property-source",
        "prop_name": "liquid_density",
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
    },
    "rho_LIQ_MIX": {
        "description": "Mixture liquid density.",
        "method": "property-constant",
        "prop_name": "liquid_density_mixture",
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
    },
    "MW": {
        "description": "Molecular weight for each component.",
        "method": "property-source",
        "prop_name": "molecular_weight",
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
    },
    "EnFo_IG_298": {
        "description": "Ideal gas formation enthalpy at 298 K for each component.",
        "method": "property-source",
        "prop_name": "ideal_gas_formation_enthalpy",
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
    },
    "dH_rxn": {
        "description": "Reaction enthalpy for each reaction in the system.",
        "method": "property",
        "prop_name": "reaction_enthalpy",
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
    },
    "Cp_IG_MIX_TOTAL": {
        "description": "Total heat capacity of gas mixture.",
        "method": "property-constant",
        "prop_name": "gas_mixture_total_heat_capacity",
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
    },
    "Cp_LIQ_MIX_TOTAL": {
        "description": "Total heat capacity of liquid mixture.",
        "method": "property-constant",
        "prop_name": "liquid_mixture_total_heat_capacity",
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
    },
    "Cp_LIQ_MIX_VOLUMETRIC": {
        "description": "Volumetric heat capacity of liquid mixture.",
        "method": "property-constant",
        "prop_name": "liquid_mixture_volumetric_heat_capacity",
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
    },
}

# NOTE: criteria for model inputs
MODEL_INPUTS_CRITERIA = {
    "gas_heat_capacity": {
        "all": {
            "gas_heat_capacity_mode": ["constant"],
            "gas_heat_capacity_source": ["model_inputs"],
        }
    },
    "ideal_gas_formation_enthalpy": {
        "all": {
            "ideal_gas_formation_enthalpy_source": ["model_inputs"],
        },
        "not": {
            "reaction_enthalpy_mode": ["reaction"],
        }
    },
    "reaction_enthalpy": {
        "all": {
            "reaction_enthalpy_mode": ["reaction"],
            "reaction_enthalpy_source": ["model_inputs"],
        }
    },
    "gas_mixture_total_heat_capacity": {
        "all": {
            "use_gas_mixture_total_heat_capacity": [True],
            "gas_mixture_total_heat_capacity_source": ["model_inputs"],
        }
    },
    "liquid_heat_capacity": {
        "all": {
            "liquid_heat_capacity_mode": ["constant"],
            "liquid_heat_capacity_source": ["model_inputs"],
        }
    },
    "liquid_density": {
        "all": {
            "liquid_density_mode": ["constant"],
            "liquid_density_source": ["model_inputs"],
        }
    },
    "liquid_density_mixture": {
        "all": {
            "liquid_density_mode": ["mixture"],
            "liquid_density_source": ["model_inputs"],
        }
    },
    "molecular_weight": {
        "all": {
            "molecular_weight_source": ["model_inputs"],
        },
        "any": {
            "operation_mode": ["variable_volume", "constant_pressure"],
        }
    },
    "liquid_mixture_total_heat_capacity": {
        "all": {
            "use_liquid_mixture_total_heat_capacity": [True],
            "liquid_mixture_total_heat_capacity_source": ["model_inputs"],
        }
    },
    "liquid_mixture_volumetric_heat_capacity": {
        "all": {
            "use_liquid_mixture_volumetric_heat_capacity": [True],
            "liquid_mixture_volumetric_heat_capacity_source": ["model_inputs"],
        }
    },
}


# SECTION: Model Source
# NOTE: configuration
MODEL_SOURCE_ATTR_CONFIG = {}

# NOTE: criteria for model source
MODEL_SOURCE_CRITERIA = {}
