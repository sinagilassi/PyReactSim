# import libs
import logging
import numpy as np
from typing import Dict, List, Tuple, Any, Literal, Optional
from pythermodb_settings.models import Component, CustomProp, ComponentKey
# locals
from ..models.br import BatchReactorOptions
from ..models.cstr import CSTRReactorOptions
from ..models.pfr import PFRReactorOptions
from ..models.pbr import PBRReactorOptions
from ..utils.tools import config_components_property
from ..models.heat import HeatTransferOptions
from ..utils.unit_tools import to_J_per_mol_K, to_g_per_m3, to_J_per_mol, to_g_per_mol

# NOTE: logger
logger = logging.getLogger(__name__)


class ThermoModelInputs:
    """
    ThermoModelInputs is a class that encapsulates the inputs required for configuring the thermodynamic properties in the reactor models.
    This class is designed to retrieve the following properties for the components in the system:

    - Ideal gas heat capacity (Cp_IG)
    - Liquid heat capacity (Cp_LIQ)
    - Liquid density (rho_LIQ)
    - Ideal gas formation enthalpy at 298 K (EnFo_IG)
    - Molecular weight (MW)
    """
    # NOTE: Attributes
    # ! heat capacity of ideal gas
    Cp_IG: np.ndarray = np.array([])
    Cp_IG_comp: Dict[str, float] = {}
    # ! heat capacity of liquid
    Cp_LIQ: np.ndarray = np.array([])
    Cp_LIQ_comp: Dict[str, float] = {}
    # ! liquid density
    rho_LIQ: np.ndarray = np.array([])
    rho_LIQ_comp: Dict[str, float] = {}
    # ! mixture liquid density
    rho_LIQ_MIX: float = 0.0
    # ! ideal gas formation enthalpy at 298 K
    EnFo_IG_298_src: Dict[str, Dict[str, Any]] = {}
    EnFo_IG_298: np.ndarray = np.array([])
    EnFo_IG_298_comp: Dict[str, float] = {}
    # ! molecular weight
    MW_src: Dict[str, Dict[str, Any]] = {}
    MW: np.ndarray = np.array([])
    MW_comp: Dict[str, float] = {}
    # ! enthalpy of reaction
    dH_rxn_src: Dict[str, CustomProp] = {}

    def __init__(
        self,
        components: List[Component],
        thermo_inputs: Dict[str, Any],
        reactor_options: BatchReactorOptions | CSTRReactorOptions | PFRReactorOptions | PBRReactorOptions,
        heat_transfer_options: HeatTransferOptions,
        component_refs: Dict[str, Any],
        component_key: ComponentKey,
    ):
        """
        Initializes the ThermoInputs instance with the provided components, source, model inputs, reactor inputs, reaction rates, and component key.

        Parameters
        ----------
        components : List[Component]
            A list of Component objects representing the chemical components involved in the model source.
        thermo_inputs : Dict[str, Any]
            A dictionary of model inputs, where the keys are the names of the inputs and the values are the input values. This can include feed specifications, initial conditions, or any other relevant parameters needed for the simulations.
        reactor_inputs : BatchReactorOptions | CSTRReactorOptions | PFRReactorOptions | PBRReactorOptions
            A reactor options object containing phase and thermodynamic configuration.
        heat_transfer_options : HeatTransferOptions
            A HeatTransferOptions object containing the inputs for heat transfer in the batch reactor simulation.
        component_refs : Dict[str, Any]
            A dictionary of component references, where the keys are the names of the references and the values are the reference values or objects.
        component_key : ComponentKey
            A ComponentKey object that serves as a key for identifying and categorizing the components in the model source.
        """
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

        # SECTION: Reactor configuration
        # ! gas heat capacity mode
        self.gas_heat_capacity_mode = reactor_options.gas_heat_capacity_mode
        # ! liquid heat capacity mode
        self.liquid_heat_capacity_mode = reactor_options.liquid_heat_capacity_mode
        # ! density mode
        self.liquid_density_mode = reactor_options.liquid_density_mode
        # ! phase
        self.phase = reactor_options.phase

        # SECTION: heat transfer options
        # ! heat transfer mode
        self.heat_transfer_mode = heat_transfer_options.heat_transfer_mode

        # SECTION: Extract property sources and configure properties
        # ! Ideal Gas Heat Capacity at reference temperature (e.g., 298 K)
        if self.heat_transfer_mode == "non-isothermal":
            # check heat capacity mode
            if (
                self.gas_heat_capacity_mode == "constant" and
                self.reactor_options.gas_heat_capacity_source == "model_inputs"  # ! source
            ):
                # NOTE: use constant heat capacity from model inputs
                # >> constant heat capacity
                # ! to J/mol.K
                (
                    self.Cp_IG,
                    self.Cp_IG_comp
                ) = self._config_constant_gas_heat_capacity()

            # NOTE: Enthalpy of formation at 298 K for ideal gas
            if self.reactor_options.ideal_gas_formation_enthalpy_source == "model_inputs":  # ! source
                # ! to J/mol
                self.EnFo_IG_298_src: Dict[
                    str, Dict[str, Any]
                ] = self._config_constant_ideal_gas_formation_enthalpy()

                # ! values in J/mol
                (
                    self.EnFo_IG_298,
                    self.EnFo_IG_298_comp
                ) = config_components_property(
                    component_ids=self.component_ids,
                    prop_source=self.EnFo_IG_298_src,
                    unit_conversion_func=to_J_per_mol
                )

            # NOTE: reaction enthalpy
            if (
                self.reactor_options.reaction_enthalpy_mode == "reaction" and
                self.reactor_options.reaction_enthalpy_source == "model_inputs"
            ):
                # ! to J/mol
                self.dH_rxn_src = self._config_reaction_enthalpy()

        # ! phase
        if self.phase == "liquid":
            # check heat capacity mode
            if (
                self.heat_transfer_mode == "non-isothermal" and
                self.liquid_heat_capacity_mode == "constant" and
                self.reactor_options.liquid_heat_capacity_source == "model_inputs"  # ! source
            ):
                # NOTE: use constant heat capacity from model inputs
                # >> constant heat capacity
                # ! to J/mol.K
                (
                    self.Cp_LIQ,
                    self.Cp_LIQ_comp
                ) = self._config_constant_liquid_heat_capacity()

            # check density mode
            if (
                self.liquid_density_mode == "constant" and
                self.reactor_options.liquid_density_source == "model_inputs"  # ! source
            ):
                # NOTE: use constant density from model inputs
                # >> constant density
                # ! to g/m3
                (
                    self.rho_LIQ,
                    self.rho_LIQ_comp
                ) = self._config_constant_liquid_density()

            if (
                self.liquid_density_mode == "mixture" and
                self.reactor_options.liquid_density_source == "model_inputs"  # ! source
            ):
                # NOTE: use mixture density from model inputs
                # >> mixture density
                # ! to g/m3
                self.rho_LIQ_MIX = to_g_per_m3(
                    self.thermo_inputs["liquid_density_mixture"].value,
                    self.thermo_inputs["liquid_density_mixture"].unit
                )

            # molecular weight
            if self.reactor_options.molecular_weight_source == "model_inputs":  # ! source
                # NOTE: use molecular weight from model inputs
                # ! to g/mol
                self.MW_src: Dict[
                    str, Dict[str, Any]
                ] = self._config_molecular_weight()

                # ! values in g/mol
                (
                    self.MW,
                    self.MW_comp
                ) = config_components_property(
                    component_ids=self.component_ids,
                    prop_source=self.MW_src,
                    unit_conversion_func=to_g_per_mol
                )

    # SECTION: configuration methods for properties
    # ! gas phase heat capacity configuration
    def _config_constant_gas_heat_capacity(
            self,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Configure the heat capacity in [J/mol.K] for the batch reactor based on the model inputs and reactor configuration."""
        # check heat capacity mode
        if self.gas_heat_capacity_mode is None:
            raise ValueError(
                "Heat capacity mode must be specified in reactor_inputs for non-isothermal reactors.")

        # heat capacity constant
        if "gas_heat_capacity" in self.thermo_inputs_keys:
            heat_capacity_: dict[
                str,
                CustomProp
            ] = self.thermo_inputs["gas_heat_capacity"]

            # iterate through components and extract heat capacity values
            heat_capacity_values = []
            heat_capacity_comp = {}

            for id in self.component_formula_state:
                if id in heat_capacity_:
                    cp_value = to_J_per_mol_K(
                        heat_capacity_[id].value,
                        heat_capacity_[id].unit
                    )

                    # add
                    heat_capacity_values.append(cp_value)
                    heat_capacity_comp[id] = cp_value
                else:
                    raise ValueError(
                        f"Heat capacity value for component '{id}' not found in model_inputs."
                    )

            heat_capacity_array = np.array(heat_capacity_values)

            # res
            return heat_capacity_array, heat_capacity_comp
        else:
            raise ValueError(
                "Heat capacity must be provided in model_inputs for constant heat capacity mode."
            )

    # ! constant liquid phase heat capacity
    def _config_constant_liquid_heat_capacity(
            self,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Configure the heat capacity in [J/mol.K] for the batch reactor based on the model inputs and reactor configuration.
        """
        # check heat capacity mode
        if self.liquid_heat_capacity_mode is None:
            raise ValueError(
                "Heat capacity mode must be specified in reactor_inputs for non-isothermal reactors.")

        # heat capacity constant
        if "liquid_heat_capacity" in self.thermo_inputs_keys:
            heat_capacity_: dict[
                str,
                CustomProp
            ] = self.thermo_inputs["liquid_heat_capacity"]

            # iterate through components and extract heat capacity values
            heat_capacity_values = []
            heat_capacity_comp = {}

            for id in self.component_formula_state:
                if id in heat_capacity_:
                    cp_value = to_J_per_mol_K(
                        heat_capacity_[id].value,
                        heat_capacity_[id].unit
                    )

                    # add
                    heat_capacity_values.append(cp_value)
                    heat_capacity_comp[id] = cp_value
                else:
                    raise ValueError(
                        f"Heat capacity value for component '{id}' not found in model_inputs."
                    )

            heat_capacity_array = np.array(heat_capacity_values)

            # res
            return heat_capacity_array, heat_capacity_comp
        else:
            raise ValueError(
                "Heat capacity must be provided in model_inputs for constant heat capacity mode."
            )

    # ! liquid density
    def _config_constant_liquid_density(
            self
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Configure the density in [g/m3] for the batch reactor based on the model inputs and reactor configuration.
        """
        # check density mode
        if self.liquid_density_mode is None:
            raise ValueError(
                "Density mode must be specified in reactor_inputs for liquid phase.")

        # density constant
        if "liquid_density" in self.thermo_inputs_keys:
            density_: dict[
                str,
                CustomProp
            ] = self.thermo_inputs["liquid_density"]

            # iterate through components and extract density values
            density_values = []
            density_comp = {}

            for id in self.component_formula_state:
                if id in density_:
                    density_value = to_g_per_m3(
                        density_[id].value,
                        density_[id].unit
                    )

                    # add
                    density_values.append(density_value)
                    density_comp[id] = density_value
                else:
                    raise ValueError(
                        f"Density value for component '{id}' not found in model_inputs."
                    )

            density_array = np.array(density_values)

            # res
            return density_array, density_comp
        else:
            raise ValueError(
                "Density must be provided in model_inputs for constant density mode."
            )

    # ! ideal gas formation enthalpy at 298 K
    def _config_constant_ideal_gas_formation_enthalpy(
            self
    ) -> Dict[str, Dict[str, Any]]:
        """
        Configure the ideal gas formation enthalpy at 298 K in [J/mol] for the batch reactor based on the model inputs and reactor configuration.
        """
        # check ideal gas formation enthalpy mode
        if self.reactor_options.ideal_gas_formation_enthalpy_source is None:
            raise ValueError(
                "Ideal gas formation enthalpy mode must be specified in reactor_inputs for non-isothermal reactors."
            )

        # ideal gas formation enthalpy constant
        if "ideal_gas_formation_enthalpy" in self.thermo_inputs_keys:
            EnFo_IG_298_: dict[
                str,
                CustomProp
            ] = self.thermo_inputs["ideal_gas_formation_enthalpy"]

            # iterate through components and extract ideal gas formation enthalpy values
            EnFo_IG_298_src = {}

            for id_formula_state, id_name_formula in zip(self.component_formula_state, self.component_ids):
                if id_formula_state in EnFo_IG_298_:
                    # add
                    EnFo_IG_298_src[id_name_formula] = {
                        "value": EnFo_IG_298_[id_formula_state].value,
                        "unit": EnFo_IG_298_[id_formula_state].unit
                    }
                else:
                    raise ValueError(
                        f"Ideal gas formation enthalpy value for component '{id_formula_state}' not found in model_inputs."
                    )

            # res
            return EnFo_IG_298_src
        else:
            raise ValueError(
                "Ideal gas formation enthalpy must be provided in model_inputs for constant ideal gas formation enthalpy mode."
            )

    # ! molecular weight
    def _config_molecular_weight(
            self
    ):
        """
        Configure the molecular weight in [g/mol] for the batch reactor based on the model inputs and reactor configuration.
        """
        # check molecular weight source
        if self.reactor_options.molecular_weight_source is None:
            raise ValueError(
                "Molecular weight source must be specified in reactor_inputs."
            )

        # molecular weight
        if "molecular_weight" in self.thermo_inputs_keys:
            molecular_weight_: dict[
                str,
                CustomProp
            ] = self.thermo_inputs["molecular_weight"]

            # iterate through components and extract molecular weight values
            molecular_weight_src = {}

            for id_formula_state, id_name_formula in zip(self.component_formula_state, self.component_ids):
                if id_formula_state in molecular_weight_:
                    # add
                    molecular_weight_src[id_name_formula] = {
                        "value": molecular_weight_[id_formula_state].value,
                        "unit": molecular_weight_[id_formula_state].unit
                    }
                else:
                    raise ValueError(
                        f"Molecular weight value for component '{id_formula_state}' not found in model_inputs."
                    )

            # res
            return molecular_weight_src
        else:
            raise ValueError(
                "Molecular weight must be provided in model_inputs for molecular weight source mode."
            )

    # ! reaction enthalpy
    def _config_reaction_enthalpy(
            self
    ) -> Dict[str, CustomProp]:
        """
        Configure the reaction enthalpy in [J/mol] for the batch reactor based on the model inputs and reactor configuration.

        Notes
        -----
        - The reaction enthalpy is given for each reaction in the system by reaction-name in model inputs.
        """
        # check reaction enthalpy source
        if self.reactor_options.reaction_enthalpy_source is None:
            raise ValueError(
                "Reaction enthalpy source must be specified in reactor_inputs."
            )

        # reaction enthalpy
        if "reaction_enthalpy" in self.thermo_inputs_keys:
            reaction_enthalpy_src: dict[
                str,
                CustomProp
            ] = self.thermo_inputs["reaction_enthalpy"]

            # iterate through reactions and extract reaction enthalpy values
            for k, v in reaction_enthalpy_src.items():
                # convert to J/mol
                reaction_enthalpy_src[k] = CustomProp(
                    value=to_J_per_mol(v.value, v.unit),
                    unit="J/mol"
                )

            # res
            return reaction_enthalpy_src
        else:
            raise ValueError(
                "Reaction enthalpy must be provided in model_inputs for reaction enthalpy source mode."
            )
