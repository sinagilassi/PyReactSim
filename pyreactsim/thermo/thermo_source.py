# import libs
import logging
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, cast
from pythermodb_settings.models import Component, ComponentKey, CustomProp, CustomProperty, Temperature, Pressure
from pythermodb_settings.utils import set_component_id, build_components_mapper
from pyThermoLinkDB.thermo import Source
from pyThermoLinkDB.models.component_models import ComponentEquationSource
from pyreactlab_core.models.reaction import Reaction
from pyThermoCalcDB.reactions.reactions import dH_rxn_STD

# locals
from ..sources.interface import (
    ext_component_dt,
    ext_components_dt,
    ext_component_eq,
    ext_components_eq,
    exec_component_eq
)
from ..utils.unit_tools import to_K, to_J_per_mol, to_J_per_mol_K, to_g_per_m3, to_g_per_mol
from ..utils.tools import find_components_property, collect_keys
from ..utils.reaction_tools import stoichiometry_mat
from ..models.rate_exp import ReactionRateExpression
from ..models.br import GasModel
from ..models.br import BatchReactorOptions

# NOTE: logger setup
logger = logging.getLogger(__name__)


class ThermoSource:
    """
    THermo class for handling thermodynamic calculations and properties related to chemical reactions and processes. This class provides methods for calculating various thermodynamic properties, such as heat capacities, enthalpies, and entropies, as well as methods for performing energy balance calculations in chemical systems.
    """
    # NOTE: Attributes
    # reference temperature
    T_ref = Temperature(value=298.15, unit="K")
    T_ref_K = 298.15
    # reference pressure
    P_ref = Pressure(value=101325, unit="Pa")

    def __init__(
        self,
        components: List[Component],
        source: Source,
        model_inputs: Dict[str, Any],
        reactor_inputs: BatchReactorOptions,
        reaction_rates: Dict[str, ReactionRateExpression],
        component_key: ComponentKey,
    ):
        """
        Initializes the THermo instance with default properties and settings for thermodynamic calculations.
        """
        # NOTE: Set attributes
        self.components = components
        self.source = source
        self.model_inputs = model_inputs
        self.reactor_inputs = reactor_inputs
        self.reaction_rates = reaction_rates
        self.component_key = component_key

        # NOTE: Create component ID list
        self.component_ids = [
            set_component_id(
                component=comp,
                component_key=self.component_key
            )
            for comp in self.components
        ]

        # >>> formula-state
        self.component_formula_state = [
            set_component_id(
                component=component,
                component_key='Formula-State'
            )
            for component in self.components
        ]

        # NOTE: build component mapper
        self.component_mapper: Dict[str, Dict[ComponentKey, str]] = build_components_mapper(
            components=self.components,
            component_key=self.component_key
        )

        # NOTE: model source
        self.model_source = self.source.model_source

        # NOTE: reactions
        self.reactions: List[Reaction] = self.build_reactions()

        # SECTION: Process model configuration
        # lower case keys for easier access
        self.model_inputs_keys = collect_keys(self.model_inputs)

        # SECTION: Reactor configuration
        self.gas_heat_capacity_mode = reactor_inputs.gas_heat_capacity_mode
        self.liquid_heat_capacity_mode = reactor_inputs.liquid_heat_capacity_mode

        # phase
        self.phase = reactor_inputs.phase
        # density mode
        self.liquid_density_mode = reactor_inputs.liquid_density_mode

        # SECTION: Thermodynamic properties
        # heat transfer more
        self.heat_transfer_mode = self.reactor_inputs.heat_transfer_mode

        # SECTION: Voids

        # ! Ideal Gas Heat Capacity at reference temperature (e.g., 298 K)
        self.Cp_IG_src: Dict[str, ComponentEquationSource] = {}
        self.gas_heat_capacity_constant_values: np.ndarray = np.array([])
        self.gas_heat_capacity_constant_comp: Dict[str, float] = {}
        self.dCp_rxns: np.ndarray = np.array([])

        # ! Ideal Gas Enthalpy of formation at 298 K
        self.EnFo_IG_298_src: Dict[str, Dict[str, Any]] = {}
        self.EnFo_IG_298_comp: Dict[str, float] = {}
        self.dH_rxns_298: np.ndarray = np.array([])

        # ! Phase-specific properties
        self.MW_src: Dict[str, Dict[str, Any]] = {}
        self.MW: np.ndarray = np.array([])
        self.MW_comp: Dict[str, float] = {}
        self.rho_LIQ_src: Dict[str, ComponentEquationSource] = {}
        self.liquid_density_constant_values: np.ndarray = np.array([])
        self.liquid_density_constant_comp: Dict[str, float] = {}
        self.Cp_LIQ_src: Dict[str, ComponentEquationSource] = {}
        self.liquid_heat_capacity_constant_values: np.ndarray = np.array([])
        self.liquid_heat_capacity_constant_comp: Dict[str, float] = {}

    # SECTION: Property equation source extraction methods
    # ! Extract property equation source for components

    def prop_eq_src(self, prop_name: str) -> Dict[str, ComponentEquationSource]:
        """
        Extracts the property equation for the components from the source and returns it as a dictionary.

        Returns
        -------
        Dict[str, ComponentEquationSource]
            A dictionary where the keys are component IDs and the values are ComponentEquationSource objects
        """
        # NOTE: Extract property equation source for all components
        eq_src = ext_components_eq(
            components=self.components,
            prop_name=prop_name,
            source=self.source,
            component_key=cast(ComponentKey, self.component_key),
            component_mapper=self.component_mapper
        )
        # >> check
        if eq_src is None:
            logger.error("Failed to extract property equation for components.")
            return {}

        return eq_src

    # ! Extract property data source for components
    def prop_dt_src(
            self,
            component_ids: List[str],
            prop_name: str
    ) -> Dict[str, Dict[str, Any]]:
        """
        Extracts the property derivative equation for the components from the source and returns it as a dictionary.

        Returns
        -------
        Dict[str, ComponentEquationSource]
            A dictionary where the keys are component IDs and the values are ComponentEquationSource objects
        """
        # NOTE: Extract property derivative equation source for all components
        dt_src = ext_components_dt(
            component_ids=component_ids,
            prop_name=prop_name,
            source=self.source,
        )
        # >> check
        if dt_src is None:
            logger.error(
                "Failed to extract property derivative equation for components.")
            return {}

        return dt_src

    # NOTE: heat capacity configuration
    # ! gas phase
    def _config_constant_gas_heat_capacity(
            self,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Configure the heat capacity for the batch reactor based on the model inputs and reactor configuration."""
        # check heat capacity mode
        if self.gas_heat_capacity_mode is None:
            raise ValueError(
                "Heat capacity mode must be specified in reactor_inputs for non-isothermal reactors.")

        # heat capacity constant
        if "gas_heat_capacity" in self.model_inputs_keys:
            heat_capacity_: dict[
                str,
                CustomProp
            ] = self.model_inputs["gas_heat_capacity"]

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
        Configure the heat capacity for the batch reactor based on the model inputs and reactor configuration.
        """
        # check heat capacity mode
        if self.liquid_heat_capacity_mode is None:
            raise ValueError(
                "Heat capacity mode must be specified in reactor_inputs for non-isothermal reactors.")

        # heat capacity constant
        if "liquid_heat_capacity" in self.model_inputs_keys:
            heat_capacity_: dict[
                str,
                CustomProp
            ] = self.model_inputs["liquid_heat_capacity"]

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
        Configure the density for the batch reactor based on the model inputs and reactor configuration.
        """
        # check density mode
        if self.liquid_density_mode is None:
            raise ValueError(
                "Density mode must be specified in reactor_inputs for liquid phase.")

        # density constant
        if "density" in self.model_inputs_keys:
            density_: dict[
                str,
                CustomProp
            ] = self.model_inputs["density"]

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

    # SECTION: Reaction and stoichiometry related methods
    # ! Extract all reactions
    def build_reactions(self):
        """
        Build the list of Reaction objects for the gas-phase batch reactor using the provided reaction rates and components.
        """
        reactions = []
        for rxn_name, rate_exp in self.reaction_rates.items():
            rxn = rate_exp.reaction
            reactions.append(rxn)
        return reactions

    # ! Build stoichiometry matrix
    def build_stoichiometry(self):
        """
        Build the stoichiometry matrix for the reactions in the gas-phase batch reactor using the provided reaction rates and components.
        """
        # >> extract reactions from reaction rates
        reactions = []

        for rxn_name, rate_exp in self.reaction_rates.items():
            rxn = rate_exp.reaction
            reactions.append(rxn)

        # >> build stoichiometry matrix
        mat = stoichiometry_mat(
            reactions=reactions,
            components=self.components,
            component_key=cast(ComponentKey, self.component_key)
        )

        return mat

    # SECTION: Thermodynamic property calculations
    # ! Calculate heat capacity at ideal gas for the components (Cp_IG)
    def calc_Cp_IG(
            self,
            temperature: Temperature,
    ):
        """
        Calculate the ideal gas heat capacity (Cp_IG) for the components in the batch reactor at the specified temperature.

        Parameters
        ----------
        temperature : Temperature
            The temperature at which to calculate the ideal gas heat capacity (Cp_IG) for the components in the batch reactor.

        Returns
        -------
        np.ndarray
            An array of ideal gas heat capacity (Cp_IG) values for the components in the batch reactor, calculated at the specified temperature.
        """
        # NOTE: temperature in K
        temp = to_K(temperature.value, temperature.unit)

        # NOTE: calculate heat capacity at ideal gas for the components based on the heat capacity mode
        if self.gas_heat_capacity_mode == "temperature-dependent":
            # NOTE: calculate temperature-dependent heat capacity
            Cp_IG_values = self.calc_Cp_IG_real(
                inputs={
                    "T": temp
                },
            )
        elif self.gas_heat_capacity_mode == "constant":
            # NOTE: use constant heat capacity from model inputs
            # ! J/mol.K
            Cp_IG_values = self.gas_heat_capacity_constant_values
        else:
            raise ValueError(
                f"Invalid heat_capacity_mode '{self.gas_heat_capacity_mode}'. Must be 'temperature-dependent' or 'constant'."
            )

        # >> check heat capacity values
        if Cp_IG_values is None:
            raise ValueError(
                "Heat capacity values could not be calculated or retrieved."
            )

        return Cp_IG_values

    # ! Heat capacity at ideal gas at temperature T (Cp_IG)
    def calc_Cp_IG_real(
            self,
            inputs: Dict[str, Any],
    ) -> np.ndarray:
        # NOTE: extract Cp_IG at reference temperature (e.g., 298 K)
        Cp_IG_ref: Dict[str, CustomProperty] = {}
        for comp in self.component_ids:
            eq_src = self.Cp_IG_src.get(comp)
            if eq_src is None:
                raise ValueError(
                    f"No Cp_IG source found for component: {comp}"
                )

            # >> execute equation to get Cp_IG at reference conditions
            cp_value = exec_component_eq(
                component_eq_src=eq_src,
                inputs=inputs,
                output_unit="J/mol.K"
            )
            if cp_value is None:
                raise ValueError(
                    f"Failed to extract Cp_IG value for component: {comp}"
                )

            # store
            Cp_IG_ref[comp] = cp_value

        # NOTE: build a list of Cp_IG values for all components
        Cp_IG_values = []

        # iterate over components
        for comp in self.component_ids:
            cp_value = Cp_IG_ref.get(comp)
            if cp_value is None:
                raise ValueError(
                    f"No Cp_IG value found for component: {comp}")
            Cp_IG_values.append(cp_value.value)

        # >> to numpy
        Cp_IG_values = np.array(Cp_IG_values, dtype=float)

        return Cp_IG_values

    # ! Calculate heat capacity at liquid phase for the components (Cp_LIQ)
    def calc_Cp_LIQ(
            self,
            temperature: Temperature,
    ):
        """
        Calculate the liquid phase heat capacity (Cp_LIQ) for the components in the batch reactor at the specified temperature.

        Parameters
        ----------
        temperature : Temperature
            The temperature at which to calculate the liquid phase heat capacity (Cp_LIQ) for the components in the batch reactor.

        Returns
        -------
        np.ndarray
            An array of liquid phase heat capacity (Cp_LIQ) values for the components in the batch reactor, calculated at the specified temperature.
        """
        # NOTE: temperature in K
        temp = to_K(temperature.value, temperature.unit)

        # NOTE: calculate heat capacity at liquid phase for the components based on the heat capacity mode
        if self.liquid_heat_capacity_mode == "temperature-dependent":
            # NOTE: calculate temperature-dependent heat capacity
            Cp_LIQ_values = self.calc_Cp_LIQ_real(
                inputs={
                    "T": temp
                },
            )
        elif self.liquid_heat_capacity_mode == "constant":
            # NOTE: use constant heat capacity from model inputs
            # ! J/mol.K
            Cp_LIQ_values = self.liquid_heat_capacity_constant_values
        else:
            raise ValueError(
                f"Invalid heat_capacity_mode '{self.liquid_heat_capacity_mode}'. Must be 'temperature-dependent' or 'constant'."
            )

        # >> check heat capacity values
        if Cp_LIQ_values is None:
            raise ValueError(
                "Heat capacity values could not be calculated or retrieved."
            )

        return Cp_LIQ_values

    # ! Heat capacity at liquid phase at temperature T (Cp_LIQ)
    def calc_Cp_LIQ_real(
            self,
            inputs: Dict[str, Any],
    ) -> np.ndarray:
        # NOTE: extract Cp_LIQ at reference temperature (e.g., 298 K)
        Cp_LIQ_ref: Dict[str, CustomProperty] = {}
        for comp in self.component_ids:
            eq_src = self.Cp_LIQ_src.get(comp)
            if eq_src is None:
                raise ValueError(
                    f"No Cp_LIQ source found for component: {comp}"
                )

            # >> execute equation to get Cp_LIQ at reference conditions
            cp_value = exec_component_eq(
                component_eq_src=eq_src,
                inputs=inputs,
                output_unit="J/mol.K"
            )
            if cp_value is None:
                raise ValueError(
                    f"Failed to extract Cp_LIQ value for component: {comp}"
                )

            # store
            Cp_LIQ_ref[comp] = cp_value

        # NOTE: build a list of Cp_LIQ values for all components
        Cp_LIQ_values = []

        # iterate over components
        for comp in self.component_ids:
            cp_value = Cp_LIQ_ref.get(comp)
            if cp_value is None:
                raise ValueError(
                    f"No Cp_LIQ value found for component: {comp}")
            Cp_LIQ_values.append(cp_value.value)

        # >> to numpy
        Cp_LIQ_values = np.array(Cp_LIQ_values, dtype=float)

        return Cp_LIQ_values

    # ! Calculate change in heat capacity at ideal gas for the reactions (ΔCp_IG)

    def calc_dCp_IG(
            self,
    ):
        """
        Calculate the change in heat capacity at ideal gas (ΔCp_IG) for the reactions as:
            ΔCp_rxn = sum(nu_i * Cp_IG_i) for all components i

        Returns
        -------
        np.ndarray
            An array of changes in heat capacity at ideal gas (ΔCp_IG) for each reaction.
        """
        # res
        dCp = []

        # check heat capacity constant
        if self.gas_heat_capacity_constant_comp is None:
            raise ValueError("Constant heat capacity values not found.")

        # iterate over reactions
        for rxn in self.reactions:
            # >> calculate reaction enthalpy for the reaction at 298 K
            # stoichiometry matrix
            nu = rxn.reaction_stoichiometry_matrix
            nu = np.array(nu, dtype=float)

            # components
            components = rxn.available_components
            # >> check
            if components is None:
                raise ValueError(
                    f"No components found for reaction: {rxn.name}")

            # heat capacity at ideal gas at reference temperature (e.g., 298 K) for the components
            Cp_IG_values, _ = find_components_property(
                components=components,
                prop_values=self.gas_heat_capacity_constant_comp,
                component_key="Formula-State"
            )

            # calculate mix heat capacity change for the reaction using the formula:
            # ΔCp_rxn = sum(nu_i * Cp_IG_i) for all components i
            dCp_rxn = np.sum(nu * Cp_IG_values)

            # >> check
            if dCp_rxn is None:
                raise ValueError(
                    f"Failed to calculate heat capacity change for reaction: {rxn.name}"
                )

            dCp.append(dCp_rxn)

        # NOTE: convert to numpy array
        res = np.array(dCp, dtype=float)

        return res

    # ! Calculate reaction enthalpies (ΔH) for reactions at 298 K
    def calc_dH_rxns_298(
            self,
    ):
        """
        Calculate the reaction enthalpies (ΔH) for the reactions in the gas-phase batch reactor at 298 K using the provided reaction rates and components.

        Returns
        -------
        np.ndarray
            An array of reaction enthalpies (ΔH) for the reactions in the gas-phase batch reactor, calculated at 298 K.
        """
        # res
        dH_rxns = []

        # iterate over reactions
        for rxn in self.reactions:
            # >> calculate reaction enthalpy for the reaction at 298 K
            # stoichiometry matrix
            nu = rxn.reaction_stoichiometry_matrix
            nu = np.array(nu, dtype=float)

            # components
            components = rxn.available_components
            # >> check
            if components is None:
                raise ValueError(
                    f"No components found for reaction: {rxn.name}")

            # Enthalpy of formation at 298 K for the components
            EnFo_IG_298_values, _ = find_components_property(
                components=components,
                prop_values=self.EnFo_IG_298_comp,
                component_key=cast(ComponentKey, self.component_key)
            )

            # calc
            dH_rxn = EnFo_IG_298_values @ nu

            # >> check
            if dH_rxn is None:
                raise ValueError(
                    f"Failed to calculate reaction enthalpy for reaction: {rxn.name}")

            dH_rxns.append(dH_rxn)

        # NOTE: convert to numpy array
        res = np.array(dH_rxns, dtype=float)

        return res

    # ! Calculate reaction enthalpies (ΔH) for reactions at temperature T
    def calc_dH_rxns(
            self,
            temperature: Temperature,
    ):
        """
        Calculate the reaction enthalpies (ΔH) for the reactions in the reactor.

        """
        # ! calculate heat generated by reactions: Q_rxn = V Σ_k [(-ΔH_k) r_k]
        # V[m3], ΔH[J/mol], r[mol/m3.s] => Q_rxn [J/s] or [W]
        # ??? ΔH_k
        if self.gas_heat_capacity_mode == "temperature-dependent":
            # NOTE: calculate temperature-dependent enthalpy of formation
            delta_h = self.calc_dH_rxns_real(
                temperature=temperature
            )
        elif self.gas_heat_capacity_mode == "constant":
            # NOTE: use constant enthalpy of formation from model inputs
            delta_h = self.calc_dH_rxns_linear(
                temperature=temperature,
            )
        else:
            raise ValueError(
                f"Invalid heat_capacity_mode '{self.gas_heat_capacity_mode}'. Must be 'temperature-dependent' or 'constant'."
            )

        # >> check enthalpy of formation values
        if delta_h is None:
            raise ValueError(
                "Enthalpy of formation values could not be calculated or retrieved."
            )

        return delta_h

    # ! Calculate reaction enthalpies (ΔH) for reactions at temperature T using temperature-dependent heat capacity values
    def calc_dH_rxns_real(
            self,
            temperature: Temperature,
    ) -> np.ndarray:
        """
        Calculate the reaction enthalpies (ΔH) for the reactions in the gas-phase batch reactor using the provided reaction rates and components.

        Parameters
        ----------
        temperature : Temperature
            The temperature at which to calculate the reaction enthalpies (ΔH) for the reactions
            in the gas-phase batch reactor.

        Returns
        -------
        np.ndarray
            An array of reaction enthalpies (ΔH) for the reactions in the gas-phase batch reactor, calculated at the specified temperature.
        """
        # create model source
        model_source = self.source.model_source
        # >> check
        if model_source is None:
            raise ValueError(
                "Model source is required to calculate reaction enthalpies.")
        #
        dH_rxns = []
        for rxn in self.reactions:
            # >> calculate reaction enthalpy for the reaction at the specified temperature
            dH_rxn = dH_rxn_STD(
                reaction=rxn,
                temperature=temperature,
                model_source=model_source,
            )

            # >> check
            if dH_rxn is None:
                raise ValueError(
                    f"Failed to calculate reaction enthalpy for reaction: {rxn.name}")

            dH_rxns.append(dH_rxn)

        # NOTE: convert to numpy array
        res = [dH.value for dH in dH_rxns]
        res = np.array(res, dtype=float)

        return res

    # ! Calculate reaction enthalpies (ΔH) for reactions at temperature T using constant heat capacity values
    def calc_dH_rxns_linear(
            self,
            temperature: Temperature,
    ):
        """
        Calculate the average reaction enthalpies (ΔH) for the reactions as:
            ΔH_rxn_avg = ΔH_rxn_298 + sum(nu_i * Cp_IG_i * (T - 298)) for all components i

        Parameters
        ----------
        temperature : Temperature
            The temperature at which to calculate the average reaction enthalpies (ΔH) for the reactions
            in the reactor.

        Returns
        -------
        np.ndarray
            An array of average reaction enthalpies (ΔH) for the reactions in the gas-phase batch reactor, calculated at the specified temperature.
        """
        # NOTE: check availability of data

        # res
        dH_rxns = []

        # iterate over reactions
        for rxn, dH_rxn_298, dCp_mix in zip(self.reactions, self.dH_rxns_298, self.dCp_rxns):
            # nu
            nu = rxn.reaction_stoichiometry_matrix
            nu = np.array(nu, dtype=float)

            # calculate average reaction enthalpy using the formula:
            # ΔH_rxn_avg = ΔH_rxn_298 + dCp_mix * (T - 298)
            T = to_K(temperature.value, temperature.unit)
            dH_rxn_avg = dH_rxn_298 + dCp_mix * (T - self.T_ref_K)

            # >> check
            if dH_rxn_avg is None:
                raise ValueError(
                    f"Failed to calculate average reaction enthalpy for reaction: {rxn.name}")

            dH_rxns.append(dH_rxn_avg)

        # NOTE: convert to numpy array
        res = np.array(dH_rxns, dtype=float)

        return res

    # SECTION: Property configuration methods
    # ! set enthalpy of formation unit to J/mol

    def _config_EnFo_IG_unit(
            self,
    ) -> Dict[str, float]:
        """
        Configure the unit for the enthalpy of formation at ideal gas at 298 K (EnFo_IG_298) for the components in the batch reactor based on the model source and reactor configuration.

        Returns
        -------
        Dict[str, float]
            A dictionary where the keys are component IDs and the values are the enthalpy of formation values for the components in J/mol.
        """
        # NOTE: extract EnFo_IG_298 at reference temperature (e.g., 298 K) and convert to output unit
        res = {}

        # normalized to unit
        for comp in self.component_ids:
            dt_src = self.EnFo_IG_298_src.get(comp)
            if dt_src is None:
                raise ValueError(
                    f"No EnFo_IG_298 source found for component: {comp}")

            # >> extract EnFo_IG_298 value
            dt_value = dt_src.get("value")
            dt_unit = dt_src.get("unit")

            # >> check
            if dt_value is None or dt_unit is None:
                raise ValueError(
                    f"Failed to extract EnFo_IG_298 value or unit for component: {comp}")

            # >> convert to output unit
            dt_value_converted = to_J_per_mol(
                value=dt_value,
                unit=dt_unit
            )

            res[comp] = dt_value_converted

        return res

    # ! set molecular weight unit to g/mol
    def _config_MW_unit(
            self,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Configure the unit for the molecular weight (MW) for the components in the batch reactor based on the model source and reactor configuration.

        Returns
        -------
        Tuple[np.ndarray, Dict[str, float]]
            A tuple containing:
            - An array of molecular weight values for the components in g/mol.
            - A dictionary where the keys are component IDs and the values are the molecular weight values for the components in g/mol.
        """
        # NOTE: extract MW at reference temperature (e.g., 298 K) and convert to output unit
        res = []
        res_comp = {}

        # normalized to unit
        for comp in self.component_ids:
            dt_src = self.MW_src.get(comp)
            if dt_src is None:
                raise ValueError(
                    f"No MW source found for component: {comp}")

            # >> extract MW value
            dt_value = dt_src.get("value")
            dt_unit = dt_src.get("unit")

            # >> check
            if (
                dt_value is None or
                dt_unit is None
            ):
                raise ValueError(
                    f"Failed to extract MW value or unit for component: {comp}")

            # >> convert to output unit
            dt_value_converted = to_g_per_mol(
                value=dt_value,
                unit=dt_unit
            )

            res_comp[comp] = dt_value_converted
            res.append(dt_value_converted)

        # convert to numpy array
        res = np.array(res, dtype=float)

        return res, res_comp

    # SECTION: EOS related methods
    # ! Calculate total pressure using ideal gas law
    def calc_tot_pressure(
            self,
            n_total: float,
            temperature: float,
            reactor_volume_value: float,
            R: float,
            gas_model: GasModel
    ) -> float:
        """
        Total pressure [Pa].
        Default: ideal gas
            P = N_total * R * T / V

        Parameters
        ----------
        n_total : float
            Total moles of gas in the reactor.
        temperature : float
            Temperature of the gas in the reactor [K].
        reactor_volume_value : float
            Volume of the reactor [m3].
        R : float
            Ideal gas constant [J/mol.K].
        gas_model : GasModel
            The gas model to use for the calculation (e.g., "ideal", "real").

        Returns
        -------
        float
            Total pressure of the gas in the reactor [Pa].
        """
        if gas_model == "real":
            # FIXME: implement real gas model
            return 0

        # ideal gas model
        return n_total * R * temperature / float(reactor_volume_value)

    # ! Calculate volume
    def calc_gas_volume(
        self,
        n_total: float,
        temperature: float,
        pressure: float,
        R: float,
        gas_model: GasModel
    ) -> float:
        """
        Calculate the volume of the gas in the reactor using the ideal gas law.
            V = N_total * R * T / P

        Parameters
        ----------
        n_total : float
            Total moles of gas in the reactor.
        temperature : float
            Temperature of the gas in the reactor [K].
        pressure : float
            Pressure of the gas in the reactor [Pa].
        R : float
            Ideal gas constant [J/mol.K].
        gas_model : GasModel
            The gas model to use for the calculation (e.g., "ideal", "real").

        Returns
        -------
        float
            Volume of the gas in the reactor [m3].
        """
        if gas_model == "real":
            # FIXME: implement real gas model
            return 0

        # ideal gas model
        return n_total * R * temperature / pressure

    def calc_liquid_volume(
            self,
            n: np.ndarray,
            molecular_weights: np.ndarray,
            density: np.ndarray
    ) -> float:
        """
        Calculate the volume of the liquid in the reactor using the formula:
            V = sigma_i (n_i * MW_i) / density_i

        Parameters
        ----------
        n : np.ndarray
            An array of moles of each component in the liquid phase.
        molecular_weights : np.ndarray
            An array of molecular weights for each component in the liquid phase [g/mol].
        density : np.ndarray
            An array of densities for each component in the liquid phase [g/m3].
        """
        # calculate volume for each component
        volumes = n * molecular_weights / density

        # total volume is the sum of the volumes of each component
        total_volume = np.sum(volumes)

        return total_volume

    def calc_rho_LIQ(
            self,
            temperature: Temperature,
    ):
        """
        Calculate the density of the liquid either using a constant value from model inputs or using a temperature-dependent equation of state.

        Parameters
        ----------
        temperature : Temperature
            The temperature at which to calculate the liquid density.

        Returns
        -------
        np.ndarray
            An array of density values for the liquid phase, calculated at the specified temperature.
        """
        # NOTE: calculate density based on the density mode
        if self.liquid_density_mode == "temperature-dependent":
            # NOTE: calculate temperature-dependent density
            rho_LIQ_values = self.calc_rho_LIQ_real(
                inputs={
                    "T": temperature
                },
            )
        elif self.liquid_density_mode == "constant":
            # NOTE: use constant density from model inputs
            rho_LIQ_values = self.liquid_density_constant_values
        else:
            raise ValueError(
                f"Invalid density_mode '{self.liquid_density_mode}'. Must be 'temperature-dependent' or 'constant'."
            )

        # >> check density values
        if rho_LIQ_values is None:
            raise ValueError(
                "Density values could not be calculated or retrieved."
            )

        return rho_LIQ_values

    def calc_rho_LIQ_real(
            self,
            inputs: Dict[str, Any],
    ) -> np.ndarray:
        # NOTE: extract density at reference temperature (e.g., 298 K)
        rho_LIQ_ref: Dict[str, CustomProperty] = {}
        for comp in self.component_ids:
            eq_src = self.rho_LIQ_src.get(comp)
            if eq_src is None:
                raise ValueError(
                    f"No density source found for component: {comp}"
                )

            # >> execute equation to get density at reference conditions
            rho_value = exec_component_eq(
                component_eq_src=eq_src,
                inputs=inputs,
                output_unit="g/m3"
            )
            if rho_value is None:
                raise ValueError(
                    f"Failed to extract density value for component: {comp}"
                )

            # store
            rho_LIQ_ref[comp] = rho_value

        # NOTE: build a list of density values for all components
        rho_LIQ_values = []

        # iterate over components
        for comp in self.component_ids:
            rho_value = rho_LIQ_ref.get(comp)
            if rho_value is None:
                raise ValueError(
                    f"No density value found for component: {comp}")
            rho_LIQ_values.append(rho_value.value)

        # >> to numpy
        rho_LIQ_values = np.array(rho_LIQ_values, dtype=float)

        return rho_LIQ_values
