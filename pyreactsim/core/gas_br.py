# import libs
import logging
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, cast
from pythermodb_settings.models import Component, Temperature, Pressure, ComponentKey, CustomProperty
from pyThermoLinkDB.thermo import Source
from pyThermoLinkDB.models.component_models import ComponentEquationSource
from pyThermoCalcDB.reactions.reactions import dH_rxn_STD
# locals
from .br import BatchReactor
from .thermo_source import ThermoSource
from ..models.br import BatchReactorOptions
from ..models.rate_exp import ReactionRateExpression
from ..utils.unit_tools import to_m3, to_Pa, to_K
from ..utils.reaction_tools import stoichiometry_mat
from ..utils.thermo_tools import calc_total_heat_capacity, calc_rxn_heat_generation
from ..sources.interface import exec_component_eq, ext_components_dt

# NOTE: logger setup
logger = logging.getLogger(__name__)


class GasBatchReactor(BatchReactor, ThermoSource):
    """
    GasBatchReactor class for simulating chemical reactions in a gas-phase batch reactor setup. This class inherits from the BatchReactor class and is specifically designed to handle gas-phase reactions, incorporating properties and methods relevant to gas-phase systems.

    Assumptions
    -----------
    - Constant heat capacity (Cp)
    """
    # NOTE: Attributes

    def __init__(
        self,
        components: List[Component],
        source: Source,
        model_inputs: Dict[str, Any],
        reactor_inputs: BatchReactorOptions,
        reaction_rates: Dict[str, ReactionRateExpression],
        component_key: ComponentKey,
        **kwargs
    ):
        """
        Initializes the GasBatchReactor instance with the provided components, source, and component key.

        Parameters
        ----------
        components : List[Component]
            A list of Component objects representing the chemical components involved in the gas-phase batch reactor.
        source : Source
            A Source object containing information about the source of the data or equations used in the gas-phase batch reactor simulations.
        model_inputs : Dict[str, Any]
            A dictionary containing the model inputs for the gas-phase batch reactor simulations, such as temperature, pressure, and initial mole numbers.
        reactor_inputs : BatchReactorOptions
            A BatchReactorOptions object containing the options and parameters specific to the gas-phase batch reactor setup, such as heat transfer mode, volume mode, and gas model.
        reaction_rates : Dict[str, ReactionRateExpression]
            A dictionary containing the reaction rate expressions for the reactions occurring in the gas-phase batch reactor,
            where the keys are the names of the reactions and the values are ReactionRateExpression objects.
        component_key : ComponentKey
            A ComponentKey object representing the key to be used for the components in the model source.
        **kwargs
            Additional keyword arguments that can be passed to the initialization of the GasBatchReactor instance.
        """
        # LINK: Initialize the parent BatchReactor class
        BatchReactor.__init__(
            self,
            components=components,
            source=source,
            component_key=component_key
        )
        # LINK: Initialize the parent ThermoSource class
        ThermoSource.__init__(
            self,
            components=components,
            source=source,
            component_key=component_key
        )

        # SECTION: Model inputs
        self.model_inputs = model_inputs
        # >> temperature
        if "temperature" in model_inputs:
            self.temperature: Temperature = model_inputs["temperature"]

        # >> pressure
        if "pressure" in model_inputs:
            self.pressure: Pressure = model_inputs["pressure"]

        # SECTION: GasBatchReactor-specific properties
        self.reactor_inputs = reactor_inputs
        # >> extract
        self.phase = "gas"
        self.gas_model = reactor_inputs.gas_model
        self.heat_transfer_mode = reactor_inputs.heat_transfer_mode
        self.volume_mode = reactor_inputs.volume_mode
        self.jacket_temperature = reactor_inputs.jacket_temperature
        self.heat_transfer_coefficient = reactor_inputs.heat_transfer_coefficient
        self.heat_transfer_area = reactor_inputs.heat_transfer_area
        self.heat_capacity_mode = reactor_inputs.heat_capacity_mode

        # >> heat exchange
        self.heat_exchange = False
        if (
            self.jacket_temperature is not None and
            self.heat_transfer_coefficient is not None and
            self.heat_transfer_area is not None
        ):
            self.heat_exchange = True

        # NOTE: Validate options
        if reactor_inputs.reactor_volume is None:
            raise ValueError(
                "reactor_volume must be provided for constant volume mode."
            )
        # >> set
        self.reactor_volume = reactor_inputs.reactor_volume
        self.reactor_volume_value = to_m3(
            self.reactor_volume.value, self.reactor_volume.unit)

        # SECTION: Reaction rates
        self.reaction_rates = reaction_rates
        # >> build reactions
        self.reactions = self.build_reactions()
        # >> extract stoichiometry matrix
        self.stoichiometry_matrix = self.build_stoichiometry()

        # SECTION: Thermodynamic properties
        # ! Ideal Gas Heat Capacity at reference temperature (e.g., 298 K)
        self.Cp_IG_src = self.prop_eq_src(prop_name="Cp_IG")
        # >> heat capacity mode
        if self.heat_capacity_mode == "constant":
            self.Cp_IG_values = self.calc_Cp_IG()

        # ! Ideal Gas Enthalpy of formation at 298 K
        self.EnFo_IG_298_src = self.prop_dt_src(
            component_ids=self.component_ids,
            prop_name="EnFo_IG"
        )

    def calc_Cp_IG(
            self,
            inputs: Optional[Dict[str, Any]] = None,
            output_unit: Optional[str] = "J/mol.K"
    ) -> np.ndarray:
        # NOTE: check if Cp_IG is constant or variable
        if self.heat_capacity_mode == "constant":
            inputs = {
                "T": to_K(self.temperature.value, self.temperature.unit)
            }
        else:
            if inputs is None:
                raise ValueError(
                    "inputs must be provided for variable heat capacity mode.")

        # NOTE: extract Cp_IG at reference temperature (e.g., 298 K)
        Cp_IG_ref: Dict[str, CustomProperty] = {}
        for comp in self.component_ids:
            eq_src = self.Cp_IG_src.get(comp)
            if eq_src is None:
                raise ValueError(
                    f"No Cp_IG source found for component: {comp}")

            # >> execute equation to get Cp_IG at reference conditions
            cp_value = exec_component_eq(
                component_eq_src=eq_src,
                inputs=inputs,
                output_unit=output_unit
            )
            if cp_value is None:
                raise ValueError(
                    f"Failed to extract Cp_IG value for component: {comp}")
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

    def calc_dH_rxns(
            self,
            temperature: Temperature
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

    def build_reactions(self):
        """
        Build the list of Reaction objects for the gas-phase batch reactor using the provided reaction rates and components.
        """
        reactions = []
        for rxn_name, rate_exp in self.reaction_rates.items():
            rxn = rate_exp.reaction
            reactions.append(rxn)
        return reactions

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

    def calc_tot_pressure(self, n_total: float, temperature: float) -> float:
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

        Returns
        -------
        float
            Total pressure of the gas in the reactor [Pa].
        """
        if self.gas_model == "real":
            # FIXME: implement real gas model
            return 0

        # ideal gas model
        return n_total * self.R * temperature / float(self.reactor_volume_value)

    # SECTION: ODE system for solve_ivp
    def rhs(
            self,
            t: float,
            y: np.ndarray,
            temperature_fixed: Optional[float] = None
    ) -> np.ndarray:
        """
        Right-hand side for solve_ivp.

        State vector:
        - isothermal: [n1, n2, ..., nNc]
        - non-isothermal: [n1, n2, ..., nNc, T]
        """
        ns = self.component_num

        if self.heat_transfer_mode == "isothermal":
            if temperature_fixed is None:
                raise ValueError(
                    "temperature_fixed must be provided for isothermal simulation.")
            n = y[:ns]
            temp = float(temperature_fixed)
        else:
            n = y[:ns]
            temp = float(y[ns])

        # Calculate total moles
        n_total = np.sum(n)
        n_total = max(n_total, 1e-30)

        # Calculate partial pressures
        y_mole = n / n_total
        p_total = self.calc_tot_pressure(n_total=n_total, temperature=temp)
        partial_pressures = {
            sp: y_mole[i] * p_total for i, sp in enumerate(self.component_ids)
        }

        # NOTE: Calculate Reaction rates for each component (partial pressures and temperature)
        # FIXME
        rates = self.stoichiometry_matrix

        # NOTE: Species balances:
        # ! dn_i/dt = V * Σ_k ν_i,k * r_k
        dn_dt = np.zeros(ns, dtype=float)
        name_to_idx = self.component_id_to_index

        for k, rxn in enumerate(self.reactions):
            r_k = rates[k]
            for sp_name, nu_ik in rxn.stoich.items():
                i = name_to_idx[sp_name]
                dn_dt[i] += self.reactor_volume_value * nu_ik * r_k

        # >>> calculate dn/dt for isothermal case
        if self.heat_transfer_mode == "isothermal":
            return dn_dt

        # NOTE: Energy balance:
        # ! (Σ_i n_i Cp_i) dT/dt = V Σ_k [(-ΔH_k) r_k] + UA (T_s - T)
        c_total = calc_total_heat_capacity(n, self.Cp_IG_values)

        if c_total <= 1e-16:
            raise ValueError("Total heat capacity is too small or zero.")

        # ! calculate heat generated by reactions: Q_rxn = V Σ_k [(-ΔH_k) r_k]
        # V[m3], ΔH[J/mol], r[mol/m3.s] => Q_rxn [J/s] or [W]
        q_rxn = calc_rxn_heat_generation(
            delta_h=self.calc_dH_rxns(
                temperature=Temperature(value=temp, unit="K")
            ),
            rates=rates,
            reactor_volume=self.reactor_volume_value
        )

        # ! calculate heat exchange with surroundings: Q_exchange = UA (T_s - T)
        q_exchange = 0.0
        if self.heat_exchange:
            q_exchange = self.calc_heat_exchange(temp=temp)

        # >>> calculate dT/dt
        dT_dt = (q_rxn + q_exchange) / c_total

        # >>> calculate both dn/dt and dT/dt
        return np.concatenate([dn_dt, np.array([dT_dt], dtype=float)])

    def calc_heat_exchange(self, temp: float) -> float:
        """
        Calculate the heat exchange with the surroundings based on the current temperature of the system.

        Parameters
        ----------
        temp : float
            Current temperature of the system [K].

        Returns
        -------
        float
            Heat exchange with the surroundings [W].
        """
        if not self.heat_exchange:
            return 0.0

        # NOTE: check if all required parameters for heat exchange are available
        if self.jacket_temperature is None or self.heat_transfer_coefficient is None or self.heat_transfer_area is None:
            raise ValueError(
                "Jacket temperature, heat transfer coefficient, and heat transfer area must be provided for heat exchange calculation."
            )

        # NOTE: Convert units if necessary
        T_s = to_K(self.jacket_temperature.value, self.jacket_temperature.unit)
        A = to_m3(self.heat_transfer_area.value, self.heat_transfer_area.unit)
        U = self.heat_transfer_coefficient.value  # Assuming it's already in W/m^2.K

        # ! calculate heat exchange using the formula: Q = U * A * (T_s - T)
        # unit check: U [W/m^2.K], A [m^2], T_s [K], temp [K] => Q [W] or [J/s]
        return U * A * (T_s - temp)
