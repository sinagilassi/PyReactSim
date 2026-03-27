# import libs
import logging
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, cast
from pythermodb_settings.models import Component, Temperature, Pressure, ComponentKey, CustomProperty
from pyThermoLinkDB.thermo import Source
# locals
from .br import BatchReactor
from .thermo_source import ThermoSource
from ..models.br import BatchReactorOptions
from ..models.rate_exp import ReactionRateExpression
from ..utils.unit_tools import to_m3, to_Pa, to_K
from ..utils.thermo_tools import calc_total_heat_capacity, calc_rxn_heat_generation
from ..utils.opt_tools import calc_heat_exchange, set_component_X
from ..models.br import GasModel

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
            reactor_inputs=reactor_inputs,
            reaction_rates=reaction_rates,
            component_key=component_key
        )

        # ! N: initial mole
        _, self._N0 = set_component_X(
            components=components,
            prop_name="mole",
            component_key=component_key
        )

        # SECTION: Model inputs
        self.model_inputs = model_inputs
        # >> temperature
        if "temperature" in model_inputs:
            self.temperature: Temperature = model_inputs["temperature"]
            self.temperature_value = to_K(
                self.temperature.value,
                self.temperature.unit
            )
            # >> update
            self.temperature = Temperature(
                value=self.temperature_value,
                unit="K"
            )
        else:
            raise ValueError("Temperature must be provided in model_inputs.")

        # >> pressure
        if "pressure" in model_inputs:
            self.pressure: Pressure = model_inputs["pressure"]
        else:
            raise ValueError("Pressure must be provided in model_inputs.")

        # SECTION: GasBatchReactor-specific properties
        self.reactor_inputs = reactor_inputs
        # >> extract
        self.phase = "gas"
        self.gas_model: GasModel = reactor_inputs.gas_model
        self.heat_transfer_mode = reactor_inputs.heat_transfer_mode
        self.volume_mode = reactor_inputs.volume_mode
        self.jacket_temperature = reactor_inputs.jacket_temperature
        self.heat_transfer_coefficient = reactor_inputs.heat_transfer_coefficient
        self.heat_transfer_area = reactor_inputs.heat_transfer_area
        self.heat_capacity_mode = reactor_inputs.heat_capacity_mode

        # NOTE: heat transfer mode
        if self.heat_transfer_mode == "isothermal":
            # fixed temperature in K
            self.temperature_fixed = self.temperature_value
            # set T0
            self._T0 = self.temperature_value
        elif self.heat_transfer_mode == "non-isothermal":
            self.temperature_fixed = None
            self._T0 = self.temperature_value
        else:
            raise ValueError(
                "Invalid heat_transfer_mode. Must be 'isothermal' or 'non-isothermal'."
            )

        # >> heat exchange
        self.heat_exchange = False
        if (
            self.jacket_temperature is not None and
            self.heat_transfer_coefficient is not None and
            self.heat_transfer_area is not None and
            self.heat_transfer_mode == 'non-isothermal'
        ):
            self.heat_exchange = True

            # >> conversions for heat exchange parameters
            self.jacket_temperature = Temperature(
                value=to_K(
                    self.jacket_temperature.value,
                    self.jacket_temperature.unit
                ),
                unit="K"
            )
            # >>> set
            self.jacket_temperature_value = self.jacket_temperature.value

            # >> heat transfer coefficient [W/m2.K]
            self.heat_transfer_coefficient_value = self.heat_transfer_coefficient.value

            # >> heat transfer area [m2]
            self.heat_transfer_area_value = self.heat_transfer_area.value

        # NOTE: Validate options
        if reactor_inputs.reactor_volume is None:
            raise ValueError(
                "reactor_volume must be provided for constant volume mode."
            )
        # >> set
        self.reactor_volume = reactor_inputs.reactor_volume
        self.reactor_volume_value = to_m3(
            self.reactor_volume.value,
            self.reactor_volume.unit
        )

        # SECTION: Reaction rates
        self.reaction_rates = reaction_rates
        # >> build reactions
        self.reactions = self.build_reactions()
        # >> extract stoichiometry matrix
        self.stoichiometry_matrix = self.build_stoichiometry()

        # SECTION: Thermodynamic properties
        # ! Ideal Gas Heat Capacity at reference temperature (e.g., 298 K)
        # ! Ideal Gas Enthalpy of formation at 298 K

    # SECTION: Properties
    @property
    def N0(self) -> np.ndarray:
        if self._N0 is None:
            raise ValueError("N0 has not been set.")
        return self._N0

    @N0.setter
    def N0(self, value: np.ndarray):
        self._N0 = value

    @property
    def T0(self) -> float:
        if self._T0 is None:
            raise ValueError("T0 has not been set.")
        return self._T0

    @T0.setter
    def T0(self, value: float):
        self._T0 = value

    # SECTION: Build initial value for n and T
    def build_y0(self) -> np.ndarray:
        # NOTE: initial moles
        n0 = self.N0

        # NOTE: initial temperature
        if self.heat_transfer_mode == "isothermal":
            T0 = self.temperature_fixed
        else:
            T0 = self.temperature_value

        # NOTE: build initial value vector
        if self.heat_transfer_mode == "isothermal":
            # state vector: [n1, n2, ..., nNc]
            y0 = n0
        else:
            # state vector: [n1, n2, ..., nNc, T]
            y0 = np.concatenate([n0, np.array([T0], dtype=float)])

        return y0

    # SECTION: ODE system for solve_ivp
    def rhs(
            self,
            t: float,
            y: np.ndarray
    ) -> np.ndarray:
        """
        Right-hand side for solve_ivp.

        State vector:
        - isothermal: [n1, n2, ..., nNc]
        - non-isothermal: [n1, n2, ..., nNc, T]

        Parameters
        ----------
        t : float
            Current time in the simulation (in seconds).
        y : np.ndarray
            Current state vector of the system, which includes the moles of each component and, if applicable, the temperature.
        """
        ns = self.component_num

        if self.heat_transfer_mode == "isothermal":
            if self.temperature_fixed is None:
                raise ValueError(
                    "temperature_fixed must be provided for isothermal simulation.")
            n = y[:ns]
            temp = float(self.temperature_fixed)
        else:
            n = y[:ns]
            temp = float(y[ns])

        # NOTE: Calculate total moles
        n_total = np.sum(n)
        n_total = max(n_total, 1e-30)

        # Calculate partial pressures
        y_mole = n / n_total

        # ! calculate concentration: C_i = n_i / V
        (
            concentration,
            concentration_std,
            concentration_total
        ) = self._calc_concentration(
            n=n,
            reactor_volume=self.reactor_volume_value
        )

        # ! calculate total pressure using ideal gas law: P = N_total * R * T / V
        # ! unit check: N_total [mol], R [J/mol.K], T [K], V [m3] => P [Pa]
        (
            _,
            partial_pressures_std,
            p_total
        ) = self._calc_partial_pressure(
            n_total=n_total,
            y_mole=y_mole,
            T=temp
        )

        # NOTE: Calculate Reaction rates for each component (partial pressures and temperature)
        # ! r_k = k(T, P_i) for each reaction k
        rates = self._calc_rates(
            partial_pressures=partial_pressures_std,
            temperature=Temperature(value=temp, unit="K"),
            pressure=Pressure(value=p_total, unit="Pa")
        )

        # NOTE: Species balances:
        # ! dn_i/dt = V * Σ_k ν_i,k * r_k
        dn_dt = self._build_dn_dt(
            ns=ns,
            rates=rates
        )

        # >>> calculate dn/dt for isothermal case
        if self.heat_transfer_mode == "isothermal":
            return dn_dt

        # NOTE: Energy balance:
        # ! (Σ_i n_i Cp_i) dT/dt = V Σ_k [(-ΔH_k) r_k] + UA (T_s - T)
        # >>> calculate dT/dt
        dT_dt = self._build_dT_dt(
            n=n,
            rates=rates,
            temp=temp
        )

        # >>> calculate both dn/dt and dT/dt
        return np.concatenate([dn_dt, np.array([dT_dt], dtype=float)])

    # SECTION: Calculate rates
    def _calc_rates(
        self,
        partial_pressures: Dict[str, CustomProperty],
        temperature: Temperature,
        pressure: Pressure
    ):
        """
        Calculate reaction rates for each reaction based on the current partial pressures and temperature.

        Parameters
        ----------
        partial_pressures : Dict[str, CustomProperty]
            Partial pressure of the components in the reactor (in Pa).
        temperature : Temperature
            Current temperature of the system (in K).
        pressure : Pressure
            Total pressure of the system (in Pa).

        Returns
        -------
        np.ndarray
            An array of reaction rates for each reaction in the reactor, calculated based on the current partial pressures and temperature.
        """
        # ! r_k = k(T, P_i) for each reaction k
        rates = []

        # iterate over reaction rate expressions
        for rxn_name, rate_exp in self.reaction_rates.items():
            # >> calculate rate for reaction
            r_k = rate_exp.calc(
                xi=partial_pressures,
                temperature=temperature,
                pressure=pressure
            )

            # extract rate value
            r_k_value = r_k.value
            # append to rates list
            rates.append(r_k_value)

        # >> to array
        rates = np.array(rates, dtype=float)

        return rates

    # SECTION: Building dn/dt
    def _build_dn_dt(
            self,
            ns: int,
            rates: np.ndarray
    ):
        """
        Build the rate of change of moles (dn/dt) for each component based on the reaction rates and stoichiometry.

        Parameters
        ----------
        ns : int
            Number of components in the reactor.
        rates : np.ndarray
            Array of reaction rates for each reaction in the reactor.

        Returns
        -------
        np.ndarray
            An array of the rate of change of moles (dn/dt) for each component in the reactor, calculated based on the reaction rates and stoichiometry.
        """
        # ! dn_i/dt = V * Σ_k ν_i,k * r_k
        dn_dt = np.zeros(ns, dtype=float)
        name_to_idx = self.component_id_to_index

        for k, rxn in enumerate(self.reactions):
            # > calculate reaction rate for reaction k
            r_k = rates[k]

            # > reaction stoichiometry for reaction k: ν_i,k
            stoich_k = rxn.reaction_stoichiometry_source[
                self.component_key
            ].items()

            # >> calculate dn/dt for each component i based on reaction k
            for sp_name, nu_ik in stoich_k:
                i = name_to_idx[sp_name]
                dn_dt[i] += self.reactor_volume_value * nu_ik * r_k

        return dn_dt

    # SECTION: Building dT/dt
    def _build_dT_dt(
            self,
            n: np.ndarray,
            rates: np.ndarray,
            temp: float
    ):
        """
        Calculate the rate of change of temperature (dT/dt) based on the energy balance for the non-isothermal gas-phase batch reactor.

        Parameters
        ----------
        n : np.ndarray
            Array of moles of each component in the reactor.
        rates : np.ndarray
            Array of reaction rates for each reaction in the reactor.
        temp : float
            Current temperature of the system (in K).

        Returns
        -------
        float
            The rate of change of temperature (dT/dt) for the non-isothermal gas-phase batch reactor.
        """
        # ! (Σ_i n_i Cp_i) dT/dt = V Σ_k [(-ΔH_k) r_k] + UA (T_s - T)
        # ??? Cp_i(T)
        Cp_IG_values = self.calc_Cp_IG(
            inputs={
                "T": temp
            },
            Cp_IG_src=self.Cp_IG_src,
            output_unit="J/mol.K"
        )

        # ??? Σ_i n_i Cp_i
        c_total = calc_total_heat_capacity(n, Cp_IG_values)

        if c_total <= 1e-16:
            raise ValueError("Total heat capacity is too small or zero.")

        # ! calculate heat generated by reactions: Q_rxn = V Σ_k [(-ΔH_k) r_k]
        # V[m3], ΔH[J/mol], r[mol/m3.s] => Q_rxn [J/s] or [W]
        # ??? ΔH_k
        delta_h = self.calc_dH_rxns(
            temperature=Temperature(value=temp, unit="K")
        )

        # ??? Q_rxn
        q_rxn = calc_rxn_heat_generation(
            delta_h=delta_h,
            rates=rates,
            reactor_volume=self.reactor_volume_value
        )

        # ! calculate heat exchange with surroundings: Q_exchange = UA (T_s - T)
        # ??? Q_exchange
        q_exchange = 0.0

        # >>> check if heat exchange is enabled
        if self.heat_exchange:
            q_exchange = calc_heat_exchange(
                temperature=temp,
                jacket_temperature=self.jacket_temperature_value,
                heat_transfer_area=self.heat_transfer_area_value,
                heat_transfer_coefficient=self.heat_transfer_coefficient_value
            )

        # >>> calculate dT/dt
        dT_dt = (q_rxn + q_exchange) / c_total

        return dT_dt

    # SECTION: Building xi (partial pressure)
    def _calc_partial_pressure(
        self,
        n_total: float,
        y_mole: np.ndarray,
        T: float,
    ):
        """
        Calculate the partial pressures of the components based on the total moles, mole fractions, and temperature.

        Parameters
        ----------
        n_total : float
            Total moles of all components in the reactor.
        y_mole : np.ndarray
            Mole fractions of the components in the reactor.
        T : float
            Current temperature of the system (in K).
        component_key : ComponentKey
            The key to be used for the components in the model source, which is necessary for mapping the component names to their corresponding properties in the model source.

        Returns
        -------
        Tuple[Dict[str, CustomProperty], Dict[str, CustomProperty], float]
            A tuple containing:
            - A dictionary of partial pressures for each component (in Pa).
            - A dictionary of partial pressures for each component as CustomProperty objects (in Pa).
            - The total pressure of the system (in Pa).
        """
        # ! calculate total pressure using ideal gas law: P = N_total * R * T / V
        # ! unit check: N_total [mol], R [J/mol.K], T [K], V [m3] => P [Pa]
        # NOTE: calculate total pressure
        p_total = self.calc_tot_pressure(
            n_total=n_total,
            temperature=T,
            reactor_volume_value=self.reactor_volume_value,
            R=self.R,
            gas_model=self.gas_model
        )

        # NOTE: partial pressures:
        # ! P_i = y_i * P_total
        partial_pressures = {
            sp: y_mole[i] * p_total for i, sp in enumerate(self.component_formula_state)
        }

        # >> std partial pressures
        partial_pressures_std = {}

        # iterate over partial pressures and convert to CustomProperty with unit "Pa"
        for k, v in partial_pressures.items():
            partial_pressures_std[k] = CustomProperty(
                value=v,
                unit="Pa",
                symbol="P"
            )

        return partial_pressures, partial_pressures_std, p_total

    def _calc_concentration(
            self,
            n: np.ndarray,
            reactor_volume: float
    ) -> Tuple[np.ndarray, Dict[str, CustomProperty], float]:
        """
        Calculate the concentration of each component in the reactor based on the moles and reactor volume.

        Parameters
        ----------
        n : np.ndarray
            Array of moles of each component in the reactor.
        reactor_volume : float
            Volume of the reactor (in m3).

        Returns
        -------
        Tuple[np.ndarray, Dict[str, CustomProperty], float]
            A tuple containing:
            - An array of concentrations for each component (in mol/m3).
            - A dictionary of concentrations for each component as CustomProperty objects (in mol/m3).
            - The total concentration of the system (in mol/m3).
        """
        # ! C_i = n_i / V
        # unit check: n_i [mol], V [m3] => C_i [mol/m3]
        concentration = n / reactor_volume

        # total concentration
        # ! C_total = N_total / V
        n_total = np.sum(n)
        concentration_total = n_total / reactor_volume

        # NOTE: create ids for concentration array
        conc_ids = [
            sp for sp in self.component_formula_state
        ]

        # std concentration as dict
        concentration_std = {
            sp: CustomProperty(
                value=conc,
                unit="mol/m3",
                symbol="C"
            ) for sp, conc in zip(conc_ids, concentration)
        }

        return concentration, concentration_std, concentration_total
