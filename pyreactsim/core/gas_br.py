# import libs
import logging
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, cast
from pythermodb_settings.models import Component, Temperature, Pressure, ComponentKey
from pyThermoLinkDB.thermo import Source
# locals
from .br import BatchReactor
from ..models.br import BatchReactorOptions
from ..models.rate_exp import ReactionRateExpression
from ..utils.unit_tools import to_m3, to_Pa, to_K

# NOTE: logger setup
logger = logging.getLogger(__name__)


class GasBatchReactor(BatchReactor):
    """
    GasBatchReactor class for simulating chemical reactions in a gas-phase batch reactor setup. This class inherits from the BatchReactor class and is specifically designed to handle gas-phase reactions, incorporating properties and methods relevant to gas-phase systems.
    """

    def __init__(
        self,
        components: List[Component],
        source: Source,
        component_key: ComponentKey,
        options: BatchReactorOptions,
        reaction_rates: Dict[str, ReactionRateExpression]
    ):
        """
        Initializes the GasBatchReactor instance with the provided components, source, and component key.

        Parameters
        ----------
        components : List[Component]
            A list of Component objects representing the chemical components involved in the gas-phase batch reactor.
        source : Source
            A Source object containing information about the source of the data or equations used in the gas-phase batch reactor simulations.
        component_key : ComponentKey
            A ComponentKey object that serves as a key for identifying and categorizing the components in the gas-phase batch reactor.
        """
        # LINK: Initialize the parent BatchReactor class
        super().__init__(components, source, component_key)

        # SECTION: GasBatchReactor-specific properties
        self.options = options
        # >> extract
        self.phase = "gas"
        self.gas_model = options.gas_model
        self.heat_transfer_mode = options.heat_transfer_mode
        self.volume_mode = options.volume_mode
        self.jacket_temperature = options.jacket_temperature
        self.heat_transfer_coefficient = options.heat_transfer_coefficient
        self.heat_transfer_area = options.heat_transfer_area

        # NOTE: Validate options
        if options.reactor_volume is None:
            raise ValueError(
                "reactor_volume must be provided for constant volume mode."
            )
        # >> set
        self.reactor_volume = options.reactor_volume

    def pressure(self, n_total: float, temperature: float) -> float:
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
        return n_total * self.R * temperature / float(self.reactor_volume.value)

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
        p_total = self.pressure(n_total=n_total, temperature=temp)
        partial_pressures = {
            sp: y_mole[i] * p_total for i, sp in enumerate(self.component_ids)
        }

        # Reaction rates
        rates = np.array([rxn.rate(partial_pressures, temp)
                         for rxn in self.reactions], dtype=float)

        # NOTE: Species balances: dn_i/dt = V * Σ_k ν_i,k * r_k
        dn_dt = np.zeros(ns, dtype=float)
        name_to_idx = self.species_index()

        for k, rxn in enumerate(self.reactions):
            r_k = rates[k]
            for sp_name, nu_ik in rxn.stoich.items():
                i = name_to_idx[sp_name]
                dn_dt[i] += self.volume * nu_ik * r_k

        if self.is_isothermal:
            return dn_dt

        # Energy balance:
        #   (Σ_i n_i Cp_i) dT/dt = V Σ_k [(-ΔH_k) r_k] + UA (T_s - T)
        c_total = self.total_heat_capacity(n)

        if c_total <= 1e-16:
            raise ValueError("Total heat capacity is too small or zero.")

        q_rxn = 0.0
        for k, rxn in enumerate(self.reactions):
            q_rxn += self.volume * (-rxn.delta_h) * rates[k]

        q_exchange = 0.0
        if self.heat_exchange is not None:
            q_exchange = self.heat_exchange.ua * \
                (self.heat_exchange.t_surroundings - temp)

        dT_dt = (q_rxn + q_exchange) / c_total

        return np.concatenate([dn_dt, np.array([dT_dt], dtype=float)])
