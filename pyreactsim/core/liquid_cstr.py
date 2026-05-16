import logging
import numpy as np
from typing import Dict, List, Tuple, cast, Literal
from pythermodb_settings.models import Component, ComponentKey, CustomProp, CustomProperty, Temperature
from pyreactsim_core.models import ReactionRateExpression
# locals
from .cstrc import CSTRReactorCore
# from ..models.rate_exp import ReactionRateExpression
from ..sources.thermo_source import ThermoSource
from ..utils.opt_tools import calc_heat_exchange
from ..utils.reaction_tools import stoichiometry_mat, stoichiometry_mat_key
from ..utils.thermo_tools import calc_rxn_heat_generation
# auxiliary
from .react_aux import ReactorAuxiliary
# log
from .react_log import ReactLog

# NOTE: logger setup
logger = logging.getLogger(__name__)


class LiquidCSTRReactor(ReactorAuxiliary, ReactLog):
    """
    Liquid-phase CSTR dynamic reactor model (mole-based mass and heat balances).

    Current supported closure
    -------------------------
    - operation_mode = "constant_volume"

    Placeholders for future extension
    ---------------------------------
    - variable-volume liquid closure
    - constant-pressure liquid closure
    """

    # NOTE: reference temperature for reaction enthalpy calculations [K]
    T_ref = Temperature(value=298.15, unit="K")

    # NOTE: Primary state/configuration values
    # ! initial component moles in reactor [mol]
    _N0: np.ndarray = np.array([])
    # ! inlet component molar flow rates [mol/s]
    _F_in: np.ndarray = np.array([])
    # ! outlet total molar flow rate [mol/s]
    _F_out_total: float = 0.0
    # ! initial reactor temperature [K]
    _T0: float = 0.0
    # ! inlet stream temperature [K]
    _T_in: float = 0.0
    # ! reactor holdup volume [m3]
    _V0: float = 0.0
    # ! initial liquid density vector [g/m3]
    _rho_LIQ0: np.ndarray = np.array([])

    def __init__(
        self,
        components: List[Component],
        reaction_rates: List[ReactionRateExpression],
        thermo_source: ThermoSource,
        cstr_reactor_core: CSTRReactorCore,
        component_key: ComponentKey,
        **kwargs
    ):
        # LINK: ReactorAuxiliary initialization
        super().__init__(
            components=components,
            reaction_rates=reaction_rates,
            thermo_source=thermo_source,
            reactor_core=cstr_reactor_core,
            component_key=component_key,
        )
        ReactLog.__init__(self)

        # SECTION: Core configuration
        self.cstr_reactor_core = cstr_reactor_core

        # SECTION: Reactor mode configuration
        self.heat_transfer_mode = cstr_reactor_core.heat_transfer_mode
        self.operation_mode = cstr_reactor_core.operation_mode
        self.holdup_volume_mode = cstr_reactor_core.holdup_volume_mode

        # ! Heat transfer configuration
        self.heat_exchange = cstr_reactor_core.heat_exchange
        self.heat_transfer_coefficient_value = cstr_reactor_core.heat_transfer_coefficient_value
        self.heat_transfer_area_value = cstr_reactor_core.heat_transfer_area_value
        self.jacket_temperature_value = cstr_reactor_core.jacket_temperature_value
        # ! heat rate [W]
        self.heat_rate = cstr_reactor_core.heat_rate
        self.heat_rate_value = cstr_reactor_core.heat_rate_value

        # SECTION: operation-mode guard (constant-volume liquid CSTR only)
        self._validate_operation_mode()

        # SECTION: Reaction and stoichiometry mapping
        self.reaction_rates = reaction_rates
        self.reactions = self.thermo_source.thermo_reaction.build_reactions()
        self.reaction_stoichiometry: List[Dict[str, float]] = stoichiometry_mat_key(
            reactions=self.reactions,
            component_key=component_key
        )
        self.reaction_stoichiometry_matrix = stoichiometry_mat(
            reactions=self.reactions,
            components=self.components,
            component_key=component_key,
        )
        self.stoichiometry_matrix = self.thermo_source.thermo_reaction.stoichiometry_matrix

        # SECTION: Component references
        self.component_num = self.thermo_source.component_refs["component_num"]
        self.component_ids = self.thermo_source.component_refs["component_ids"]
        self.component_formula_state = self.thermo_source.component_refs[
            "component_formula_state"]
        self.component_mapper = self.thermo_source.component_refs["component_mapper"]
        self.component_id_to_index = self.thermo_source.component_refs["component_id_to_index"]

        # SECTION: CSTR input streams and initial holdup
        # ! initial moles [mol]
        _, self._N0 = self.cstr_reactor_core.config_initial_mole()

        # ! inlet molar flow rates [mol/s]
        _, self._F_in, self._F_in_total = self.cstr_reactor_core.config_inlet_mole_flows()

        # ! outlet total molar flow rate [mol/s]
        self._F_out_total = self.cstr_reactor_core.config_outlet_mole_flow_total()

        # ! T0: initial temperature [K]
        self.temperature_initial = self.cstr_reactor_core.temperature_initial
        self._T0 = self.cstr_reactor_core._T0

        # ! Tin: inlet temperature [K]
        self.temperature_inlet = self.cstr_reactor_core.temperature_inlet
        self._T_in = self.temperature_inlet.value

        # ! V0: fixed reactor holdup volume [m3]
        self.volume = self.cstr_reactor_core.config_reactor_volume()
        self._V0 = self.volume.value

        # ! rho_LIQ0: initial liquid density [g/m3]
        # kept to follow liquid-property estimation flow used in liquid_br.py
        self._rho_LIQ0 = self.thermo_source.calc_rho_LIQ(
            temperature=self.temperature_initial
        )

        # ! Cp_LIQ_in: inlet liquid enthalpy [J/mol]
        _, self.En_LIQ_values_in = self.thermo_source.calc_En_LIQ_ref(
            temperature=self.temperature_inlet
        )

        # ! C_in, C_in_total: initial concentrations for flow closure [mol/m3]
        self._C_in, _, self.C_in_total = self._calc_concentration(
            n=self._N0,
            reactor_volume=self._V0,
        )

        # ! Q_in: inlet volumetric flow rate [m3/s]
        self.q_in = self._F_in_total / self.C_in_total

    def _validate_operation_mode(self):
        """
        Validate liquid CSTR closure selection for current implementation scope.
        """
        if self.operation_mode == "constant_volume":
            return

        if self.operation_mode == "constant_pressure":
            raise NotImplementedError(
                "Liquid CSTR with operation_mode='constant_pressure' is not supported in this version. "
                "Use operation_mode='constant_volume'. "
                "Variable-volume/pressure-linked liquid closure is reserved for future implementation."
            )

        raise ValueError(
            f"Invalid operation mode '{self.operation_mode}' for liquid CSTR. "
            "Supported mode is 'constant_volume'."
        )

    @property
    def N0(self) -> np.ndarray:
        if self._N0 is None:
            raise ValueError("N0 has not been set.")
        return self._N0

    @property
    def F_in(self) -> np.ndarray:
        if self._F_in is None:
            raise ValueError("F_in has not been set.")
        return self._F_in

    # SECTION: Build initial value for n and T
    def build_y0(self) -> np.ndarray:
        n0 = self.N0
        T0 = self._T0

        if self.heat_transfer_mode == "isothermal":
            return n0

        return np.concatenate([n0, np.array([T0], dtype=float)])

    def rhs(self, t: float, y: np.ndarray) -> np.ndarray:
        """
        Right-hand side of the dynamic liquid CSTR ODE system.
        """
        ns = self.component_num

        # NOTE: unpack state vector
        if self.heat_transfer_mode == "isothermal":
            if self._T0 is None:
                raise ValueError(
                    "initial temperature must be provided for isothermal simulation."
                )
            n = y[:ns]
            temp = float(self._T0)
        else:
            n = y[:ns]
            temp = float(y[ns])

        # NOTE: enforce physically non-negative moles
        n = np.clip(n, 0.0, None)

        # ! total moles [mol]
        n_total = float(np.sum(n))
        n_total = max(n_total, 1e-30)
        # ! mole fractions [-]
        y_mole = n / n_total

        # ! temperature [K]
        temperature = Temperature(value=temp, unit="K")

        # ! calculate liquid density [g/m3]
        rho_LIQ = self.thermo_source.calc_rho_LIQ(
            temperature=temperature
        )

        # ! calculate system volume [m3]
        reactor_volume = self._calc_reactor_volume(
            n=n,
            rho_LIQ=rho_LIQ,
            temperature=temp,
        )

        # ! concentrations [mol/m3]
        (
            _,
            concentration_std,
            concentration_total
        ) = self._calc_concentration(
            n=n,
            reactor_volume=reactor_volume
        )

        # NOTE: evaluate reaction rates [mol/m3.s]
        # ! r_k = k(T, P_i) for each reaction k [mol/m3.s]
        rates = self._calc_rates_concentration_basis(
            concentration=concentration_std,
            temperature=temperature
        )

        # NOTE: outlet total molar flow [mol/s]
        # ! F_out,total = f(C_total, Q_out) from flow closure or specified directly
        F_out_total = self._calc_F_out_total(
            concentration_total=concentration_total
        )

        # NOTE: species balance
        # ! dn_i/dt = F_in,i - F_out,i + V_R * sum_j(nu_i,j * r_j)
        dn_dt = self._build_dn_dt(
            ns=ns,
            rates=rates,
            reactor_volume=reactor_volume,
            y_mole=y_mole,
            F_out_total=F_out_total
        )

        # >>> isothermal balance: return species derivatives only
        if self.cstr_reactor_core.is_isothermal:
            return dn_dt

        # NOTE: non-isothermal energy balance
        # ! dT/dt = (F_in,i*{hi(Tin)-hi(T)} + Q_rxn + Q_exchange + Q_constant) / sum_i(n_i * Cp_i^L)
        dT_dt = self._build_dT_dt(
            n=n,
            rates=rates,
            temp=temp,
            reactor_volume=reactor_volume
        )

        # >>> non-isothermal balance: return concatenated species and temperature derivatives
        return np.concatenate([dn_dt, np.array([dT_dt], dtype=float)])

    def _calc_reactor_volume(
        self,
        n: np.ndarray,
        rho_LIQ: np.ndarray,
        temperature: float,
    ) -> float:
        """
        Reactor-volume closure for liquid CSTR.

        Current implementation only supports fixed constant volume.
        """
        if self.operation_mode == "constant_volume":
            # ??? Constant volume assumption: V = V0
            reactor_volume = self._V0
        elif self.operation_mode == "variable_volume":
            # ??? Variable volume assumption: V = f(n_total(t), T(t), P(t))
            reactor_volume = self.thermo_source.calc_liquid_volume(
                n=n,
                molecular_weights=self.thermo_source.MW,
                density=rho_LIQ
            )
        else:
            raise ValueError(
                "Invalid operation mode '{self.operation_mode}'. Must be constant pressure or volume."
            )

        return reactor_volume

    def _build_dn_dt(
        self,
        ns: int,
        rates: np.ndarray,
        reactor_volume: float,
        y_mole: np.ndarray,
        F_out_total: float
    ) -> np.ndarray:
        """
        Build species-time derivatives for liquid CSTR mole balances.

        Balance
        -------
        dn_i/dt = F_in,i - F_out,i + V_R * sum_j(nu_i,j * r_j)
        """
        dn_dt = np.zeros(ns, dtype=float)
        name_to_idx = self.component_id_to_index

        # ! outlet component molar flows: F_out,i = y_i * F_out,total
        F_out_i = y_mole * F_out_total
        # ! flow terms: F_in,i - F_out,i
        dn_dt += (self.F_in - F_out_i)

        # ! source terms: V_R * sum_j(nu_i,j * r_j)
        for k, _ in enumerate(self.reactions):
            r_k = rates[k]
            stoich_k = self.reaction_stoichiometry[k].items()
            for sp_name, nu_ik in stoich_k:
                i = name_to_idx[sp_name]
                dn_dt[i] += reactor_volume * nu_ik * r_k

        return dn_dt

    def _build_dT_dt(
        self,
        n: np.ndarray,
        rates: np.ndarray,
        temp: float,
        reactor_volume: float
    ) -> float:
        """
        Build temperature derivative from liquid-phase CSTR energy balance.

        Energy balance
        --------------
        dT/dt = (En_in - En_out + Q_rxn + Q_exchange + Q_constant) / sum_i(n_i * Cp_i^L)
        """
        temperature = Temperature(value=temp, unit="K")

        # ! Cp_LIQ_MIX_TOTAL: total heat capacity of liquid mixture (in J/K), either calculated or constant from model source
        # ? Cp_LIQ_MIX_TOTAL = sum_i(n_i * Cp_i^L) where Cp_i^L is in J/mol.K and n_i is in mol => Cp_LIQ_MIX_TOTAL in J/K
        Cp_LIQ_MIX_TOTAL = self._calc_total_heat_capacity_liquid(
            n=n,
            temperature=temperature,
            reactor_volume=reactor_volume,
            mode=cast(
                Literal['calculate', 'constant'],
                self.Cp_LIQ_MIX_TOTAL_MODE
            ),
        )

        # NOTE: sensible enthalpy stream term [W]
        # ? use ideal gas enthalpy for liquid enthalpy estimation
        # ! En = sum_i(F_in,i * (h_i^L(Tin) - h_i^L(T)))
        _, En_LIQ_values_out = self.thermo_source.calc_En_LIQ_ref(
            temperature=temperature
        )

        # calculate sensible enthalpy stream term
        # [W] En = sum_i(F_in,i * (h_i^L(Tin) - h_i^L(T))) = sum_i(F_in,i * h_i^L(Tin)) - sum_i(F_in,i * h_i^L(T)))
        En = np.sum(self.F_in * (self.En_LIQ_values_in - En_LIQ_values_out))

        # NOTE: reaction heat term [W]
        # ! Q_rxn = V_R * sum_k((-dH_k) * r_k)
        # ??? ΔH_k
        delta_h = self._calc_dH_rxns(
            temperature=temperature,
            phase=cast(Literal['gas', 'liquid'], 'liquid')
        )

        # ??? Q_rxn
        # [W] Q_rxn = V_R * sum_k((-dH_k) * r_k)
        q_rxn = calc_rxn_heat_generation(
            delta_h=delta_h,
            rates=rates,
            reactor_volume=reactor_volume
        )

        # NOTE: jacket/surrounding heat transfer term [W]
        # ! calculate heat exchange with surroundings: Q_exchange = UA (T_s - T)
        # U [W/m^2.K], A [m^2], T_s [K], temp [K] => Q_exchange [W]
        q_exchange = 0.0

        # >>> check
        if self.heat_exchange:
            q_exchange = calc_heat_exchange(
                temperature=temp,
                jacket_temperature=self.jacket_temperature_value,
                heat_transfer_area=self.heat_transfer_area_value,
                heat_transfer_coefficient=self.heat_transfer_coefficient_value,
                reactor_volume=1.0
            )

        # NOTE: user-defined constant heat term [W]
        q_constant = 0.0
        if self.heat_rate_value:
            q_constant = self.heat_rate_value

        # ! dT/dt [K/s]
        return (En + q_rxn + q_exchange + q_constant) / Cp_LIQ_MIX_TOTAL

    # NOTE: flow closure for outlet molar flow rate
    def _calc_F_out_total(
        self,
        concentration_total: float
    ) -> float:
        """
        Compute total outlet molar flow rate [mol/s] from liquid-flow closure.

        Constant-volume closure:
            Q_out = Q_in (fixed from initialization)
            F_out,total = C_total * Q_out
        """
        if self.operation_mode == "constant_volume":
            return float(concentration_total * self.q_in)

        raise NotImplementedError(
            "Outlet-flow policy for non-constant-volume liquid CSTR is not implemented yet."
        )
