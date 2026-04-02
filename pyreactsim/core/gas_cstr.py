import logging
import numpy as np
from typing import Dict, List, Tuple, cast
from pythermodb_settings.models import Component, ComponentKey, CustomProperty, Pressure, Temperature, Volume
# locals
from .cstrc import CSTRReactorCore
from ..models.br import GasModel
from ..models.rate_exp import ReactionRateExpression
from ..sources.thermo_source import ThermoSource
from ..utils.opt_tools import calc_heat_exchange
from ..utils.reaction_tools import stoichiometry_mat, stoichiometry_mat_key
from ..utils.thermo_tools import calc_rxn_heat_generation, calc_total_heat_capacity

# NOTE: logger setup
logger = logging.getLogger(__name__)


class GasCSTRReactor:
    """
    Gas-phase CSTR dynamic reactor model.

    State vector:
    - isothermal cases: [n1, n2, ..., nNc]
    - non-isothermal cases: [n1, n2, ..., nNc, T]
    """

    R = 8.314

    _N0: np.ndarray = np.array([])
    _F_in: np.ndarray = np.array([])
    _F_out_total: float = 0.0
    _T0: float = 0.0
    _T_in: float = 0.0
    _P0: float = 0.0
    _V0: float = 0.0

    def __init__(
        self,
        components: List[Component],
        reaction_rates: List[ReactionRateExpression],
        thermo_source: ThermoSource,
        cstr_reactor_core: CSTRReactorCore,
        component_key: ComponentKey,
        **kwargs
    ):
        self.components = components
        self.component_key = component_key
        self.thermo_source = thermo_source
        self.cstr_reactor_core = cstr_reactor_core

        self.cstr_reactor_options = cstr_reactor_core.cstr_reactor_options
        self.case = self.cstr_reactor_options.case
        self.gas_model = cstr_reactor_core.gas_model
        self.heat_transfer_mode = cstr_reactor_core.heat_transfer_mode

        self.heat_exchange = cstr_reactor_core.heat_exchange
        self.heat_transfer_coefficient_value = cstr_reactor_core.heat_transfer_coefficient_value
        self.heat_transfer_area_value = cstr_reactor_core.heat_transfer_area_value
        self.jacket_temperature_value = cstr_reactor_core.jacket_temperature_value
        self.heat_rate = cstr_reactor_core.heat_rate
        self.heat_rate_value = cstr_reactor_core.heat_rate_value

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

        self.component_num = self.thermo_source.component_refs["component_num"]
        self.component_ids = self.thermo_source.component_refs["component_ids"]
        self.component_formula_state = self.thermo_source.component_refs["component_formula_state"]
        self.component_mapper = self.thermo_source.component_refs["component_mapper"]
        self.component_id_to_index = self.thermo_source.component_refs["component_id_to_index"]

        _, self._N0 = self.cstr_reactor_core.config_initial_mole()
        _, self._F_in = self.cstr_reactor_core.config_feed_mole_flow()
        self._F_out_total = self.cstr_reactor_core.config_outlet_mole_flow_total()
        self._T0 = self.cstr_reactor_core.initial_temperature.value
        self._T_in = self.cstr_reactor_core.feed_temperature.value

        self._configure_pressure_volume_initial()

    def _configure_pressure_volume_initial(self):
        if self.cstr_reactor_options.is_constant_pressure:
            pressure = self.cstr_reactor_core.config_pressure()
            self._P0 = pressure.value

        if self.cstr_reactor_options.is_constant_volume:
            volume = self.cstr_reactor_core.config_reactor_volume()
            self._V0 = volume.value

        # fixed-volume in constant-pressure cases (1,3,5)
        if (
            self.cstr_reactor_options.is_constant_pressure
            and not self.cstr_reactor_options.is_variable_volume
            and not self.cstr_reactor_options.is_constant_volume
        ):
            volume = self.cstr_reactor_core.config_reactor_volume()
            self._V0 = volume.value

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

    def build_y0(self) -> np.ndarray:
        if self.cstr_reactor_options.is_isothermal:
            return self.N0
        return np.concatenate([self.N0, np.array([self._T0], dtype=float)])

    def rhs(self, t: float, y: np.ndarray) -> np.ndarray:
        ns = self.component_num

        if self.cstr_reactor_options.is_isothermal:
            n = y[:ns]
            temp = float(self._T0)
        else:
            n = y[:ns]
            temp = float(y[ns])

        n = np.clip(n, 0.0, None)
        n_total = float(np.sum(n))
        n_total = max(n_total, 1e-30)
        y_mole = n / n_total

        p_total, reactor_volume = self._calc_pressure_volume(
            n_total=n_total,
            temperature=temp
        )

        (
            _,
            partial_pressures_std,
            _
        ) = self._calc_partial_pressure(
            y_mole=y_mole,
            p_total=p_total
        )

        (
            _,
            concentration_std,
            _
        ) = self._calc_concentration(
            n=n,
            reactor_volume=reactor_volume
        )

        rates = self._calc_rates(
            partial_pressures=partial_pressures_std,
            concentration=concentration_std,
            temperature=Temperature(value=temp, unit="K"),
            pressure=Pressure(value=p_total, unit="Pa")
        )

        dn_dt = self._build_dn_dt(
            ns=ns,
            rates=rates,
            reactor_volume=reactor_volume,
            y_mole=y_mole
        )

        if self.cstr_reactor_options.is_isothermal:
            return dn_dt

        dT_dt = self._build_dT_dt(
            n=n,
            rates=rates,
            temp=temp,
            reactor_volume=reactor_volume
        )

        return np.concatenate([dn_dt, np.array([dT_dt], dtype=float)])

    def _calc_pressure_volume(self, n_total: float, temperature: float) -> Tuple[float, float]:
        if self.cstr_reactor_options.is_constant_pressure:
            p_total = self._P0

            if self.cstr_reactor_options.is_variable_volume:
                reactor_volume = self.thermo_source.calc_gas_volume(
                    n_total=n_total,
                    temperature=temperature,
                    pressure=p_total,
                    R=self.R,
                    gas_model=cast(GasModel, self.gas_model)
                )
            else:
                reactor_volume = self._V0
        else:
            reactor_volume = self._V0
            p_total = self.thermo_source.calc_tot_pressure(
                n_total=n_total,
                temperature=temperature,
                reactor_volume_value=reactor_volume,
                R=self.R,
                gas_model=cast(GasModel, self.gas_model)
            )

        if reactor_volume <= 1e-30:
            raise ValueError(
                "Calculated reactor volume is too small or non-positive. "
                "Check case closure, model inputs, and gas_model support."
            )

        if p_total <= 0.0:
            raise ValueError(
                "Calculated total pressure is non-positive. "
                "Check case closure, model inputs, and gas_model support."
            )

        return p_total, reactor_volume

    def _calc_partial_pressure(
        self,
        y_mole: np.ndarray,
        p_total: float
    ) -> Tuple[Dict[str, float], Dict[str, CustomProperty], float]:
        partial_pressures = {
            sp: y_mole[i] * p_total for i, sp in enumerate(self.component_formula_state)
        }
        partial_pressures_std = {
            k: CustomProperty(value=v, unit="Pa", symbol="P")
            for k, v in partial_pressures.items()
        }
        return partial_pressures, partial_pressures_std, p_total

    def _calc_concentration(
        self,
        n: np.ndarray,
        reactor_volume: float
    ) -> Tuple[np.ndarray, Dict[str, CustomProperty], float]:
        concentration = n / reactor_volume
        concentration_total = float(np.sum(n)) / reactor_volume
        concentration_std = {
            sp: CustomProperty(value=conc, unit="mol/m3", symbol="C")
            for sp, conc in zip(self.component_formula_state, concentration)
        }
        return concentration, concentration_std, concentration_total

    def _calc_rates(
        self,
        partial_pressures: Dict[str, CustomProperty],
        concentration: Dict[str, CustomProperty],
        temperature: Temperature,
        pressure: Pressure
    ) -> np.ndarray:
        rates = []
        for rate_exp in self.reaction_rates:
            basis = rate_exp.basis
            if basis == "pressure":
                r_k = rate_exp.calc(
                    xi=partial_pressures,
                    temperature=temperature,
                    pressure=pressure
                )
            elif basis == "concentration":
                r_k = rate_exp.calc(
                    xi=concentration,
                    temperature=temperature,
                    pressure=pressure
                )
            else:
                raise ValueError(
                    f"Invalid basis '{basis}' for reaction rate expression '{rate_exp.name}'."
                )
            rates.append(float(r_k.value))
        return np.array(rates, dtype=float)

    def _build_dn_dt(
        self,
        ns: int,
        rates: np.ndarray,
        reactor_volume: float,
        y_mole: np.ndarray
    ) -> np.ndarray:
        # CSTR mole balance:
        # dn_i/dt = F_in,i - F_out,i + V * sum_j(nu_i,j * r_j)
        dn_dt = np.zeros(ns, dtype=float)
        name_to_idx = self.component_id_to_index

        f_out_i = y_mole * self._F_out_total
        dn_dt += self.F_in - f_out_i

        for k, _rxn in enumerate(self.reactions):
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
        temperature = Temperature(value=temp, unit="K")
        cp_values = self.thermo_source.calc_Cp_IG(temperature=temperature)
        cp_total = calc_total_heat_capacity(n, cp_values)
        if cp_total <= 1e-16:
            raise ValueError("Total gas heat capacity is too small or zero.")

        delta_h = self.thermo_source.calc_dH_rxns(temperature=temperature)
        q_rxn = calc_rxn_heat_generation(
            delta_h=delta_h,
            rates=rates,
            reactor_volume=reactor_volume
        )

        q_sensible_in = float(np.sum(self.F_in * cp_values * (self._T_in - temp)))

        q_exchange = 0.0
        if self.heat_exchange:
            # use reactor_volume=1 to obtain total UA(Tj-T) [W]
            q_exchange = calc_heat_exchange(
                temperature=temp,
                jacket_temperature=self.jacket_temperature_value,
                heat_transfer_area=self.heat_transfer_area_value,
                heat_transfer_coefficient=self.heat_transfer_coefficient_value,
                reactor_volume=1.0
            )

        q_constant = 0.0
        if self.heat_rate_value:
            q_constant = self.heat_rate_value

        return (q_sensible_in + q_rxn + q_exchange + q_constant) / cp_total
