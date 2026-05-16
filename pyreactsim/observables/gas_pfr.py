import numpy as np
from typing import Any, Dict, cast
from pythermodb_settings.models import Temperature, Pressure, CustomProperty

# locals
from ..core.gas_pfr import GasPFRReactor
from ..core.gas_pfrx import GasPFRReactorX
from ..models.ref import GasModel
from ..utils.thermo_tools import calc_pressure_using_PFT


class GasPFRReactorObservables:
    """
    Post-solution observable evaluator for gas PFR trajectories.
    """

    def __init__(self, reactor: GasPFRReactor | GasPFRReactorX):
        self.reactor = reactor

    def evaluate_all(self, t: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate all available observables on the provided trajectory.
        """
        v_arr, y_arr = self._normalize_inputs(t=t, y=y)
        ns = self.reactor.component_num
        n_points = y_arr.shape[1]
        n_rxn = len(self.reactor.reaction_rates)

        total_mole_flow = np.empty(n_points, dtype=float)
        total_pressure = np.empty(n_points, dtype=float)
        volumetric_flow = np.empty(n_points, dtype=float)
        concentration = np.empty((ns, n_points), dtype=float)
        concentration_total = np.empty(n_points, dtype=float)
        rates = np.empty((n_rxn, n_points), dtype=float)
        temperature_profile = np.empty(n_points, dtype=float)

        for j in range(n_points):
            f = np.clip(y_arr[:ns, j], 0.0, None)
            f_total = float(max(np.sum(f), 1e-30))
            total_mole_flow[j] = f_total

            idx = ns
            if self.reactor.heat_transfer_mode == "non-isothermal":
                temp = float(y_arr[idx, j])
                idx += 1
            else:
                temp = float(self.reactor._T_in)
            temperature_profile[j] = temp

            pressure_mode = self.reactor.pressure_mode
            if pressure_mode in ("state_variable", "calculated"):
                p_total = float(max(y_arr[idx, j], 1e-9))
            elif pressure_mode == "shortcut":
                p_total = calc_pressure_using_PFT(
                    P_in=self.reactor._P0,
                    F_in_total=self.reactor._F_in_total,
                    T_in=self.reactor._T_in,
                    F_out_total=f_total,
                    T_out=temp,
                    heat_transfer_mode=cast(
                        Any, self.reactor.heat_transfer_mode
                    ),
                )
            elif pressure_mode == "constant":
                p_total = float(self.reactor._P0)
            else:
                raise ValueError(
                    f"Invalid pressure_mode '{pressure_mode}' for gas PFR observables."
                )
            total_pressure[j] = p_total

            q_vol = self.reactor.thermo_source.calc_gas_volumetric_flow_rate(
                molar_flow_rate=f_total,
                temperature=temp,
                pressure=p_total,
                R=self.reactor.R,
                gas_model=cast(GasModel, self.reactor.gas_model),
            )
            q_vol = float(max(q_vol, 1e-30))
            volumetric_flow[j] = q_vol

            conc_vec = f / q_vol
            concentration[:, j] = conc_vec
            concentration_total[j] = float(np.sum(conc_vec))

            y_mole = f / f_total
            partial_pressures_std = {
                sp: CustomProperty(
                    value=float(y_mole[i] * p_total), unit="Pa", symbol="P"
                )
                for i, sp in enumerate(self.reactor.component_formula_state)
            }
            concentration_std = {
                sp: CustomProperty(
                    value=float(conc_vec[i]), unit="mol/m3", symbol="C"
                )
                for i, sp in enumerate(self.reactor.component_formula_state)
            }

            rates[:, j] = self.reactor._calc_rates(
                partial_pressures=partial_pressures_std,
                concentration=concentration_std,
                temperature=Temperature(value=temp, unit="K"),
                pressure=Pressure(value=p_total, unit="Pa"),
            )

        return {
            "volume": v_arr,
            "total_mole_flow": total_mole_flow,
            "temperature": temperature_profile,
            "pressure_total": total_pressure,
            "volumetric_flow": volumetric_flow,
            "concentration": concentration,
            "concentration_total": concentration_total,
            "rate": rates,
        }

    def _normalize_inputs(self, t: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Normalize and validate trajectory arrays.
        """
        t_arr = np.asarray(t, dtype=float).reshape(-1)
        y_arr = np.asarray(y, dtype=float)

        if y_arr.ndim != 2:
            raise ValueError(
                "State array must be 2D with shape (n_states, n_points).")

        if y_arr.shape[1] != t_arr.size and y_arr.shape[0] == t_arr.size:
            y_arr = y_arr.T

        if y_arr.shape[1] != t_arr.size:
            raise ValueError(
                "Incompatible trajectory shape: expected y.shape[1] == len(t) "
                f"but got y.shape={y_arr.shape}, len(t)={t_arr.size}."
            )

        expected_states = self.reactor.component_num
        if self.reactor.heat_transfer_mode == "non-isothermal":
            expected_states += 1
        if self.reactor.pressure_mode in ("state_variable", "calculated"):
            expected_states += 1

        if y_arr.shape[0] != expected_states:
            raise ValueError(
                "Invalid number of states for gas PFR reactor: "
                f"expected {expected_states}, got {y_arr.shape[0]}."
            )

        return t_arr, y_arr
