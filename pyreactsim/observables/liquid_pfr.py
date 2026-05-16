import numpy as np
from typing import Any, Dict
from pythermodb_settings.models import Temperature, CustomProperty

# locals
from ..core.liquid_pfr import LiquidPFRReactor
from ..core.liquid_pfrx import LiquidPFRReactorX


class LiquidPFRReactorObservables:
    """
    Post-solution observable evaluator for liquid PFR trajectories.
    """

    def __init__(self, reactor: LiquidPFRReactor | LiquidPFRReactorX):
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
        volumetric_flow = np.empty(n_points, dtype=float)
        concentration = np.empty((ns, n_points), dtype=float)
        concentration_total = np.empty(n_points, dtype=float)
        rates = np.empty((n_rxn, n_points), dtype=float)
        rho_liq = np.empty(n_points, dtype=float)
        temperature_profile = np.empty(n_points, dtype=float)

        for j in range(n_points):
            f = np.clip(y_arr[:ns, j], 0.0, None)
            f_total = float(max(np.sum(f), 1e-30))
            total_mole_flow[j] = f_total

            if self.reactor.heat_transfer_mode == "non-isothermal":
                temp = float(y_arr[ns, j])
            else:
                temp = float(self.reactor._T_in)
            temperature_profile[j] = temp

            temperature = Temperature(value=temp, unit="K")
            rho_raw = self.reactor.thermo_source.calc_rho_LIQ(
                temperature=temperature
            )
            rho_liq[j] = self._to_mixture_density(f=f, rho_raw=rho_raw)

            q_vol = self.reactor._calc_q_vol(F=f, rho_LIQ=rho_raw)
            q_vol = float(max(q_vol, 1e-30))
            volumetric_flow[j] = q_vol

            conc_vec = f / q_vol
            concentration[:, j] = conc_vec
            concentration_total[j] = float(np.sum(conc_vec))

            concentration_std = {
                sp: CustomProperty(
                    value=float(conc_vec[i]), unit="mol/m3", symbol="C"
                )
                for i, sp in enumerate(self.reactor.component_formula_state)
            }
            rates[:, j] = self.reactor._calc_rates_concentration_basis(
                concentration=concentration_std,
                temperature=temperature,
            )

        return {
            "volume": v_arr,
            "total_mole_flow": total_mole_flow,
            "temperature": temperature_profile,
            "rho_LIQ": rho_liq,
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

        expected_states = self.reactor.component_num + (
            0 if self.reactor.heat_transfer_mode == "isothermal" else 1
        )
        if y_arr.shape[0] != expected_states:
            raise ValueError(
                "Invalid number of states for liquid PFR reactor: "
                f"expected {expected_states}, got {y_arr.shape[0]}."
            )

        return t_arr, y_arr

    def _to_mixture_density(self, f: np.ndarray, rho_raw: Any) -> float:
        """
        Convert raw liquid-density output to scalar mixture density.
        """
        rho_arr = np.asarray(rho_raw, dtype=float)
        if rho_arr.ndim == 0:
            return float(rho_arr)

        rho_arr = rho_arr.reshape(-1)
        if rho_arr.size != self.reactor.component_num:
            raise ValueError(
                "rho_LIQ size must match number of components when vector-valued."
            )

        f_arr = np.asarray(f, dtype=float).reshape(-1)
        mw = np.asarray(self.reactor.thermo_source.MW, dtype=float).reshape(-1)
        if mw.size != self.reactor.component_num:
            raise ValueError(
                "Molecular-weight array size does not match components.")

        mass = f_arr * mw
        mass_total = float(np.sum(mass))
        if mass_total <= 0.0:
            return float(np.mean(rho_arr))

        volume_total = float(np.sum(mass / np.maximum(rho_arr, 1e-30)))
        return mass_total / max(volume_total, 1e-30)
