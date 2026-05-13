import numpy as np
from typing import Any, Dict
from pythermodb_settings.models import Temperature, Pressure

# locals
from ..core.gas_br import GasBatchReactor
from ..core.gas_brx import GasBatchReactorX


class GasBatchReactorObservables:
    """
    Post-solution observable evaluator for gas batch reactor trajectories.
    """

    def __init__(self, reactor: GasBatchReactor | GasBatchReactorX):
        self.reactor = reactor

    def evaluate_all(self, t: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate all available observables on the provided trajectory.
        """
        t_arr, y_arr = self._normalize_inputs(t=t, y=y)
        ns = self.reactor.component_num
        n_points = y_arr.shape[1]
        n_rxn = len(self.reactor.reaction_rates)

        total_mole = np.empty(n_points, dtype=float)
        reactor_volume = np.empty(n_points, dtype=float)
        total_pressure = np.empty(n_points, dtype=float)
        concentration = np.empty((ns, n_points), dtype=float)
        concentration_total = np.empty(n_points, dtype=float)
        rates = np.empty((n_rxn, n_points), dtype=float)

        for j in range(n_points):
            n = y_arr[:ns, j]
            n_total = float(max(np.sum(n), 1e-30))
            total_mole[j] = n_total

            if self.reactor.heat_transfer_mode == "isothermal":
                temp = float(self.reactor.T0)
            else:
                temp = float(y_arr[ns, j])

            temperature = Temperature(value=temp, unit="K")
            y_mole = n / n_total

            _, partial_pressures_std, p_total, v_reactor = self.reactor._calc_partial_pressure(
                n_total=n_total,
                y_mole=y_mole,
                T=temp,
            )
            total_pressure[j] = float(p_total)
            reactor_volume[j] = float(v_reactor)

            conc_vec, concentration_std, conc_total = self.reactor._calc_concentration(
                n=n,
                reactor_volume=float(v_reactor),
            )
            concentration[:, j] = np.asarray(conc_vec, dtype=float)
            concentration_total[j] = float(conc_total)

            rates[:, j] = self.reactor._calc_rates(
                partial_pressures=partial_pressures_std,
                concentration=concentration_std,
                temperature=temperature,
                pressure=Pressure(value=float(p_total), unit="Pa"),
            )

        return {
            "time": t_arr,
            "total_mole": total_mole,
            "pressure_total": total_pressure,
            "reactor_volume": reactor_volume,
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
            raise ValueError("State array must be 2D with shape (n_states, n_points).")

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
                "Invalid number of states for gas batch reactor: "
                f"expected {expected_states}, got {y_arr.shape[0]}."
            )

        return t_arr, y_arr
