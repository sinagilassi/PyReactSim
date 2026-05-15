import numpy as np
from typing import Any, Dict
from pythermodb_settings.models import Temperature, CustomProperty

# locals
from ..core.liquid_br import LiquidBatchReactor
from ..core.liquid_brx import LiquidBatchReactorX


class LiquidBatchReactorObservables:
    """
    Post-solution observable evaluator for liquid batch reactor trajectories.

    This class evaluates trajectory-aligned variables from solved time/state
    arrays rather than collecting values inside RHS calls.
    """

    def __init__(self, reactor: LiquidBatchReactor | LiquidBatchReactorX):
        self.reactor = reactor

    def evaluate_all(self, t: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate all available observables on the provided trajectory.
        """
        t_arr, y_arr = self._normalize_inputs(t=t, y=y)
        reactor_volume = self.evaluate_reactor_volume(t=t_arr, y=y_arr)
        total_mole = self.evaluate_total_mole(t=t_arr, y=y_arr)
        rho_liq = self.evaluate_liquid_density(t=t_arr, y=y_arr)
        concentration, concentration_total = self.evaluate_concentration(
            t=t_arr,
            y=y_arr,
            reactor_volume=reactor_volume,
        )
        rates = self.evaluate_rates(t=t_arr, y=y_arr, concentration=concentration)

        return {
            "time": t_arr,
            "total_mole": total_mole,
            "rho_LIQ": rho_liq,
            "reactor_volume": reactor_volume,
            "concentration": concentration,
            "concentration_total": concentration_total,
            "rate": rates,
        }

    def evaluate_reactor_volume(self, t: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Evaluate reactor volume at each trajectory point.
        """
        _ = t  # reserved for future time-dependent observables
        _, y_arr = self._normalize_inputs(t=t, y=y)
        ns = self.reactor.component_num
        n_points = y_arr.shape[1]
        volumes = np.empty(n_points, dtype=float)

        for j in range(n_points):
            n = y_arr[:ns, j]

            if self.reactor.heat_transfer_mode == "isothermal":
                temp = float(self.reactor.T0)
            else:
                temp = float(y_arr[ns, j])

            rho_liq = self.reactor.thermo_source.calc_rho_LIQ(
                temperature=Temperature(value=temp, unit="K"),
                operation_mode=self.reactor.operation_mode,
            )
            volumes[j] = float(
                self.reactor._calc_system_volume(
                    n=n,
                    rho_LIQ=rho_liq,
                    temperature=temp,
                )
            )

        return volumes

    def evaluate_total_mole(self, t: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Evaluate total mole amount at each trajectory point.
        """
        _ = t
        _, y_arr = self._normalize_inputs(t=t, y=y)
        ns = self.reactor.component_num
        return np.sum(y_arr[:ns, :], axis=0)

    def evaluate_liquid_density(self, t: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Evaluate mixture liquid density at each trajectory point.
        """
        _ = t
        _, y_arr = self._normalize_inputs(t=t, y=y)
        ns = self.reactor.component_num
        n_points = y_arr.shape[1]
        rho = np.empty(n_points, dtype=float)

        for j in range(n_points):
            if self.reactor.heat_transfer_mode == "isothermal":
                temp = float(self.reactor.T0)
            else:
                temp = float(y_arr[ns, j])

            rho_raw = self.reactor.thermo_source.calc_rho_LIQ(
                temperature=Temperature(value=temp, unit="K"),
                operation_mode=self.reactor.operation_mode,
            )
            if np.asarray(rho_raw).size == 0:
                rho[j] = np.nan
                continue
            rho[j] = self._to_mixture_density(
                n=y_arr[:ns, j],
                rho_raw=rho_raw,
            )

        return rho

    def evaluate_concentration(
        self,
        t: np.ndarray,
        y: np.ndarray,
        reactor_volume: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Evaluate component and total concentrations over the trajectory.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            component concentration with shape (n_components, n_points),
            and total concentration with shape (n_points,).
        """
        _ = t
        _, y_arr = self._normalize_inputs(t=t, y=y)
        ns = self.reactor.component_num
        n = y_arr[:ns, :]

        if reactor_volume is None:
            reactor_volume = self.evaluate_reactor_volume(t=t, y=y_arr)

        volume = np.asarray(reactor_volume, dtype=float).reshape(-1)
        if volume.size != y_arr.shape[1]:
            raise ValueError(
                "reactor_volume length must match number of time points."
            )

        concentration = n / volume[np.newaxis, :]
        concentration_total = np.sum(n, axis=0) / volume
        return concentration, concentration_total

    def evaluate_rates(
        self,
        t: np.ndarray,
        y: np.ndarray,
        concentration: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Evaluate reaction rates at each trajectory point.

        Returns
        -------
        np.ndarray
            Rate matrix with shape (n_reactions, n_points).
        """
        _ = t
        _, y_arr = self._normalize_inputs(t=t, y=y)
        ns = self.reactor.component_num
        n_points = y_arr.shape[1]
        n_rxn = len(self.reactor.reaction_rates)
        rates = np.empty((n_rxn, n_points), dtype=float)

        if concentration is None:
            concentration, _ = self.evaluate_concentration(t=t, y=y_arr)
        conc_arr = np.asarray(concentration, dtype=float)
        if conc_arr.shape != (ns, n_points):
            raise ValueError(
                "concentration must have shape (n_components, n_points)."
            )

        conc_ids = list(self.reactor.component_formula_state)

        for j in range(n_points):
            if self.reactor.heat_transfer_mode == "isothermal":
                temp = float(self.reactor.T0)
            else:
                temp = float(y_arr[ns, j])

            temperature = Temperature(value=temp, unit="K")
            concentration_std: Dict[str, CustomProperty] = {
                sp: CustomProperty(
                    value=float(conc_arr[i, j]),
                    unit="mol/m3",
                    symbol="C",
                )
                for i, sp in enumerate(conc_ids)
            }
            rates[:, j] = self.reactor._calc_rates(
                concentration=concentration_std,
                temperature=temperature,
            )

        return rates

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
                "Invalid number of states for liquid batch reactor: "
                f"expected {expected_states}, got {y_arr.shape[0]}."
            )

        return t_arr, y_arr

    def _to_mixture_density(self, n: np.ndarray, rho_raw: Any) -> float:
        """
        Convert raw liquid-density output to a scalar mixture density.
        """
        rho_arr = np.asarray(rho_raw, dtype=float)
        if rho_arr.ndim == 0:
            return float(rho_arr)

        rho_arr = rho_arr.reshape(-1)
        if rho_arr.size != self.reactor.component_num:
            raise ValueError(
                "rho_LIQ size must match number of components when vector-valued."
            )

        n_arr = np.asarray(n, dtype=float).reshape(-1)
        mw = np.asarray(self.reactor.thermo_source.MW, dtype=float).reshape(-1)
        if mw.size != self.reactor.component_num:
            raise ValueError("Molecular-weight array size does not match components.")

        mass = n_arr * mw
        mass_total = float(np.sum(mass))
        if mass_total <= 0.0:
            return float(np.mean(rho_arr))

        volume_total = float(np.sum(mass / np.maximum(rho_arr, 1e-30)))
        return mass_total / max(volume_total, 1e-30)
