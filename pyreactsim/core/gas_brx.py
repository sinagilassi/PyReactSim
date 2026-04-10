import numpy as np
from typing import Tuple, Optional

# locals
from .gas_br import GasBatchReactor
from ..utils.tools import smooth_floor


class GasBatchReactorX(GasBatchReactor):
    """
    Scaled gas-phase batch reactor model.

    The physical model equations are unchanged; only the ODE solver state is
    scaled to improve numerical conditioning.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Species scaling uses initial moles with a small floor so species with
        # zero initial amount still receive a finite solver scale.
        self.N_scale = np.maximum(self.N0.astype(float), 1e-8)

        # Temperature scaling as deviation from initial temperature.
        self.T_ref = float(self._T0)
        self.T_scale = 100.0  # K

    def build_y0_scaled(self) -> np.ndarray:
        """
        Build scaled inlet state vector for solve_ivp.

        Scaling
        -------
        - Species: n_i_scaled = n_i / N_scale_i
        - Temperature: theta = (T - T_ref) / T_scale
        """
        n0_scaled = self.N0.astype(float) / self.N_scale
        y0_parts = [n0_scaled]

        if self.heat_transfer_mode == "non-isothermal":
            theta0 = (float(self._T0) - self.T_ref) / self.T_scale
            y0_parts.append(np.array([theta0], dtype=float))

        if len(y0_parts) == 1:
            return y0_parts[0]
        return np.concatenate(y0_parts)

    def _unscale_state(self, y_scaled: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Convert scaled state vector to physical units.
        """
        ns = self.component_num

        n = np.asarray(
            smooth_floor(y_scaled[:ns], xmin=0.0, s=1e-9),
            dtype=float
        ) * self.N_scale

        if self.heat_transfer_mode == "non-isothermal":
            theta = float(y_scaled[ns])
            temp = self.T_ref + self.T_scale * theta
            temp = float(smooth_floor(temp, xmin=1.0, s=1e-3))
        else:
            temp = float(self._T0)

        return n, temp

    def _scale_rhs(
        self,
        dn_dt: np.ndarray,
        dT_dt: Optional[float] = None
    ) -> np.ndarray:
        """
        Convert physical derivatives to scaled derivatives.
        """
        dn_scaled_dt = dn_dt / self.N_scale

        if dT_dt is None:
            return dn_scaled_dt

        dtheta_dt = dT_dt / self.T_scale
        return np.concatenate([dn_scaled_dt, np.array([dtheta_dt], dtype=float)])

    def rhs_physical(self, t: float, y: np.ndarray) -> np.ndarray:
        """
        Physical RHS wrapper.
        """
        return super().rhs(t, y)

    def rhs_scaled(self, t: float, y_scaled: np.ndarray) -> np.ndarray:
        """
        Scaled RHS for solve_ivp.
        """
        ns = self.component_num

        n, temp = self._unscale_state(y_scaled)

        if self.heat_transfer_mode == "non-isothermal":
            y_physical = np.concatenate([n, np.array([temp], dtype=float)])
        else:
            y_physical = n

        dy_physical_dt = self.rhs_physical(t, y_physical)
        dn_dt = dy_physical_dt[:ns]

        if self.heat_transfer_mode == "isothermal":
            return self._scale_rhs(dn_dt=dn_dt)

        dT_dt = float(dy_physical_dt[ns])
        return self._scale_rhs(dn_dt=dn_dt, dT_dt=dT_dt)
