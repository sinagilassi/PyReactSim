import numpy as np
from typing import Optional, Tuple

# locals
from .gas_pbr import GasPBRReactor
from ..utils.tools import smooth_floor


class GasPBRReactorX(GasPBRReactor):
    """
    Scaled gas-phase packed-bed reactor model.

    Physical balances are inherited from GasPBRReactor; only solver-state
    scaling/unscaling is added.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Species scaling uses inlet molar flows with a protective floor so
        # product species with zero inlet flow still get a meaningful scale.
        self.F_scale = np.maximum(self.F_in.astype(float), 1e-8)

        # T_scale_ref is the numerical scaling center, not thermodynamic reference state.
        self.T_scale_ref = float(self._T_in)
        self.T_scale = 100.0  # K

    def build_y0_scaled(self) -> np.ndarray:
        """
        Build scaled inlet state vector for solve_ivp.

        Scaling
        -------
        - Species: f_i = F_i / F_scale_i
        - Temperature: theta = (T - T_scale_ref) / T_scale
        """
        f0_scaled = self.F_in.astype(float) / self.F_scale
        y0_parts = [f0_scaled]

        if self.heat_transfer_mode == "non-isothermal":
            theta0 = (float(self._T_in) - self.T_scale_ref) / self.T_scale
            y0_parts.append(np.array([theta0], dtype=float))

        if len(y0_parts) == 1:
            return y0_parts[0]
        return np.concatenate(y0_parts)

    def _unscale_state(self, y_scaled: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Convert a scaled state vector to physical units.
        """
        ns = self.component_num

        F = np.asarray(
            smooth_floor(y_scaled[:ns], xmin=0.0, s=1e-9),
            dtype=float
        ) * self.F_scale

        if self.pbr_reactor_core.is_non_isothermal:
            theta = float(y_scaled[ns])
            temp = self.T_scale_ref + self.T_scale * theta
            temp = float(smooth_floor(temp, xmin=1.0, s=1e-3))
        else:
            temp = float(self._T_in)

        return F, temp

    def _scale_rhs(
        self,
        dF_dV: np.ndarray,
        dT_dV: Optional[float] = None,
    ) -> np.ndarray:
        """
        Convert physical derivatives to scaled derivatives.
        """
        dF_scaled_dV = dF_dV / self.F_scale

        if dT_dV is None:
            return dF_scaled_dV

        dtheta_dV = dT_dV / self.T_scale
        return np.concatenate([dF_scaled_dV, np.array([dtheta_dV], dtype=float)])

    def rhs_physical(
            self,
            V: float,
            y: np.ndarray
    ) -> np.ndarray:
        """
        Physical RHS wrapper.
        """
        return super().rhs(V, y)

    def rhs_scaled(
            self,
            V: float,
            y_scaled: np.ndarray
    ) -> np.ndarray:
        """
        Scaled right-hand side for solve_ivp.

        Notes
        -----
        The reactor model itself is still evaluated in physical units.
        Only the state vector seen by the ODE solver is scaled.
        """
        ns = self.component_num

        F, temp = self._unscale_state(y_scaled)

        if self.pbr_reactor_core.is_non_isothermal:
            y_physical = np.concatenate([F, np.array([temp], dtype=float)])
        else:
            y_physical = F

        dy_physical_dV = self.rhs_physical(V, y_physical)

        dF_dV = dy_physical_dV[:ns]

        if self.heat_transfer_mode == "isothermal":
            return self._scale_rhs(dF_dV=dF_dV)

        dT_dV = float(dy_physical_dV[ns])
        return self._scale_rhs(dF_dV=dF_dV, dT_dV=dT_dV)
