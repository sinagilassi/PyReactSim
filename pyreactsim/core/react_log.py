# import libs
import logging
import time
from typing import Any, Optional
# locals

# NOTE: logger setup
logger = logging.getLogger(__name__)


class ReactLog:
    """
    ReactLog class for logging and tracking the state of the reactor during simulations. This class provides methods for logging various aspects of the reactor's state, such as component concentrations, temperatures, pressures, and any other relevant information that may be useful for analyzing the simulation results or debugging purposes.
    """

    def __init__(self):
        self.rhs_log_interval: Optional[float] = None
        self.rhs_log_enabled: bool = False
        self.rhs_log_level: int = logging.INFO
        self.rhs_log_timing_enabled: bool = True
        self._rhs_next_log_x: Optional[float] = None
        self._rhs_last_x: Optional[float] = None
        self._rhs_log_active: bool = False
        self._rhs_log_x: Optional[float] = None
        self._rhs_axis_label: str = "t"
        self._rhs_wall_t0: Optional[float] = None
        self._rhs_last_log_wall: Optional[float] = None

    def configure_rhs_logging(
        self,
        interval: Optional[float],
        enabled: bool = True,
        level: int = logging.INFO,
        timing_enabled: bool = True,
        axis_label: str = "t"
    ) -> None:
        if interval is None:
            self.rhs_log_interval = None
            self.rhs_log_enabled = False
        else:
            interval = float(interval)
            if interval < 0.0:
                raise ValueError("interval must be >= 0.")
            self.rhs_log_interval = interval
            self.rhs_log_enabled = bool(enabled)

        self.rhs_log_level = int(level)
        self.rhs_log_timing_enabled = bool(timing_enabled)
        self._rhs_axis_label = axis_label
        self._rhs_next_log_x = None
        self._rhs_last_x = None
        self._rhs_log_active = False
        self._rhs_log_x = None
        self._rhs_wall_t0 = None
        self._rhs_last_log_wall = None

    def _should_log_rhs(self, x: float) -> bool:
        if not self.rhs_log_enabled:
            return False

        interval = self.rhs_log_interval
        if interval is None:
            return False

        x = float(x)
        tol = 1e-12

        # Some solvers can revisit lower values (rejected/restarted step).
        if self._rhs_last_x is not None and x < (self._rhs_last_x - tol):
            self._rhs_next_log_x = None
        self._rhs_last_x = x

        if interval == 0.0:
            return True

        if self._rhs_next_log_x is None:
            self._rhs_next_log_x = x

        if x + tol < self._rhs_next_log_x:
            return False

        while self._rhs_next_log_x is not None and self._rhs_next_log_x <= (x + tol):
            self._rhs_next_log_x += interval

        return True

    def _log_rhs(self, section: str, **fields: Any) -> None:
        if not self._rhs_log_active:
            return

        x_now = 0.0 if self._rhs_log_x is None else self._rhs_log_x
        if self.rhs_log_timing_enabled:
            now = time.perf_counter()
            if self._rhs_wall_t0 is None:
                self._rhs_wall_t0 = now
            if self._rhs_last_log_wall is None:
                self._rhs_last_log_wall = self._rhs_wall_t0

            elapsed_ms = (now - self._rhs_wall_t0) * 1e3
            step_ms = (now - self._rhs_last_log_wall) * 1e3
            self._rhs_last_log_wall = now
            fields = {"step_ms": step_ms, "elapsed_ms": elapsed_ms, **fields}

        if fields:
            details = ", ".join(f"{k}={v}" for k, v in fields.items())
            logger.log(
                self.rhs_log_level,
                "rhs[%s=%.6g] %s | %s",
                self._rhs_axis_label,
                x_now,
                section,
                details,
            )
        else:
            logger.log(
                self.rhs_log_level,
                "rhs[%s=%.6g] %s",
                self._rhs_axis_label,
                x_now,
                section,
            )
