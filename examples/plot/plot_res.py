from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from pyreactsim.models.br import BatchReactorResult
from pyreactsim.models.cstr import CSTRReactorResult
from pyreactsim.models.pfr import PFRReactorResult
from pyreactsim.models.pbr import PBRReactorResult


def _component_label(component: Any, fallback_index: int) -> str:
    formula = getattr(component, "formula", None)
    state = getattr(component, "state", None)
    name = getattr(component, "name", None)

    if formula and state:
        return f"{formula}({state})"
    if formula:
        return str(formula)
    if name:
        return str(name)
    return f"species_{fallback_index + 1}"


def _build_component_labels(num_mole_states: int, components: Iterable[Any] | None) -> list[str]:
    component_list = list(components) if components is not None else []
    component_labels = [
        _component_label(c, idx) for idx, c in enumerate(component_list)
    ]
    if component_labels and len(component_labels) == num_mole_states:
        return component_labels
    return [f"species_{idx + 1}" for idx in range(num_mole_states)]


def _detect_temperature_state(num_states: int, components: Iterable[Any] | None) -> bool:
    """Return True when the last state row is temperature (non-isothermal case)."""
    component_list = list(components) if components is not None else []
    if component_list:
        return num_states == len(component_list) + 1
    return False


def _plot_reactor_result(
    result: Any,
    components: Iterable[Any] | None = None,
    save_path: Path | None = None,
    show: bool = True,
    title_prefix: str = "Reactor",
    x_values: Iterable[float] | None = None,
    x_label: str | None = None,
) -> None:
    if not result:
        raise ValueError("simulation_result is empty.")

    if not result.success:
        raise RuntimeError(
            f"Simulation failed: {result.message or 'unknown error'}")

    # NOTE: support explicit x-axis override, then fallback to result coordinates.
    if x_values is not None:
        x = np.asarray(list(x_values), dtype=float)
        x_axis_label = x_label or "Coordinate"
        species_y_label = "Molar flow rate (mol/s)"
    elif hasattr(result, "time"):
        x = np.asarray(result.time, dtype=float)
        x_axis_label = "Time (s)"
        species_y_label = "Moles (mol)"
    elif hasattr(result, "volume"):
        x = np.asarray(result.volume, dtype=float)
        x_axis_label = "Reactor volume coordinate (m3)"
        species_y_label = "Molar flow rate (mol/s)"
    else:
        raise ValueError(
            "Unsupported result coordinate. Expected 'time' or 'volume', "
            "or provide x_values explicitly.")

    state = np.asarray(result.state, dtype=float)

    if state.ndim != 2:
        raise ValueError(
            "Expected result.state to be a 2D array with shape (n_states, n_points).")
    if x.shape[0] != state.shape[1]:
        raise ValueError(
            f"Coordinate/state length mismatch: len(x)={x.shape[0]} "
            f"but state has {state.shape[1]} points."
        )

    has_temperature = _detect_temperature_state(state.shape[0], components)

    if has_temperature:
        mole_state = state[:-1]
        temp_state = state[-1]
    else:
        mole_state = state
        temp_state = None

    mole_labels = _build_component_labels(mole_state.shape[0], components)

    if has_temperature:
        fig, axes = plt.subplots(
            2, 1, figsize=(10, 9), sharex=True,
            constrained_layout=True
        )
        ax_mole, ax_temp = axes[0], axes[1]
    else:
        fig, ax_mole = plt.subplots(figsize=(10, 6), constrained_layout=True)
        ax_temp = None

    # species subplot
    for idx, label in enumerate(mole_labels):
        ax_mole.plot(x, mole_state[idx], linewidth=2, label=label)
    ax_mole.set_ylabel(species_y_label)
    ax_mole.set_title(f"{title_prefix} - Species")
    ax_mole.grid(True, alpha=0.3)
    ax_mole.legend(loc="best")
    ax_mole.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.2f}'))

    if ax_temp is not None and temp_state is not None:
        # temperature subplot
        ax_temp.plot(x, temp_state, linewidth=2,
                     color="tab:red", label="Temperature")
        ax_temp.set_xlabel(x_axis_label)
        ax_temp.set_ylabel("Temperature (K)")
        ax_temp.set_title(f"{title_prefix} - Temperature")
        ax_temp.grid(True, alpha=0.3)
        ax_temp.legend(loc="best")
        ax_temp.yaxis.set_major_formatter(
            FuncFormatter(lambda y, _: f'{y:.2f}'))
    else:
        ax_mole.set_xlabel(x_axis_label)

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200)
        print(f"Saved plot to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_batch_reactor_result(
    result: BatchReactorResult,
    components: Iterable[Any] | None = None,
    save_path: Path | None = None,
    show: bool = True,
) -> None:
    _plot_reactor_result(
        result=result,
        components=components,
        save_path=save_path,
        show=show,
        title_prefix="Batch Reactor",
    )


def plot_cstr_reactor_result(
    result: CSTRReactorResult,
    components: Iterable[Any] | None = None,
    save_path: Path | None = None,
    show: bool = True,
    reactor: Any | None = None,
) -> None:
    plot_result: Any = result

    if reactor is not None and getattr(reactor, "modeling_type", None) == "scale":
        reactor_model = getattr(reactor, "reactor", None)
        state = np.asarray(result.state, dtype=float)
        if reactor_model is not None and state.ndim == 2 and hasattr(reactor_model, "_unscale_state"):
            # Detect if state looks scaled by comparing the initial point
            # against scaled vs physical initial vectors.
            should_unscale = False
            if hasattr(reactor_model, "build_y0_scaled") and hasattr(reactor_model, "build_y0"):
                y0_col = state[:, 0]
                y0_scaled = np.asarray(reactor_model.build_y0_scaled(), dtype=float).ravel()
                y0_physical = np.asarray(reactor_model.build_y0(), dtype=float).ravel()
                if y0_col.shape == y0_scaled.shape and y0_col.shape == y0_physical.shape:
                    err_scaled = np.linalg.norm(y0_col - y0_scaled)
                    err_physical = np.linalg.norm(y0_col - y0_physical)
                    should_unscale = err_scaled <= err_physical
            if should_unscale:
                n_points = state.shape[1]
                physical_cols = []
                for j in range(n_points):
                    y_scaled = state[:, j]
                    n, temp = reactor_model._unscale_state(y_scaled)
                    if getattr(reactor_model, "heat_transfer_mode", "isothermal") == "non-isothermal":
                        y_physical = np.concatenate([n, np.array([temp], dtype=float)])
                    else:
                        y_physical = n
                    physical_cols.append(y_physical)
                plot_result = SimpleNamespace(
                    time=result.time,
                    state=np.column_stack(physical_cols),
                    success=result.success,
                    message=result.message,
                )

    _plot_reactor_result(
        result=plot_result,
        components=components,
        save_path=save_path,
        show=show,
        title_prefix="CSTR Reactor",
    )


def plot_pfr_reactor_result(
    result: PFRReactorResult,
    components: Iterable[Any] | None = None,
    save_path: Path | None = None,
    show: bool = True,
) -> None:
    _plot_reactor_result(
        result=result,
        components=components,
        save_path=save_path,
        show=show,
        title_prefix="PFR Reactor",
    )


def plot_pbr_reactor_result(
    result: PBRReactorResult,
    components: Iterable[Any] | None = None,
    save_path: Path | None = None,
    show: bool = True,
    x_axis: Literal["auto", "volume", "time"] = "auto",
    reactor: Any | None = None,
) -> None:
    x_values = None
    x_label = None

    if x_axis == "time":
        if hasattr(result, "time"):
            x_values = result.time
            x_label = "Time (s)"
        elif reactor is not None and hasattr(result, "volume"):
            core = getattr(reactor, "pbr_reactor_core", None)
            thermo_source = getattr(reactor, "thermo_source", None)
            reactor_model = getattr(reactor, "reactor", None)

            if core is None or thermo_source is None or reactor_model is None:
                raise ValueError(
                    "x_axis='time' requires either result.time or a valid `reactor` object."
                )
            phase = getattr(core, "phase", None)
            if phase != "gas":
                raise ValueError(
                    "Automatic x_axis='time' estimation is currently supported only for gas-phase PBR."
                )

            q_in = thermo_source.calc_gas_volumetric_flow_rate(
                molar_flow_rate=core._F_in_total,
                temperature=core._T_in,
                pressure=core._P0,
                R=reactor_model.R,
                gas_model=core.gas_model,
            )
            q_in_value = max(float(q_in), 1e-30)
            x_values = np.asarray(result.volume, dtype=float) / q_in_value
            x_label = "Time (s)"
        else:
            raise ValueError(
                "x_axis='time' requires result.time or `reactor` to estimate residence time."
            )
    elif x_axis == "volume":
        if not hasattr(result, "volume"):
            raise ValueError("x_axis='volume' is not available for this result object.")
        x_values = result.volume
        x_label = "Reactor volume coordinate (m3)"

    _plot_reactor_result(
        result=result,
        components=components,
        save_path=save_path,
        show=show,
        title_prefix="PBR Reactor",
        x_values=x_values,
        x_label=x_label,
    )
