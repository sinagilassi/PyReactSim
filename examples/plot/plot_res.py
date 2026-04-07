from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
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
) -> None:
    if not result:
        raise ValueError("simulation_result is empty.")

    if not result.success:
        raise RuntimeError(
            f"Simulation failed: {result.message or 'unknown error'}")

    # NOTE: support both time-based (Batch/CSTR) and volume-based (PFR) coordinates
    if hasattr(result, "time"):
        x = np.asarray(result.time, dtype=float)
        x_label = "Time (s)"
        species_y_label = "Moles (mol)"
    elif hasattr(result, "volume"):
        x = np.asarray(result.volume, dtype=float)
        x_label = "Reactor volume coordinate (m3)"
        species_y_label = "Molar flow rate (mol/s)"
    else:
        raise ValueError(
            "Unsupported result coordinate. Expected 'time' or 'volume'.")

    state = np.asarray(result.state, dtype=float)

    if state.ndim != 2:
        raise ValueError(
            "Expected result.state to be a 2D array with shape (n_states, n_points).")

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

    if ax_temp is not None and temp_state is not None:
        # temperature subplot
        ax_temp.plot(x, temp_state, linewidth=2,
                     color="tab:red", label="Temperature")
        ax_temp.set_xlabel(x_label)
        ax_temp.set_ylabel("Temperature (K)")
        ax_temp.set_title(f"{title_prefix} - Temperature")
        ax_temp.grid(True, alpha=0.3)
        ax_temp.legend(loc="best")
    else:
        ax_mole.set_xlabel(x_label)

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
) -> None:
    _plot_reactor_result(
        result=result,
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
) -> None:
    _plot_reactor_result(
        result=result,
        components=components,
        save_path=save_path,
        show=show,
        title_prefix="PBR Reactor",
    )
