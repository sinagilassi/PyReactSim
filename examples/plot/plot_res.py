from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
from pyreactsim.models.br import BatchReactorResult


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


def plot_batch_reactor_result(
    result: BatchReactorResult,
    components: Iterable[Any] | None = None,
    save_path: Path | None = None,
    show: bool = True,
) -> None:
    if not result:
        raise ValueError("simulation_result is empty.")

    if not result.success:
        raise RuntimeError(
            f"Simulation failed: {result.message or 'unknown error'}")

    time = np.asarray(result.time, dtype=float)
    state = np.asarray(result.state, dtype=float)

    if state.ndim != 2:
        raise ValueError(
            "Expected result.state to be a 2D array with shape (n_states, n_time).")

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
            gridspec_kw={"hspace": 0.35}
        )
        ax_mole, ax_temp = axes[0], axes[1]
    else:
        fig, ax_mole = plt.subplots(figsize=(10, 6))
        ax_temp = None

    # ── moles subplot ──
    for idx, label in enumerate(mole_labels):
        ax_mole.plot(time, mole_state[idx], linewidth=2, label=label)
    ax_mole.set_ylabel("Moles (mol)")
    ax_mole.set_title("Batch Reactor — Species")
    ax_mole.grid(True, alpha=0.3)
    ax_mole.legend(loc="best")

    if ax_temp is not None and temp_state is not None:
        # ── temperature subplot ──
        ax_temp.plot(time, temp_state, linewidth=2,
                     color="tab:red", label="Temperature")
        ax_temp.set_xlabel("Time (s)")
        ax_temp.set_ylabel("Temperature (K)")
        ax_temp.set_title("Batch Reactor — Temperature")
        ax_temp.grid(True, alpha=0.3)
        ax_temp.legend(loc="best")
    else:
        ax_mole.set_xlabel("Time (s)")

    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200)
        print(f"Saved plot to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)
