# import libs
import logging
import numpy as np
from typing import Union
# locals
from ..models import GasModel


class ThermoCalc:
    # SECTION: EOS related methods
    # ! Calculate total pressure using ideal gas law
    def calc_tot_pressure(
            self,
            n_total: float,
            temperature: float,
            reactor_volume_value: float,
            R: float,
            gas_model: GasModel
    ) -> float:
        """
        Total pressure [Pa].
        Default: ideal gas
            P = N_total * R * T / V

        Parameters
        ----------
        n_total : float
            Total moles of gas in the reactor.
        temperature : float
            Temperature of the gas in the reactor [K].
        reactor_volume_value : float
            Volume of the reactor [m3].
        R : float
            Ideal gas constant [J/mol.K].
        gas_model : GasModel
            The gas model to use for the calculation (e.g., "ideal", "real").

        Returns
        -------
        float
            Total pressure of the gas in the reactor [Pa].
        """
        if gas_model == "real":
            # FIXME: implement real gas model
            return 0

        # ideal gas model
        return n_total * R * temperature / float(reactor_volume_value)

    # ! Calculate volume
    def calc_gas_volume(
        self,
        n_total: float,
        temperature: float,
        pressure: float,
        R: float,
        gas_model: GasModel
    ) -> float:
        """
        Calculate the volume of the gas in the reactor using the ideal gas law.
            V = N_total * R * T / P

        Parameters
        ----------
        n_total : float
            Total moles of gas in the reactor.
        temperature : float
            Temperature of the gas in the reactor [K].
        pressure : float
            Pressure of the gas in the reactor [Pa].
        R : float
            Ideal gas constant [J/mol.K].
        gas_model : GasModel
            The gas model to use for the calculation (e.g., "ideal", "real").

        Returns
        -------
        float
            Volume of the gas in the reactor [m3].
        """
        if gas_model == "real":
            # FIXME: implement real gas model
            return 0

        # ideal gas model
        return n_total * R * temperature / pressure

    def calc_liquid_volume(
            self,
            n: np.ndarray,
            molecular_weights: np.ndarray,
            density: np.ndarray
    ) -> float:
        """
        Calculate the volume of the liquid in the reactor using the formula:
            V = sigma_i (n_i * MW_i) / density_i

        Parameters
        ----------
        n : np.ndarray
            An array of moles of each component in the liquid phase.
        molecular_weights : np.ndarray
            An array of molecular weights for each component in the liquid phase [g/mol].
        density : np.ndarray
            An array of densities for each component in the liquid phase [g/m3].
        """
        # calculate volume for each component
        volumes = n * molecular_weights / density

        # total volume is the sum of the volumes of each component
        total_volume = np.sum(volumes)

        return total_volume
