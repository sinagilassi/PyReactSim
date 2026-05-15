# import libs
import logging
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Literal, cast
from pythermodb_settings.models import Component, ComponentKey, Pressure, Temperature, Volume, CustomProperty
from pyreactsim_core.models import ReactionRateExpression
# ! locals
from .cstrc import CSTRReactorCore
from .brc import BatchReactorCore
from .pfrc import PFRReactorCore
from .pbrc import PBRReactorCore
from ..utils.reaction_tools import stoichiometry_mat_key, stoichiometry_mat
from ..utils.thermo_tools import calc_total_heat_capacity, calc_rxn_heat_generation
from ..sources.thermo_source import ThermoSource
from ..models import GasModel

# NOTE: logger setup
logger = logging.getLogger(__name__)


class ReactorAuxiliary:
    """
    Reactor Auxiliary class for calculating various variables and properties related to the reactor simulation. This class serves as a helper for performing calculations that are auxiliary to the main reactor core computations, such as calculating auxiliary variables, properties, or any other relevant information needed for the simulations.
    """
    # NOTE: Properties
    # reference temperature
    T_ref = Temperature(value=298.15, unit="K")
    # reference pressure
    P_ref = Pressure(value=101325, unit="Pa")
    # universal gas constant J/mol.K
    R = 8.314

    def __init__(
        self,
        components: List[Component],
        reaction_rates: List[ReactionRateExpression],
        thermo_source: ThermoSource,
        reactor_core: BatchReactorCore | CSTRReactorCore | PFRReactorCore | PBRReactorCore,
        component_key: ComponentKey,
    ):
        """
        Initializes the ReactorAuxiliary instance with the provided components, model inputs, component references, and component key.

        Parameters
        ----------
        components : List[Component]
            A list of Component objects representing the chemical components involved in the reactor simulation.
        reaction_rates : List[ReactionRateExpression]
            A list of ReactionRateExpression objects representing the reaction rate expressions for the reactions occurring in the reactor simulation.
        thermo_source : ThermoSource
            A ThermoSource object that provides thermodynamic data and properties for the components involved in the reactor simulation.
        reactor_core : BatchReactorCore | CSTRReactorCore | PFRReactorCore | PBRReactorCore
            An instance of the reactor core class (BatchReactorCore, CSTRReactorCore, PFRReactorCore, or PBRReactorCore) that contains the core configuration and properties of the reactor simulation.
        component_key : ComponentKey
            A ComponentKey object that serves as a key for identifying and categorizing the components in the reactor simulation.
        """
        # set
        self.components = components
        self.reaction_rates = reaction_rates
        self.thermo_source = thermo_source
        self.reactor_core = reactor_core
        self.component_key = component_key

        # SECTION: Reaction rates
        self.reaction_rates = reaction_rates
        # >> build reactions
        self.reactions = self.thermo_source.thermo_reaction.build_reactions()
        # >>> build stoichiometry matrix
        self.reaction_stoichiometry: List[Dict[str, float]] = stoichiometry_mat_key(
            reactions=self.reactions,
            component_key=component_key
        )
        # >> matrix
        self.reaction_stoichiometry_matrix = stoichiometry_mat(
            reactions=self.reactions,
            components=self.components,
            component_key=component_key,
        )

        # SECTION: component references
        self.component_num = self.thermo_source.component_refs['component_num']
        self.component_ids = self.thermo_source.component_refs['component_ids']
        self.component_formula_state = self.thermo_source.component_refs[
            'component_formula_state'
        ]
        self.component_mapper = self.thermo_source.component_refs['component_mapper']
        self.component_id_to_index = self.thermo_source.component_refs['component_id_to_index']

        # SECTION: Thermo inputs
        self.Cp_IG_MIX_TOTAL = self.thermo_source.thermo_model_inputs.Cp_IG_MIX_TOTAL
        self.Cp_IG_MIX_TOTAL_MODE = "constant" if self.Cp_IG_MIX_TOTAL is not None else "calculate"

        # >> check
        if self.reactor_core.use_gas_mixture_total_heat_capacity:
            if self.Cp_IG_MIX_TOTAL is None:
                raise ValueError(
                    "Cp_IG_MIX_TOTAL must be provided in the thermo model inputs when use_gas_mixture_total_heat_capacity is True."
                )

    # SECTION: Calculate rates

    def _calc_rates(
        self,
        partial_pressures: Dict[str, CustomProperty],
        concentration: Dict[str, CustomProperty],
        temperature: Temperature,
        pressure: Pressure
    ):
        """
        Calculate reaction rates for each reaction based on the current partial pressures and temperature.

        Parameters
        ----------
        partial_pressures : Dict[str, CustomProperty]
            Partial pressure of the components in the reactor (in Pa).
        concentration : Dict[str, CustomProperty]
            Concentration of the components in the reactor (in mol/m3).
        temperature : Temperature
            Current temperature of the system (in K).
        pressure : Pressure
            Total pressure of the system (in Pa).

        Returns
        -------
        np.ndarray
            An array of reaction rates for each reaction in the reactor, calculated based on the current partial pressures and temperature.
        """
        # ! r_k = k(T, P_i) for each reaction k
        rates = []

        # iterate over reaction rate expressions
        for rate_exp in self.reaction_rates:
            # >> check basis
            basis = rate_exp.basis

            # >> calculate rate for reaction
            if basis == "pressure":
                # >> calculate rate based on partial pressures
                r_k = rate_exp.calc(
                    xi=partial_pressures,
                    temperature=temperature,
                    pressure=pressure
                )
            elif basis == "concentration":
                # >> calculate rate based on concentrations
                r_k = rate_exp.calc(
                    xi=concentration,
                    temperature=temperature,
                    pressure=pressure
                )
            else:
                raise ValueError(
                    f"Invalid basis '{basis}' for reaction rate expression '{rate_exp.name}'. Must be 'pressure' or 'concentration'."
                )

            # extract rate value
            r_k_value = r_k.value
            # append to rates list
            rates.append(r_k_value)

        # >> to array
        rates = np.array(rates, dtype=float)

        return rates

    # SECTION: Calculating total heat capacity of gas mixture
    def _calc_total_heat_capacity(
            self,
            n: np.ndarray,
            temperature: Temperature,
            mode: Literal['calculate', 'constant']
    ) -> float:
        """
        Calculate the total heat capacity of the gas mixture based on the moles of each component and the temperature.

        Parameters
        ----------
        n : np.ndarray
            Array of moles of each component in the reactor.
        temperature : Temperature
            Current temperature of the system (in K).

        Returns
        -------
        float
            The total heat capacity of the gas mixture (in J/K).
        """

        if mode == "constant":
            # NOTE: if use_gas_mixture_total_heat_capacity is True, use constant heat capacity from model source
            if self.Cp_IG_MIX_TOTAL is None:
                raise ValueError(
                    "Cp_IG_MIX_TOTAL must be provided in the thermo model inputs when use_gas_mixture_total_heat_capacity is True."
                )

            return float(self.Cp_IG_MIX_TOTAL.value)

        elif mode == "calculate":
            # NOTE: calculate total heat capacity of gas mixture based on individual component heat capacities and moles
            # ??? Cp_i(T)
            Cp_IG_values = self.thermo_source.calc_Cp_IG(
                temperature=temperature
            )

            # ??? Σ_i n_i Cp_i
            # unit check: n_i [mol], Cp_i [J/mol.K] => n_i * Cp_i [J/K]
            Cp_IG_MIX_TOTAL = calc_total_heat_capacity(n, Cp_IG_values)

            if Cp_IG_MIX_TOTAL <= 1e-16:
                raise ValueError("Total heat capacity is too small or zero.")

            return Cp_IG_MIX_TOTAL
        else:
            raise ValueError(
                f"Invalid mode '{mode}' for calculating total heat capacity. Must be 'calculate' or 'constant'."
            )

    # SECTION: Calculate concentration from moles and reactor volume
    def _calc_concentration(
            self,
            n: np.ndarray,
            reactor_volume: float
    ) -> Tuple[np.ndarray, Dict[str, CustomProperty], float]:
        """
        Calculate the concentration of each component in the reactor based on the moles and reactor volume.

        Parameters
        ----------
        n : np.ndarray
            Array of moles of each component in the reactor.
        reactor_volume : float
            Volume of the reactor (in m3).

        Returns
        -------
        Tuple[np.ndarray, Dict[str, CustomProperty], float]
            A tuple containing:
            - An array of concentrations for each component (in mol/m3).
            - A dictionary of concentrations for each component as CustomProperty objects (in mol/m3).
            - The total concentration of the system (in mol/m3).
        """
        # ! C_i = n_i / V
        # unit check: n_i [mol], V [m3] => C_i [mol/m3]
        concentration = n / reactor_volume

        # total concentration
        # ! C_total = N_total / V
        n_total = np.sum(n)
        concentration_total = n_total / reactor_volume

        # NOTE: create ids for concentration array
        conc_ids = [
            sp for sp in self.component_formula_state
        ]

        # std concentration as dict
        concentration_std = {
            sp: CustomProperty(
                value=conc,
                unit="mol/m3",
                symbol="C"
            ) for sp, conc in zip(conc_ids, concentration)
        }

        return concentration, concentration_std, concentration_total

    # SECTION: Building xi (partial pressure)
    def _calc_partial_pressure(
        self,
        n_total: float,
        y_mole: np.ndarray,
        T: float,
        _V0: float,
        _P0: float,
        gas_model: GasModel,
        operation_mode: str
    ):
        """
        Calculate the partial pressures of the components based on the total moles, mole fractions, and temperature.

        Parameters
        ----------
        n_total : float
            Total moles of all components in the reactor.
        y_mole : np.ndarray
            Mole fractions of the components in the reactor.
        T : float
            Current temperature of the system (in K).
        _V0 : float
            Initial/reference volume of the reactor (in m3).
        _P0 : float
            Initial/reference pressure of the reactor (in Pa).
        gas_model : GasModel
            The gas model to be used for calculating the total pressure and volume based on the ideal gas law or any other relevant gas model.
        operation_mode : str
            The operation mode of the reactor, which determines how the total pressure and volume are calculated based on the ideal gas law or any other relevant gas model.

        Returns
        -------
        Tuple[Dict[str, CustomProperty], Dict[str, CustomProperty], float]
            A tuple containing:
            - A dictionary of partial pressures for each component (in Pa).
            - A dictionary of partial pressures for each component as CustomProperty objects (in Pa).
            - The total pressure of the system (in Pa).
        """
        # ! calculate total pressure using ideal gas law: P = N_total * R * T / V
        # ! unit check: N_total [mol], R [J/mol.K], T [K], V [m3] => P [Pa]
        if operation_mode == "constant_volume":
            # ??? Constant volume assumption: V = V0
            reactor_volume = _V0

            # NOTE: calculate total pressure
            # ! P_total = f(n_total(t), P(t))
            p_total = self.thermo_source.calc_tot_pressure(
                n_total=n_total,
                temperature=T,
                reactor_volume_value=reactor_volume,
                R=self.R,
                gas_model=cast(GasModel, gas_model)
            )
        elif operation_mode == "constant_pressure":
            # ??? Constant pressure assumption: P = P0
            p_total = _P0

            # NOTE: calculate volume
            # ! V(t) = f(n_total(t), T(t))
            reactor_volume = self.thermo_source.calc_gas_volume(
                n_total=n_total,
                temperature=T,
                pressure=p_total,
                R=self.R,
                gas_model=cast(GasModel, gas_model)
            )
        else:
            raise ValueError(
                f"Invalid operation mode '{operation_mode}'. Must be constant pressure or volume."
            )

        # NOTE: partial pressures:
        # ! P_i = y_i * P_total
        partial_pressures = {
            sp: y_mole[i] * p_total for i, sp in enumerate(self.component_formula_state)
        }

        # NOTE: standardize partial pressures to be used in rate calculations:
        # ??? r[k] = k(T, P_i) for each reaction k
        # >> std partial pressures
        partial_pressures_std = {}

        for k, v in partial_pressures.items():
            partial_pressures_std[k] = CustomProperty(
                value=v,
                unit="Pa",
                symbol="P"
            )

        return partial_pressures, partial_pressures_std, p_total, reactor_volume
