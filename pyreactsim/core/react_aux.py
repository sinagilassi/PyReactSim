# import libs
import logging
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Literal, cast
from pythermodb_settings.models import Component, ComponentKey, Pressure, Temperature, Volume, CustomProperty
from pyreactsim_core.models import ReactionRateExpression
# ! locals
from ..configs.constants import R_J_per_mol_K
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
    # ! universal gas constant [J/mol.K]
    R = R_J_per_mol_K

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
        # stoichiometry matrix dimensions: rows = reactions, columns = components, values = ν_i,j
        self.stoichiometry_matrix = self.thermo_source.thermo_reaction.stoichiometry_matrix

        # SECTION: component references
        self.component_num = self.thermo_source.component_refs['component_num']
        self.component_ids = self.thermo_source.component_refs['component_ids']
        self.component_formula_state = self.thermo_source.component_refs[
            'component_formula_state'
        ]
        self.component_mapper = self.thermo_source.component_refs['component_mapper']
        self.component_id_to_index = self.thermo_source.component_refs['component_id_to_index']

        # SECTION: Thermo inputs
        # ! basis
        self.Cp_LIQ_MIX_TOTAL_BASIS = None

        # ! Cp_IG_MIX_TOTAL: total heat capacity of gas mixture (in J/K)
        self.Cp_IG_MIX_TOTAL = self.thermo_source.thermo_model_inputs.Cp_IG_MIX_TOTAL
        self.Cp_IG_MIX_TOTAL_MODE = "constant" if self.Cp_IG_MIX_TOTAL is not None else "calculate"

        # >> check
        if self.reactor_core.use_gas_mixture_total_heat_capacity:
            if self.Cp_IG_MIX_TOTAL is None:
                raise ValueError(
                    "Cp_IG_MIX_TOTAL must be provided in the thermo model inputs when use_gas_mixture_total_heat_capacity is True."
                )

        # ! Cp_LIQ_MIX_TOTAL: total heat capacity of liquid mixture (in J/K)
        self.Cp_LIQ_MIX_TOTAL = self.thermo_source.thermo_model_inputs.Cp_LIQ_MIX_TOTAL
        self.Cp_LIQ_MIX_TOTAL_MODE = "constant" if self.Cp_LIQ_MIX_TOTAL is not None else "calculate"

        # >> check
        if self.reactor_core.use_liquid_mixture_total_heat_capacity:
            if self.Cp_LIQ_MIX_TOTAL is None:
                raise ValueError(
                    "Cp_LIQ_MIX_TOTAL must be provided in the thermo model inputs when use_liquid_mixture_total_heat_capacity is True."
                )
            # set
            self.Cp_LIQ_MIX_TOTAL_BASIS = "molar"

        # ! Cp_LIQ_MIX_VOLUMETRIC: volumetric heat capacity of liquid mixture (in J/K.m3)
        self.Cp_LIQ_MIX_VOLUMETRIC = self.thermo_source.thermo_model_inputs.Cp_LIQ_MIX_VOLUMETRIC
        self.Cp_LIQ_MIX_VOLUMETRIC_MODE = "constant" if self.Cp_LIQ_MIX_VOLUMETRIC is not None else "calculate"

        # >> check
        if self.reactor_core.use_liquid_mixture_volumetric_heat_capacity:
            if self.Cp_LIQ_MIX_VOLUMETRIC is None:
                raise ValueError(
                    "Cp_LIQ_MIX_VOLUMETRIC must be provided in the thermo model inputs when use_liquid_mixture_volumetric_heat_capacity is True."
                )
            # set
            self.Cp_LIQ_MIX_VOLUMETRIC_BASIS = "volumetric"

        # ! dH_rxns
        if self.reactor_core.reaction_enthalpy_mode == "reaction":
            self.dH_rxns = self.thermo_source.set_dH_rxns()
        else:
            self.dH_rxns = None

    # SECTION: Calculate rates

    def _calc_rates(
        self,
        partial_pressures: Dict[str, CustomProperty],
        concentration: Dict[str, CustomProperty],
        temperature: Temperature,
        pressure: Pressure,
        args: Optional[Dict[str, CustomProperty]] = None,
    ) -> np.ndarray:
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

        Notes
        -----
        - basis='pressure' -> xi = partial pressures
        - basis='concentration' -> xi = concentrations
        - The args parameter contains the bulk density rho_B for packed-bed conversion, which is passed to the rate expression calculations.
        - The final rate unit is mol/m3.s after conversion, which is suitable for the species balance equations in the PBR model.
        - The raw rates are on catalyst-mass basis r' [mol/kg.s] and will be converted to reactor-volume basis r_V [mol/m3.s] via r_V = rho_B * r'.
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
                    args=args,
                    temperature=temperature,
                    pressure=pressure
                )
            elif basis == "concentration":
                # >> calculate rate based on concentrations
                r_k = rate_exp.calc(
                    xi=concentration,
                    args=args,
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

    def _calc_rates_concentration_basis(
        self,
        concentration: Dict[str, CustomProperty],
        temperature: Temperature,
        args: Optional[Dict[str, CustomProperty]] = None
    ):
        """
        Calculate reaction rates in mol/m3.s for each reaction based on the current partial pressures and temperature.

        Parameters
        ----------
        concentration : Dict[str, CustomProperty]
            Concentration of the components in the reactor (in mol/m3).
        temperature : Temperature
            Current temperature of the system (in K).
        args : Optional[Dict[str, CustomProperty]], optional
            Additional arguments that may be needed for rate calculations, such as pressure or any other relevant properties. This is optional and can be used to provide additional information for the rate calculations.

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
            if basis == "concentration":
                # >> calculate rate based on concentrations
                r_k = rate_exp.calc(
                    xi=concentration,
                    args=args,
                    temperature=temperature,
                    pressure=None
                )
            else:
                raise ValueError(
                    f"Invalid basis '{basis}' for reaction rate expression '{rate_exp.name}'. Must be 'concentration'."
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

    # NOTE: calculate total heat capacity of liquid mixture
    def _calc_total_heat_capacity_liquid(
            self,
            n: np.ndarray,
            temperature: Temperature,
            reactor_volume: float,
            mode: Literal['calculate', 'constant'],
    ) -> float:
        """
        Calculate the total heat capacity of the liquid mixture based on the moles of each component and the temperature.

        Parameters
        ----------
        n : np.ndarray
            Array of moles of each component in the reactor.
        temperature : Temperature
            Current temperature of the system (in K).
        reactor_volume : float
            Volume of the reactor (in m3).
        mode : Literal['calculate', 'constant']
            The mode for calculating the total heat capacity. If 'calculate', it will calculate based on individual component heat capacities and moles. If 'constant', it will use a constant value provided in the model inputs.

        Returns
        -------
        float
            The total heat capacity of the liquid mixture (in J/K).
        """
        if mode == "constant":
            if self.Cp_LIQ_MIX_TOTAL_BASIS == "molar":
                # NOTE: if use_gas_mixture_total_heat_capacity is True, use constant heat capacity from model source
                if self.Cp_LIQ_MIX_TOTAL is None:
                    raise ValueError(
                        "Cp_LIQ_MIX_TOTAL must be provided in the thermo model inputs when use_liquid_mixture_total_heat_capacity is True."
                    )

                return float(self.Cp_LIQ_MIX_TOTAL.value)
            elif self.Cp_LIQ_MIX_TOTAL_BASIS == "volumetric":
                # NOTE: if use_liquid_mixture_volumetric_heat_capacity is True, use constant volumetric heat capacity from model source
                if self.Cp_LIQ_MIX_VOLUMETRIC is None:
                    raise ValueError(
                        "Cp_LIQ_MIX_VOLUMETRIC must be provided in the thermo model inputs when use_liquid_mixture_volumetric_heat_capacity is True."
                    )

                return float(self.Cp_LIQ_MIX_VOLUMETRIC.value)
            else:
                raise ValueError(
                    f"Invalid basis '{self.Cp_LIQ_MIX_TOTAL_BASIS}' for constant liquid mixture heat capacity. Must be 'molar' or 'volumetric'."
                )

        elif mode == "calculate":
            # NOTE: calculate total heat capacity of liquid mixture based on individual component heat capacities and moles
            # ??? Cp_i(T)
            Cp_LIQ_values = self.thermo_source.calc_Cp_LIQ(
                temperature=temperature
            )

            # ??? Σ_i n_i Cp_i
            # unit check: n_i [mol], Cp_i [J/mol.K] => n_i * Cp_i [J/K]
            Cp_LIQ_MIX_TOTAL = calc_total_heat_capacity(n, Cp_LIQ_values)

            if Cp_LIQ_MIX_TOTAL <= 1e-16:
                raise ValueError("Total heat capacity is too small or zero.")

            return Cp_LIQ_MIX_TOTAL
        else:
            raise ValueError(
                f"Invalid mode '{mode}' for calculating total heat capacity. Must be 'calculate' or 'constant'."
            )

    # SECTION: Calculate volumetric heat capacity of liquid mixture

    def _calc_volumetric_heat_capacity(
            self,
            c: np.ndarray,
            temperature: Temperature,
            mode: Literal['calculate', 'constant']
    ) -> float:
        """
        Calculate the volumetric heat capacity of the liquid mixture based on the concentrations and heat capacities of the individual components.

        Parameters
        ----------
        c : np.ndarray
            Array of concentration of each component in the reactor (in mol/m3).
        temperature : Temperature
            Current temperature of the system (in K).
        mode : Literal['calculate', 'constant']
            The mode for calculating the volumetric heat capacity. If 'calculate', it will calculate based on individual component heat capacities and concentrations. If 'constant', it will use a constant value provided in the model inputs.

        Returns
        -------
        float
            The calculated volumetric heat capacity of the liquid mixture (in J/K.m3), calculated as the sum of the products of concentration and heat capacity for each component.
        """
        if mode == "constant":
            # NOTE: calculate volumetric heat capacity of liquid mixture
            if self.Cp_LIQ_MIX_VOLUMETRIC is None:
                raise ValueError(
                    "Cp_LIQ_MIX_VOLUMETRIC must be provided in the thermo model inputs when use_liquid_mixture_volumetric_heat_capacity is True."
                )

            return float(self.Cp_LIQ_MIX_VOLUMETRIC.value)
        elif mode == "calculate":
            # NOTE: calculate total heat capacity of liquid phase (Cp_LIQ) for each component
            # ??? Cp_i(T)
            # J/mol.K
            Cp_LIQ_values = self.thermo_source.calc_Cp_LIQ(
                temperature=temperature
            )

            # ! mixture volumetric heat capacity (Cp_LIQ_total) for the system
            # ??? Σ_i c_i Cp_i
            # J/K.m3 = (mol/m3) * (J/mol.K)
            Cp_LIQ_MIX_VOLUMETRIC = calc_total_heat_capacity(c, Cp_LIQ_values)

            if Cp_LIQ_MIX_VOLUMETRIC <= 1e-16:
                raise ValueError("Total heat capacity is too small or zero.")

            return Cp_LIQ_MIX_VOLUMETRIC
        else:
            raise ValueError(
                f"Invalid mode '{mode}' for calculating volumetric heat capacity. Must be 'calculate' or 'constant'."
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
        if reactor_volume <= 1e-30:
            raise ValueError(
                "Calculated liquid reactor volume is too small or non-positive."
            )

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

    # SECTION: Building xi (partial pressure) - simplified version for constant pressure or volume operation
    def _set_partial_pressure(
        self,
        y_mole: np.ndarray,
        p_total: float
    ) -> Tuple[Dict[str, float], Dict[str, CustomProperty], float]:
        """
        Set component partial pressures from mole fractions.

        Formula
        -------
        P_i = y_i * P_total
        """
        partial_pressures = {
            sp: y_mole[i] * p_total for i, sp in enumerate(self.component_formula_state)
        }
        partial_pressures_std = {
            k: CustomProperty(value=v, unit="Pa", symbol="P")
            for k, v in partial_pressures.items()
        }
        return partial_pressures, partial_pressures_std, p_total
