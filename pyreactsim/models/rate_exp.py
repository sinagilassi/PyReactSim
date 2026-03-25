# import libs
# annotations
from __future__ import annotations
from pydantic import BaseModel, Field, PrivateAttr, model_validator
from typing import Dict, List, Literal, Optional, Callable
from pythermodb_settings.models import Component, CustomProperty, Pressure, Temperature, ComponentKey
from pythermodb_settings.utils import measure_time, set_component_id
from pyreactlab_core.models.reaction import Reaction
from .rate import ReactionRate
# locals
from .rate_exp_refs import rArgs, rParams, rRet, rXs

# SECTION: Rate Expression


class ReactionRateExpression(BaseModel):
    name: str = Field(
        ...,
        description="The name of the reaction rate expression. This should be unique and descriptive of the reaction it represents."
    )
    basis: Literal['concentration', 'pressure'] = Field(
        ...,
        description="The basis for the reaction rate expression, either 'concentration' or 'pressure'."
    )
    components: List[Component] = Field(
        ...,
        description="The list of components involved in the reaction for which the rate expression is defined."
    )
    component_key: ComponentKey = Field(
        ...,
        description="The key to use for identifying components in the reaction rate expression. This should match the keys used in the state (Xi) and parameters (rParams) dictionaries."
    )
    reaction: Reaction = Field(
        ...,
        description="The reaction for which the rate expression is defined."
    )
    params: rParams = Field(
        default_factory=dict,
        description="A dictionary of parameters that may be used in the rate expression."
    )
    args: rArgs = Field(
        default_factory=dict,
        description="A dictionary of arguments that may be used in the rate expression, such as temperature and pressure."
    )
    ret: rRet = Field(
        ...,
        description="A dictionary defining the expected return values from the rate expression calculation."
    )
    state: rXs = Field(
        default_factory=dict,
        description="Component state template keyed by component id. Use this to define reaction orders and default values."
    )
    state_key: ComponentKey = Field(
        default='Formula-State',
        description="The key to use for identifying components in the state (Xi) dictionary. This should match the keys used in the reaction and parameters (rParams) dictionaries."
    )
    eq: Callable[[rXs, rArgs, rParams], rRet] = Field(
        ...,
        description="A callable that takes the state (Xi), arguments (rArgs), parameters (rParams) and returns (rRet)."
    )
    _reaction_rate: ReactionRate = PrivateAttr()

    @model_validator(mode="after")
    def validate_state(self):
        """
        State must be defined based on formula-state as CO2-g, CO2-l, etc. This is required to ensure that the correct component states are used in the rate expression calculation.
        """
        components = self.components
        # >> create formula-state
        component_ids = [
            set_component_id(
                comp,
                self.state_key
            ) for comp in components
        ]

        state = self.state
        # >> keys
        state_keys = set(state.keys())

        # >> check that state keys match component ids based on component_key
        for s in state_keys:
            if s not in component_ids:
                raise ValueError(
                    f"State key '{s}' does not match any component id based on the provided component_key '{self.component_key}'. Please ensure that state keys are defined based on the component ids using the specified component_key."
                )

        return self

    @model_validator(mode="after")
    def init(self):
        # init
        self._reaction_rate = ReactionRate(
            basis=self.basis,
            components=self.components,
            reaction=self.reaction,
            params=self.params,
            args=self.args,
            returns=self.ret,
            eq=self.eq,
            state=self.state,
            component_key=self.component_key
        )

        return self

    @measure_time
    def calc(
            self,
            xi: Dict[str, CustomProperty],
            *,
            args: Optional[rArgs] = None,
            temperature: Optional[Temperature] = None,
            pressure: Optional[Pressure] = None,
            **kwargs
    ) -> rRet:
        """
        Update the reaction rate based on the provided state (xi) either concentration or pressure.
        """
        # NOTE: Check xi keys to ensure they match the expected component ids based on the state_key
        # component ids
        component_ids = [set_component_id(
            comp, self.state_key) for comp in self.components]
        # xi keys
        xi_keys = set(xi.keys())
        # >> check
        for id_ in xi_keys:
            if id_ not in component_ids:
                raise ValueError(
                    f"xi key '{id_}' does not match any component id based on the provided state_key '{self.state_key}'. Please ensure that xi keys are defined based on the component ids using the specified state_key."
                )

        # NOTE: calculate rate
        return self._reaction_rate.calc(
            xi=xi,
            args=args,
            temperature=temperature,
            pressure=pressure
        )
