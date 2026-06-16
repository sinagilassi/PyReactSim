# ThermoSource

`ThermoSource` is the public orchestration class for thermodynamic source setup.
It does not directly extract every property itself. Instead, it builds three
specialized helper objects and passes them into `ThermoSourceCore`, which exposes
the calculation API used by reactor simulations.

## Construction Diagram

```mermaid
flowchart TD
    A["Caller creates ThermoSource"] --> B["Store components and component_refs"]

    B --> C["ThermoReaction"]
    B --> D["ThermoModelSource"]
    B --> E["ThermoCustomInputs"]

    C --> C1["Extract Reaction objects from reaction_rates"]
    C --> C2["Build HSG reactions when model_source exists"]

    D --> D1["Read source/model_source"]
    D --> D2["Apply MODEL_SOURCE_ATTR_CONFIG"]
    D2 --> D3["Extract equation sources<br/>Cp_IG, Cp_LIQ, rho_LIQ"]
    D2 --> D4["Extract data sources<br/>EnFo_IG_298, MW"]

    E --> E1["Read custom_inputs"]
    E --> E2["Apply CUSTOM_INPUTS_ATTR_CONFIG"]
    E2 --> E3["Configure component properties<br/>Cp_IG, Cp_LIQ, rho_LIQ, MW, EnFo_IG_298"]
    E2 --> E4["Configure constants<br/>dH_rxn, rho_LIQ_MIX, mixture heat capacities"]

    C --> F["ThermoSourceCore.__init__"]
    D --> F
    E --> F

    F --> G["ThermoCalc.__init__"]
    F --> H["ThermoSourceLauncher.__init__"]
    F --> I["Set shared state<br/>components, source, options, refs, reactions"]

    H --> H1["ThermoPropertyFields"]
    H --> H2["SourceUtils"]

    I --> J["ThermoSource instance ready"]
    J --> K["initialize()"]
    K --> L["launch()"]
```

## Initialization / Property Assignment Flow

After construction, call `initialize()` to run the source assignment step.

```mermaid
flowchart TD
    A["ThermoSource.initialize()"] --> B["ThermoSourceLauncher.launch()"]
    B --> C["Loop over AVAILABLE_VARIABLES"]
    C --> D["Find property config<br/>custom inputs or model source"]
    D --> E["Read reactor option source selector<br/>example: gas_heat_capacity_source"]

    E --> F{Selected source?}
    F -->|"custom_inputs"| G["Use CUSTOM_INPUTS_ATTR_CONFIG"]
    F -->|"model_source"| H["Use MODEL_SOURCE_ATTR_CONFIG"]
    F -->|"missing/other"| I["Skip property"]

    G --> J["Read assigner_mode"]
    H --> J

    J --> K{Assigner mode}
    K -->|"data"| L["Assign values array<br/>*_comp mapping<br/>*_src source"]
    K -->|"equation"| L
    K -->|"constant"| M["Assign single CustomProp"]
    K -->|"constants"| N["Assign dict of constants"]

    L --> O["ThermoSource property fields populated"]
    M --> O
    N --> O
```

## How `ThermoSource` Works

`ThermoSource` receives:

- `components`: component definitions used throughout the thermodynamic model.
- `source`: the `pyThermoLinkDB` source object used to retrieve equation/data sources.
- `model_source`: optional source model used for HSG and enthalpy calculations.
- `custom_inputs`: optional user-provided constants or component-wise properties.
- `reactor_options`: controls phase, property modes, and where each property should come from.
- `heat_transfer_options`: controls whether heat-capacity and enthalpy calculations are needed.
- `reaction_rates`: kinetic expressions that contain the underlying reactions.
- `component_refs`: normalized component ids, formula/state ids, mapper, and index map.
- `component_key`: the key convention used to match components and property data.

The constructor performs these steps:

1. It stores `components` and `component_refs` on the instance.
2. It creates `ThermoReaction`, which extracts `Reaction` objects from
   `reaction_rates` and builds HSG reactions when `model_source` is available.
3. It creates `ThermoModelSource`, which extracts property equation/data sources
   from the external `source` according to `MODEL_SOURCE_ATTR_CONFIG` and
   `MODEL_SOURCE_CRITERIA`.
4. It creates `ThermoCustomInputs`, which validates and converts properties from
   `custom_inputs` according to `CUSTOM_INPUTS_ATTR_CONFIG` and
   `CUSTOM_INPUTS_CRITERIA`.
5. It initializes `ThermoSourceCore` with the three helper objects.

`ThermoSourceCore` is where the final runtime object is assembled. It:

- initializes `ThermoCalc`;
- initializes `ThermoSourceLauncher`;
- stores reactor, heat-transfer, source, reaction, and component-reference state;
- exposes reaction lists through `self.reactions` and `self.hsg_reactions`;
- copies important reactor settings such as phase, density mode, heat-capacity
  mode, formation-enthalpy source, and reaction-enthalpy mode.

## Source Selection Logic

The property source is not hard-coded in `ThermoSource`. It is selected from
`reactor_options`.

For example:

- `gas_heat_capacity_source == "model_source"` uses model-source configuration
  and usually stores an equation source in `Cp_IG_src`.
- `gas_heat_capacity_source == "custom_inputs"` uses custom inputs and stores
  numeric values in `Cp_IG`, `Cp_IG_comp`, and `Cp_IG_src`.
- `liquid_density_source`, `molecular_weight_source`,
  `ideal_gas_formation_enthalpy_source`, and other source selector fields work
  the same way.

The criteria blocks decide whether a property is configured. Typical checks are:

- phase must match, such as `gas` or `liquid`;
- heat transfer mode must match, such as `non-isothermal`;
- property mode must match, such as `constant` or `temperature-dependent`;
- selected source must match, such as `custom_inputs` or `model_source`.

## Runtime Calculations

Once initialized, the object can calculate or return thermodynamic properties.
Important methods live in `ThermoSourceCore`, including:

- `calc_Cp_IG(temperature)`: ideal-gas heat capacity.
- `calc_Cp_LIQ(temperature)`: liquid heat capacity.
- `calc_rho_LIQ(temperature, operation_mode=None)`: liquid density.
- `calc_dCp_IG()`: reaction heat-capacity changes from stoichiometry.
- `calc_dH_rxns_298()`: reaction enthalpies at 298 K from formation enthalpies.
- `calc_dH_rxns_IG(temperature)`: ideal-gas reaction enthalpies at temperature.
- `calc_dH_rxns_LIQ(temperature)`: liquid-phase reaction enthalpy path.
- `calc_En_IG(temperature)`: component ideal-gas enthalpies.

The calculation methods choose between configured constants and executable
equation sources. For example, `calc_Cp_IG()`:

```mermaid
flowchart TD
    A["calc_Cp_IG(temperature)"] --> B{heat_transfer_mode}
    B -->|"isothermal"| C["Return empty array"]
    B -->|"non-isothermal"| D{gas_heat_capacity_mode}
    D -->|"temperature-dependent"| E["Execute Cp_IG_src equations at T"]
    D -->|"constant"| F["Return configured Cp_IG array"]
    D -->|"other"| G["Raise ValueError"]
```

## Main Responsibility Split

```mermaid
classDiagram
    class ThermoSource {
        +__init__(...)
    }

    class ThermoSourceCore {
        +initialize()
        +calc_Cp_IG(temperature)
        +calc_Cp_LIQ(temperature)
        +calc_rho_LIQ(temperature, operation_mode)
        +calc_dH_rxns_298()
        +calc_En_IG(temperature)
    }

    class ThermoSourceLauncher {
        +launch()
    }

    class ThermoModelSource {
        +prop_eq_src(prop_name)
        +prop_dt_src(component_ids, prop_name)
    }

    class ThermoCustomInputs {
        -_launch_property_configuration()
        -_config_property_source(...)
        -_config_property_constant(...)
    }

    class ThermoReaction {
        +build_reactions()
        +build_stoichiometry()
        +build_stoichiometry_matrix()
        +get_reaction_names()
        +get_reaction_index()
    }

    ThermoSource --|> ThermoSourceCore
    ThermoSourceCore --|> ThermoSourceLauncher
    ThermoSource --> ThermoReaction
    ThermoSource --> ThermoModelSource
    ThermoSource --> ThermoCustomInputs
    ThermoSourceCore --> ThermoReaction
    ThermoSourceCore --> ThermoModelSource
    ThermoSourceCore --> ThermoCustomInputs
```
