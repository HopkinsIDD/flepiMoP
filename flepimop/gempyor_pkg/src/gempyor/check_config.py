import yaml
from pydantic import BaseModel, ValidationError, model_validator, Field, AfterValidator, validator
from datetime import date
from typing import Dict, List, Union, Literal, Optional, Annotated, Any
from functools import partial
from gempyor import compartments

def read_yaml(file_path: str) -> dict:
    with open(file_path, 'r') as stream:
        config = yaml.safe_load(stream)
    
    return CheckConfig(**config).model_dump()
    
def allowed_values(v, values):
    assert v in values
    return v

# def parse_value(cls, values):
#     value = values.get('value')
#     parsed_val = compartments.Compartments.parse_parameter_strings_to_numpy_arrays_v2(value)
#     return parsed_val
    
class SubpopSetupConfig(BaseModel):
    geodata: str
    mobility: Optional[str]
    selected: List[str] = Field(default_factory=list)
    # state_level: Optional[bool] = False # pretty sure this doesn't exist anymore

class InitialConditionsConfig(BaseModel):
    method: Annotated[str, AfterValidator(partial(allowed_values, values=['Default', 'SetInitialConditions', 'SetInitialConditionsFolderDraw', 'InitialConditionsFolderDraw', 'FromFile', 'plugin']))] = 'Default'
    initial_file_type: Optional[str]
    initial_conditions_file: Annotated[str, AfterValidator(partial(allowed_values, values=['Default', 'SetInitialConditions', 'SetInitialConditionsFolderDraw', 'InitialConditionsFolderDraw', 'FromFile']))] = None
    proportional: Optional[bool] = None
    allow_missing_subpops: Optional[bool] = None
    allow_missing_compartments: Optional[bool] = None
    ignore_population_checks: Optional[bool] = None
    plugin_file_path: Optional[str] = None

    @model_validator(mode='before')
    def validate_initial_file_check(cls, values):
        method = values.get('method')
        initial_conditions_file = values.get('initial_conditions_file')
        initial_file_type = values.get('initial_file_type')        
        if method in {'FromFile', 'SetInitialConditions'} and not initial_conditions_file:
            raise ValueError('An initial_conditions_file is required when method is FromFile')
        if method in {'InitialConditionsFolderDraw','SetInitialConditionsFolderDraw'} and not initial_file_type:
            raise ValueError('initial_file_type is required when method is InitialConditionsFolderDraw')
        return values
    
    @model_validator(mode='before')
    def plugin_filecheck(cls, values):
        method = values.get('method')
        plugin_file_path = values.get('plugin_file_path')   
        if method == 'plugin' and not plugin_file_path:
            raise ValueError('a plugin file path is required when method is plugin')
        return values


class SeedingConfig(BaseModel):
    method: Annotated[str, AfterValidator(partial(allowed_values, values=['NoSeeding', 'NegativeBinomialDistributed', 'PoissonDistributed', 'FolderDraw', 'FromFile', 'plugin']))] = 'NoSeeding'
    plugin_file_path: Optional[str] = None

    @model_validator(mode='before')
    def plugin_filecheck(cls, values):
        method = values.get('method')
        plugin_file_path = values.get('plugin_file_path')   
        if method == 'plugin' and not plugin_file_path:
            raise ValueError('a plugin file path is required when method is plugin')
        return values
    
class IntegrationConfig(BaseModel):
    method: Annotated[str, AfterValidator(partial(allowed_values, values=['rk4', 'rk4.jit', 'best.current', 'legacy']))] = 'rk4'
    dt: float = 2.0

class ValueConfig(BaseModel):
    distribution: str = 'fixed'
    value: Optional[float] = None
    mean: Optional[float] = None
    sd: Optional[float] = None
    a: Optional[float] = None
    b: Optional[float] = None
    # NEED TO ADD ABILITY TO PARSE PARAMETERS

    @model_validator(mode='before')
    def check_distr(cls, values):
        distr = values.get('distribution')
        value = values.get('value')
        mean = values.get('mean')
        sd = values.get('sd')
        a = values.get('a')
        b = values.get('b')
        if distr != 'fixed':
            if not mean and not sd:
                raise ValueError('mean and sd must be provided for non-fixed distributions')
            if distr == 'truncnorm' and not a and not b:
                raise ValueError('a and b must be provided for truncated normal distributions')
        return values

class BaseParameterConfig(BaseModel):
    value: Optional[ValueConfig] = None

class SeirParameterConfig(BaseParameterConfig):
    value: Optional[ValueConfig] = None
    stacked_modifier_method: Annotated[str, AfterValidator(partial(allowed_values, values=['sum', 'product', 'reduction_product']))] = None
    rolling_mean_windows: Optional[float] = None
    timeseries: Optional[str] = None

    @model_validator(mode='before')
    def which_value(cls, values):
        value = values.get('value')
        timeseries = values.get('timeseries')
        if value and timeseries:
            raise ValueError('your parameter is both a timeseries and a value, please choose one')
        return values
    
    
class TransitionConfig(BaseModel): 
    # !! sometimes these are lists of lists and sometimes they are lists... how to deal with this?
    source: List[List[str]]
    destination: List[List[str]]
    proportion_exponent: List[List[str]]
    proportional_to: List[str]

class SeirConfig(BaseModel):
    integration: IntegrationConfig # is this Optional?
    parameters: Dict[str, SeirParameterConfig]
    transitions: List[TransitionConfig]

class SinglePeriodModifierConfig(BaseModel):
    method: Literal["SinglePeriodModifier"]
    parameter: str
    period_start_date: date
    period_end_date: date
    subpop: str
    subpop_groups: Optional[str] = None
    value: ValueConfig
    perturbation: Optional[ValueConfig] = None

class MultiPeriodDatesConfig(BaseModel):
    start_date: date
    end_date: date
    
class MultiPeriodGroupsConfig(BaseModel):
    subpop: List[str]
    periods: List[MultiPeriodDatesConfig]

class MultiPeriodModifierConfig(BaseModel):
    method: Literal["MultiPeriodModifier"]
    parameter: str
    groups: List[MultiPeriodGroupsConfig]
    period_start_date: date
    period_end_date: date
    subpop: str
    value: ValueConfig
    perturbation: Optional[ValueConfig] = None

class StackedModifierConfig(BaseModel):
    method: Literal["StackedModifier"]
    modifiers: List[str]

class ModifiersConfig(BaseModel):
    scenarios: List[str]
    modifiers: Dict[str, Any]
    
    @validator("modifiers")
    def validate_data_dict(cls, value: Dict[str, Any]) -> Dict[str, Any]:
        errors = []
        for key, entry in value.items():
            method = entry.get("method")
            if method not in {"SinglePeriodModifier", "MultiPeriodModifier", "StackedModifier"}:
                errors.append(f"Invalid modifier method: {method}")
        if errors:
            raise ValueError("Errors in dictionary entries:\n" + "\n".join(errors))
        return value


class SourceConfig(BaseModel): # i think this can be incidence or prevalence, or any other source name? (this one is maybe a bit complicated to validate...)
    incidence: Dict[str, str]
    # TO FIX

    def get_source_names(self):
        source_names = []
        for key in self.incidence:
            source_names.append(key)
        return source_names  # Access keys using a loop
    # def get_source_names(self):
    #     return self.incidence.keys()

class DelayFrameConfig(BaseModel):
    source: Optional[SourceConfig] = None
    probability: Optional[BaseParameterConfig] = None
    delay: Optional[BaseParameterConfig] = None
    duration: Optional[BaseParameterConfig] = None
    name: Optional[str] = None
    sum: Optional[List[str]] = None

    # @validator("sum")
    # def validate_sum_elements(cls, value: Optional[List[str]]) -> Optional[List[str]]:
    #     if value is None:
    #         return None
    #     # source = value.get('source')
    #     source_names = {name for name in cls.source.get_source_names()}  # Get source names from source config
    #     invalid_elements = [element for element in value if element not in source_names]
    #     if invalid_elements:
    #         raise ValueError(f"Invalid elements in 'sum': {', '.join(invalid_elements)} not found in source names")
    #     return value
    # NOTE: ^^ this doesn't work yet because it needs to somehow be a level above? to access all OTHER source names

    @model_validator(mode='before')
    def check_outcome_type(cls, values):
        sum_present = values.get('sum') is not None
        source_present = values.get('source') is not None

        if sum_present and source_present:
            raise ValueError(f"Error in outcome: Both 'sum' and 'source' are present. Choose one.")
        elif not sum_present and not source_present:
            raise ValueError(f"Error in outcome: Neither 'sum' nor 'source' is present. Choose one.")
        return values
    

class OutcomesConfig(BaseModel):
    method: Literal["delayframe"] # Is this required? I don't see it anywhere in the gempyor code
    outcomes: Dict[str, DelayFrameConfig]

class ResampleConfig(BaseModel):
    aggregator: str
    freq: str
    skipna: bool = False

class LikelihoodParams(BaseModel):
    scale: float
    # are there other options here?

class LikelihoodConfig(BaseModel):
    dist: str
    params: Optional[LikelihoodParams] = None

class StatisticsConfig(BaseModel):
    name: str
    sim_var: str
    data_var: str
    aggregator: Optional[str] = None
    period: Optional[str] = None
    remove_na: Optional[bool] = None
    add_one: Optional[bool] = None
    # resample: Optional[ResampleConfig] = None
    # zero_to_one: Optional[bool] = False # is this the same as add_one? remove_na?
    likelihood: LikelihoodConfig

class InferenceConfig(BaseModel):
    method: Optional[str] = None # for now - i can only see emcee as an option here, otherwise ignored in classical - need to add these options
    iterations_per_slot: Optional[int] # i think this is optional because it is also set in command line??
    do_inference: bool 
    gt_data_path: str
    statistics: Dict[str, StatisticsConfig]
    # Need to determine here what is needed in classical vs other applications

class CheckConfig(BaseModel):
    name: str
    setup_name: Optional[str] = None
    model_output_dirname: Optional[str] = None
    start_date: date
    end_date: date
    start_date_groundtruth: Optional[date] = None
    end_date_groundtruth: Optional[date] = None
    nslots: Optional[int] = 1
    subpop_setup: SubpopSetupConfig
    compartments: Dict[str, List[str]]
    initial_conditions: Optional[InitialConditionsConfig] = None
    seeding: Optional[SeedingConfig] = None
    seir: SeirConfig
    seir_modifiers: Optional[ModifiersConfig] = None
    outcomes: Optional[OutcomesConfig] = None
    outcome_modifiers: Optional[ModifiersConfig] = None
    inference: Optional[InferenceConfig] = None

# add validator for if modifiers exist but seir/outcomes do not
    
# there is an error in the one below 
    @model_validator(mode='before')
    def verify_inference(cls, values):
        inference_present = values.get('inference') is not None
        start_date_groundtruth = values.get('start_date_groundtruth') is not None
        if inference_present and not start_date_groundtruth:
            raise ValueError('Inference mode is enabled but no groundtruth dates are provided')
        elif start_date_groundtruth and not inference_present:
            raise ValueError('Groundtruth dates are provided but inference mode is not enabled')
        return values
    
    @model_validator(mode='before')
    def check_dates(cls, values):
        start_date = values.get('start_date')
        end_date = values.get('end_date')
        if start_date and end_date:
            if end_date <= start_date:
                raise ValueError('end_date must be greater than start_date')
        return values
    
    @model_validator(mode='before')
    def init_or_seed(cls, values):
        init = values.get('initial_conditions')
        seed = values.get('seeding')
        if not init or seed:
            raise ValueError('either initial_conditions or seeding must be provided')
        return values
    
