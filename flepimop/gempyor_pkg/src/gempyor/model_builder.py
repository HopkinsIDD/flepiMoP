from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict, model_validator
import numpy as np

from gempyor.builder_utilities import build_initial_array


class ModelBuilder(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # === Input configuration ===
    config_path: Path
    ic_file: Path
    compartment_labels: list[str]
    n_nodes: int

    # === Column names in IC file
    node_col: str = "subpop"
    comp_col: str = "compartment"
    count_col: str = "count"

    # === Config flags
    allow_missing_subpops: bool = True
    allow_missing_compartments: bool = True
    proportional_ic: bool = False
    allowed_node_labels: list[str] | None = None

    # === Output arrays and mappings
    initial_array: np.ndarray = Field(default=None, exclude=True)
    compartment_to_index: dict[str, int] = Field(default=None, exclude=True)
    index_to_compartment: dict[int, str] = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def build(self) -> "ModelBuilder":
        self.compartment_to_index = {
            name: i for i, name in enumerate(self.compartment_labels)
        }
        self.index_to_compartment = {
            i: name for i, name in enumerate(self.compartment_labels)
        }

        self.initial_array = build_initial_array(
            ic_file=self.ic_file,
            compartment_labels=self.compartment_labels,
            n_nodes=self.n_nodes,
            node_col=self.node_col,
            comp_col=self.comp_col,
            count_col=self.count_col,
            allow_missing_subpops=self.allow_missing_subpops,
            allow_missing_compartments=self.allow_missing_compartments,
            proportional_ic=self.proportional_ic,
            allowed_node_labels=self.allowed_node_labels,
        )
        return self
