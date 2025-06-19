"""Model metadata for gempyor."""

__all__ = ("ModelMeta",)


from datetime import datetime
from pathlib import Path
from typing import Any, Literal
import warnings

from confuse import ConfigView
import pandas as pd
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    model_validator,
)

from .file_paths import create_dir_name, create_file_name, run_id
from .utils import read_df, write_df


def _construct_setup_name(
    setup_name: str | None,
    name: str,
    seir_modifiers_scenario: str | None,
    outcome_modifiers_scenario: str | None,
) -> str:
    """
    Construct the setup name based on the provided parameters.

    If `setup_name` is provided, it is returned directly.
    Otherwise, the setup name is constructed from the model name and any modifiers
    scenarios.

    Args:
        setup_name: The explicit setup name, if provided.
        name: The model name.
        seir_modifiers_scenario: The SEIR modifiers scenario, if any.
        outcome_modifiers_scenario: The outcome modifiers scenario, if any.

    Returns:
        A string representing the setup name.

    Examples:
        >>> from gempyor.model_meta import _construct_setup_name
        >>> _construct_setup_name("setup-name", "model_name", None, None)
        'setup-name'
        >>> _construct_setup_name(None, "model_name", None, None)
        'model_name'
        >>> _construct_setup_name(None, "model_name", "seir-scenario", None)
        'model_name_seir-scenario'
        >>> _construct_setup_name(None, "model_name", None, "out-scenario")
        'model_name_out-scenario'
        >>> _construct_setup_name(None, "model_name", "seir-scenario", "out-scenario")
        'model_name_seir-scenario_out-scenario'
        >>> _construct_setup_name(
        ...     "setup-name", "model_name", "seir-scenario", "out-scenario"
        ... )
        'setup-name'

    """
    if setup_name is None:
        setup_name = name
        if seir_modifiers_scenario is not None:
            setup_name += f"_{seir_modifiers_scenario}"
        if outcome_modifiers_scenario is not None:
            setup_name += f"_{outcome_modifiers_scenario}"
    return setup_name


def _construct_prefix(data: dict[str, Any], kind: Literal["in", "out"]) -> str:
    """
    Construct the prefix for input or output based on the model metadata.

    This method is a thin wrapper around `_construct_setup_name` to create a prefix
    that includes the setup name and the run ID for the specified kind ("in" or "out")
    from the model metadata provided in `data` via pydantic's default factory.

    Args:
        data: The model metadata as a dictionary.
        kind: Either "in" or "out" to specify the type of prefix.

    Returns:
        A string representing the constructed prefix.
    """
    setup_name = _construct_setup_name(
        data.get("setup_name_"),
        data["name"],
        data.get("seir_modifiers_scenario"),
        data.get("outcome_modifiers_scenario"),
    )
    return f"{setup_name}/{data[f'{kind}_run_id']}/"


class ModelMeta(BaseModel):
    # pylint: disable=line-too-long
    """
    Metadata for a model.

    Attributes:
        name: The name of the model.
        nslots: The number of slots for the model.
        write_csv: Whether to write output as CSV files.
        write_parquet: Whether to write output as Parquet files.
        first_sim_index: The index of the first simulation.
        seir_modifiers_scenario: The SEIR modifiers scenario, if any.
        outcome_modifiers_scenario: The outcome modifiers scenario, if any.
        setup_name_: An explicit setup name, if provided.
        inference_filename_prefix: The prefix for inference filenames.
        inference_filepath_suffix: The suffix for inference file paths.
        timestamp: The timestamp for the model metadata, defaults to the current time.
        path_prefix: The prefix path for the model files, defaults to the current
            working directory.
        in_run_id: The run ID for input files, defaults to a generated run ID.
        out_run_id: The run ID for output files, defaults to the input run ID.
        in_prefix: The prefix for input files, constructed from the model metadata.
        out_prefix: The prefix for output files, constructed from the model metadata.

    Examples:
        >>> from datetime import datetime
        >>> from pathlib import Path
        >>> from gempyor.file_paths import run_id
        >>> from gempyor.model_meta import ModelMeta
        >>> meta = ModelMeta(
        ...     name="simple-model",
        ...     write_csv=True,
        ...     in_run_id=run_id(datetime(2025, 1, 2, 3, 4, 5)),
        ... )
        >>> meta.setup_name
        'simple-model'
        >>> meta.in_run_id
        '20250102_030405'
        >>> meta.out_run_id
        '20250102_030405'
        >>> meta.in_prefix
        'simple-model/20250102_030405/'
        >>> meta.out_prefix
        'simple-model/20250102_030405/'
        >>> meta.filename("spar", 42, "out").relative_to(Path.cwd())
        PosixPath('model_output/simple-model/20250102_030405/spar/000000042.20250102_030405.spar.csv')

    """
    # pylint: enable=line-too-long

    model_config = ConfigDict(extra="allow")

    name: str
    nslots: int = 1
    write_csv: bool = False
    write_parquet: bool = False
    first_sim_index: int = 1
    seir_modifiers_scenario: str | None = None
    outcome_modifiers_scenario: str | None = None
    setup_name_: str | None = Field(
        validation_alias=AliasChoices("setup_name_", "setup_name"), default=None
    )
    inference_filename_prefix: str = ""
    inference_filepath_suffix: str = ""
    timestamp: str = Field(default_factory=lambda: datetime.now().strftime("%Y%m%d-%H%M%S"))
    path_prefix: Path = Field(default_factory=Path.cwd)
    in_run_id: str = Field(default_factory=run_id, coerce_numbers_to_str=True)
    out_run_id: str = Field(
        default_factory=lambda data: data["in_run_id"], coerce_numbers_to_str=True
    )
    in_prefix: str = Field(
        default_factory=lambda data: _construct_prefix(data, "in"),
        coerce_numbers_to_str=True,
    )
    out_prefix: str = Field(
        default_factory=lambda data: _construct_prefix(data, "out"),
        coerce_numbers_to_str=True,
    )

    @model_validator(mode="after")
    def _validate_write_attributes(self) -> "ModelMeta":
        """
        Validate the write attributes of the model metadata.

        If both `write_csv` and `write_parquet` are set to `True`, a warning is issued
        indicating that only one format should be used for writing output files.

        Returns:
            The validated instance of `ModelMeta`.

        Raises:
            ValueError: If neither `write_csv` nor `write_parquet` is set to `True`.
        """
        if not (self.write_csv or self.write_parquet):
            self.write_parquet = True
        elif self.write_csv and self.write_parquet:
            warnings.warn(
                "Both `write_csv` and `write_parquet` are set to `True`. Only one "
                "format is used for writing output files, assuming `write_parquet`.",
                UserWarning,
            )
            self.write_csv = False
        return self

    @classmethod
    def from_confuse_config(
        cls,
        config: ConfigView,
        **kwargs: dict[
            Literal[
                "name",
                "nslots",
                "write_csv",
                "write_parquet",
                "first_sim_index",
                "seir_modifiers_scenario",
                "outcome_modifiers_scenario",
                "setup_name_",
                "inference_filename_prefix",
                "inference_filepath_suffix",
                "timestamp",
                "path_prefix",
                "in_run_id",
                "out_run_id",
                "in_prefix",
                "out_prefix",
            ],
            Any,
        ],
    ) -> "ModelMeta":
        """
        Create a `ModelMeta` instance from a confuse configuration view.

        Args:
            config: A configuration view containing the model metadata.
            **kwargs: Additional keyword arguments to pass to the model validation.

        Returns:
            A `ModelMeta` instance.
        """
        obj = {"name": config["name"].as_str()}
        for key in (
            "name",
            "nslots",
            "write_csv",
            "write_parquet",
            "first_sim_index",
            "seir_modifiers_scenario",
            "outcome_modifiers_scenario",
            "setup_name_",
            "inference_filename_prefix",
            "inference_filepath_suffix",
            "timestamp",
            "path_prefix",
            "in_run_id",
            "out_run_id",
            "in_prefix",
            "out_prefix",
        ):
            if (val := kwargs.get(key)) is not None:
                obj[key] = val
        return cls.model_validate(obj)

    @computed_field
    @property
    def extension(self) -> str:
        """
        Compute the file extension based on the write format.

        If `write_csv` is `True`, the extension is set to ".csv".
        If `write_parquet` is `True`, the extension is set to ".parquet".
        Otherwise, it defaults to ".txt".

        Returns:
            A string representing the file extension.
        """
        if self.write_csv:
            return "csv"
        if self.write_parquet:
            return "parquet"
        raise NotImplementedError(
            "Neither `write_csv` nor `write_parquet` is set to True. "
            "Please set one of them to True to determine the file extension."
        )

    @computed_field
    @property
    def setup_name(self) -> str:
        """
        Compute the setup name based on the model names and modifiers scenarios.

        If an explicit setup name is provided via `setup_name_`, it is used directly.
        If not, the setup name is constructed from the model name and any modifiers
        scenarios (if they are not `None`).

        Returns:
            A string representing the setup name.
        """
        return _construct_setup_name(
            setup_name=self.setup_name_,
            name=self.name,
            seir_modifiers_scenario=self.seir_modifiers_scenario,
            outcome_modifiers_scenario=self.outcome_modifiers_scenario,
        )

    def _run_id_and_prefix(self, kind: Literal["in", "out"]) -> tuple[str, str]:
        """
        Get the run ID and prefix for input or output based on the kind.

        Args:
            kind: Either "in" or "out" to specify the type of run ID and
                prefix to retrieve.

        Returns:
            A tuple containing the run ID and prefix for the specified kind.
        """
        return (
            (self.in_run_id, self.in_prefix)
            if kind == "in"
            else (self.out_run_id, self.out_prefix)
        )

    def filename(
        self,
        ftype: str,
        sim_id: int,
        kind: Literal["in", "out"],
        extension_override: str | None = None,
    ) -> Path:
        """
        Generate a filename for the model output.

        Args:
            ftype: The file type (e.g., "seir", "snpi", etc.).
            sim_id: The simulation ID.
            extension_override: An optional file extension to override the default.

        Returns:
            A string representing the filename for the model output.
        """
        run_id_, prefix = self._run_id_and_prefix(kind)
        return self.path_prefix / create_file_name(
            run_id_,
            prefix,
            sim_id + self.first_sim_index - 1,
            ftype,
            extension=self.extension if extension_override is None else extension_override,
            inference_filepath_suffix=self.inference_filepath_suffix,
            inference_filename_prefix=self.inference_filename_prefix,
        )

    def read_sim_id(
        self, ftype: str, sim_id: int, extension_override: str | None = None
    ) -> pd.DataFrame:
        """
        Read data for a specific simulation ID from the model output.

        Args:
            ftype: The file type (e.g., "seir", "snpi", etc.).
            sim_id: The simulation ID.
            extension_override: An optional file extension to override the default.

        Returns:
            The data read from the specified file.
        """
        return read_df(
            self.filename(ftype, sim_id, kind="in", extension_override=extension_override)
        )

    def write_sim_id(
        self,
        df: pd.DataFrame,
        ftype: str,
        sim_id: int,
        extension_override: str | None = None,
    ) -> None:
        """
        Write data for a specific simulation ID to the model output.

        Args:
            df: The data to write.
            ftype: The file type (e.g., "seir", "snpi", etc.).
            sim_id: The simulation ID.
            extension_override: An optional file extension to override the default.
        """
        filename = self.filename(
            ftype, sim_id, kind="out", extension_override=extension_override
        )
        filename.parent.mkdir(parents=True, exist_ok=True)
        write_df(filename, df)

    def create_model_output_directories(
        self, ftypes: list[str], kind: Literal["in", "out"]
    ) -> None:
        """
        Create the model output directories.

        Args:
            ftypes: A list of file types for which to create directories.
            kind: Either "in" or "out" to specify the type of directories to create.
        """
        run_id_, prefix = self._run_id_and_prefix(kind)
        for ftype in ftypes:
            Path(
                create_dir_name(
                    run_id_,
                    prefix,
                    ftype,
                    self.inference_filename_prefix,
                    self.inference_filepath_suffix,
                )
            ).mkdir(parents=True, exist_ok=True)
