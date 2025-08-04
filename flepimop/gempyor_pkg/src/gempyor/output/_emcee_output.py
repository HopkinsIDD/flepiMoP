__all__: tuple[str, ...] = ()

from pathlib import Path
from typing import Any, Literal

import confuse
import yaml
from emcee.backends import HDFBackend

from ..inference_parameter import InferenceParameters
from ..subpopulation_structure import SubpopulationStructure
from ._base import OutputABC
from ._types import Chains, ModifierInfo, ModifierInfoPeriod


def _parse_modifier_subpopulation_groups(
    modifier_config: dict[str, Any], subpopulation_names: list[str]
) -> list[list[str]]:
    # pylint: disable=line-too-long
    """
    Parse the subpopulation groups from a modifier configuration.

    Args:
        modifier_config: The modifier configuration dictionary.
        subpopulation_names: A reference list of subpopulation names.

    Returns:
        A list of subpopulation groups, where each group is a list of subpopulation
        names.

    Raises:
        ValueError: If the subpopulation groups cannot be parsed correctly.

    Examples:
        >>> from gempyor.output._emcee_output import _parse_modifier_subpopulation_groups
        >>> modifier_config = {
        ...     "subpop_groups": [["subpop1", "subpop2"], ["subpop3"]],
        ...     "subpop": "all",
        ... }
        >>> subpopulation_names = ["subpop1", "subpop2", "subpop3"]
        >>> _parse_modifier_subpopulation_groups(modifier_config, subpopulation_names)
        [['subpop1', 'subpop2'], ['subpop3']]
        >>> modifier_config = {
        ...     "subpop_groups": ["subpop1", "subpop2"],
        ...     "subpop": "all",
        ... }
        >>> _parse_modifier_subpopulation_groups(modifier_config, subpopulation_names)
        [['subpop1', 'subpop2']]
        >>> modifier_config = {
        ...     "subpop_groups": "subpop1",
        ...     "subpop": "all",
        ... }
        >>> _parse_modifier_subpopulation_groups(modifier_config, subpopulation_names)
        [['subpop1']]
        >>> modifier_config = {
        ...     "subpop_groups": None,
        ...     "subpop": "all",
        ... }
        >>> _parse_modifier_subpopulation_groups(modifier_config, subpopulation_names)
        [['subpop1'], ['subpop2'], ['subpop3']]
        >>> modifier_config = {
        ...     "subpop_groups": [[1, 2], ["subpop3"]],
        ...     "subpop": "all",
        ... }
        >>> _parse_modifier_subpopulation_groups(modifier_config, subpopulation_names)
        Traceback (most recent call last):
            ...
        ValueError: Unable to parse subpopulation groups from the modifier configuration. Expected a list of lists of strings. The relevant config keys are subpop_groups=[[1, 2], ['subpop3']] and subpop='all'.
    """
    # pylint: enable=line-too-long
    subpop_groups = modifier_config.get("subpop_groups")
    subpop = modifier_config.get("subpop", "all")
    if not subpop_groups:
        subpop_groups = [[n] for n in subpopulation_names] if subpop == "all" else [subpop]
    elif isinstance(subpop_groups, list) and all(isinstance(g, str) for g in subpop_groups):
        subpop_groups = [subpop_groups]
    elif isinstance(subpop_groups, str):
        subpop_groups = [[subpop_groups]]
    elif not (
        isinstance(subpop_groups, list)
        and all(isinstance(g, list) for g in subpop_groups)
        and all(isinstance(s, str) for g in subpop_groups for s in g)
    ):
        msg = (
            "Unable to parse subpopulation groups from the modifier "
            "configuration. Expected a list of lists of strings. The "
            f"relevant config keys are {subpop_groups=} and {subpop=}."
        )
        raise ValueError(msg)
    return subpop_groups


class EmceeOutput(OutputABC):
    def __init__(
        self,
        config: Path | str,
        run_id: str,
        seir_modifiers_scenario: str | None = None,
        outcome_modifers_scenario: str | None = None,
        path_prefix: Path | str | None = None,
    ) -> None:
        super().__init__(
            config, run_id, seir_modifiers_scenario, outcome_modifers_scenario, path_prefix
        )

        # Construct an instance of InferenceParameters so we can use
        # it to determine the order of the parameters in the H5 file
        with self._config.open("r") as f:
            conf = yaml.safe_load(f)
        cfg = confuse.RootView([confuse.ConfigSource.of(conf)])
        subpopulation_structure = SubpopulationStructure.from_confuse_config(
            cfg["subpop_setup"], path_prefix=path_prefix
        )
        self._inference_parameters = InferenceParameters(
            confuse.RootView([confuse.ConfigSource.of(conf)]),
            subpopulation_structure.subpop_names,
        )

        # Extract the names of the SEIR and outcome
        # modifiers from the inference parameters
        modifiers_names: dict[Literal["seir", "outcome"], set[str]] = {
            "seir": set(),
            "outcome": set(),
        }
        for i in range(len(self._inference_parameters)):
            kind = self._inference_parameters.ptypes[i][:-10]
            modifiers_names[kind].add(self._inference_parameters.pnames[i])

        # Parse the underlying modifiers parameters
        # and periods directly from the config
        modifiers_lib: dict[int, (str, list[ModifierInfoPeriod])] = {}
        for kind in ("seir", "outcome"):
            for modifier_name, modifier_conf in (
                conf.get(f"{kind}_modifiers", {}).get("modifiers", {}).items()
            ):
                parameter = modifier_conf["parameter"]
                if (method := modifier_conf["method"]) == "SinglePeriodModifier":
                    subpop_groups = _parse_modifier_subpopulation_groups(
                        modifier_conf, subpopulation_structure.subpop_names
                    )
                    for subpop_group in subpop_groups:
                        subpop_group = tuple(sorted(subpop_group))
                        lookup_hash = hash((kind, modifier_name, subpop_group))
                        modifiers_lib[lookup_hash] = (
                            parameter,
                            [
                                ModifierInfoPeriod(
                                    start_date=modifier_conf["period_start_date"],
                                    end_date=modifier_conf["period_end_date"],
                                )
                            ],
                        )
                elif method == "MultiPeriodModifier":
                    for group_conf in modifier_conf.get("groups", []):
                        subpop_groups = _parse_modifier_subpopulation_groups(
                            group_conf, subpopulation_structure.subpop_names
                        )
                        periods = [
                            ModifierInfoPeriod(
                                start_date=p["start_date"], end_date=p["end_date"]
                            )
                            for p in group_conf.get("periods", [])
                        ]
                        for subpop_group in subpop_groups:
                            subpop_group = tuple(sorted(subpop_group))
                            lookup_hash = hash((kind, modifier_name, subpop_group))
                            modifiers_lib[lookup_hash] = (
                                parameter,
                                periods,
                            )
                else:
                    msg = (
                        f"Unsupported modifier method '{method}' for "
                        f"the {kind} modifier '{modifier_name}'."
                    )
                    raise NotImplementedError(msg)

        # Create a list of ModifierInfo objects from the
        # inference parameters and parsed modifiers config
        self._modifiers: list[ModifierInfo] = []
        for i in range(len(self._inference_parameters)):
            kind = self._inference_parameters.ptypes[i][:-10]
            modifier_name = self._inference_parameters.pnames[i]
            subpops = sorted(self._inference_parameters.subpops[i].split(","))
            parameter, periods = modifiers_lib[hash((kind, modifier_name, tuple(subpops)))]
            self._modifiers.append(
                ModifierInfo(
                    kind=kind,
                    name=modifier_name,
                    subpops=subpops,
                    periods=periods,
                    parameter=parameter,
                )
            )

        # Open the HDF5 backend file produced by EMCEE which contains the chains
        h5 = self._path_prefix / f"{self._run_id}_backend.h5"
        if not h5.exists():
            msg = f"The EMCEE inference H5 backend file '{h5}' does not exist."
            raise FileNotFoundError(msg)
        if not h5.is_file():
            msg = f"The EMCEE inference H5 backend file '{h5}' is not a file."
            raise NotADirectoryError(msg)
        self._reader = HDFBackend(h5, read_only=True)

    def get_chains(self) -> Chains:
        log_prob = self._reader.get_log_prob()
        samples = self._reader.get_chain()
        shape = samples.shape
        return Chains(
            shape=shape,
            log_probability=log_prob,
            samples=samples,
            modifiers=self._modifiers,
        )
