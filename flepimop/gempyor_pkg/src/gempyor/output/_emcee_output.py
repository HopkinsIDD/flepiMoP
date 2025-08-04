__all__: tuple[str, ...] = ()

from pathlib import Path
from typing import Literal

import confuse
import numpy as np
import yaml
from emcee.backends import HDFBackend

from ..inference_parameter import InferenceParameters
from ..NPI.helpers import SpatialGroups
from ..subpopulation_structure import SubpopulationStructure
from ._base import OutputABC
from ._types import Chains, ModifierInfo, ModifierInfoPeriod


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
                parameter = modifier_conf.get("parameter")
                if (method := modifier_conf["method"]) == "SinglePeriodModifier":
                    spatial_groups = SpatialGroups.from_subpopulations(
                        (
                            subpopulation_structure.subpop_names
                            if ((subpop := modifier_conf.get("subpop", "all") == "all"))
                            else subpop
                        ),
                        modifier_conf.get("subpop_groups", None),
                    )
                    for _, subpop_group in spatial_groups:
                        subpop_group = tuple(subpop_group)
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
                        spatial_groups = SpatialGroups.from_subpopulations(
                            (
                                subpopulation_structure.subpop_names
                                if ((subpop := modifier_conf.get("subpop", "all") == "all"))
                                else subpop
                            ),
                            group_conf.get("subpop_groups", None),
                        )
                        periods = [
                            ModifierInfoPeriod(
                                start_date=p["start_date"], end_date=p["end_date"]
                            )
                            for p in group_conf.get("periods", [])
                        ]
                        for _, subpop_group in spatial_groups:
                            subpop_group = tuple(sorted(subpop_group))
                            lookup_hash = hash((kind, modifier_name, subpop_group))
                            modifiers_lib[lookup_hash] = (
                                parameter,
                                periods,
                            )
                elif method == "StackedModifier":
                    # Inference is not done directly on StackedModifiers, but instead
                    # on the underlying constituents which are captured above.
                    continue
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
            lookup_hash = hash((kind, modifier_name, tuple(subpops)))
            parameter, periods = modifiers_lib[lookup_hash]
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
        log_prob = self._reader.get_log_prob().T
        samples = np.transpose(self._reader.get_chain(), axes=(1, 0, 2))
        shape = samples.shape
        return Chains(
            shape=shape,
            log_probability=log_prob,
            samples=samples,
            modifiers=self._modifiers,
        )
