"""Unit tests for the `gempyor.inference_parameter.InferenceParameter` class."""

from math import inf
from typing import Any, Literal, NamedTuple, TypedDict

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from gempyor.distributions import (
    BetaDistribution,
    DistributionABC,
    GammaDistribution,
    UniformDistribution,
)
from gempyor.inference_parameter import InferenceParameters
from gempyor.testing import create_confuse_configview_from_dict


class AddSingleParameterArg(TypedDict):
    """Arguments for adding a single parameter."""

    ptype: Literal["outcome_modifiers", "seir_modifiers"]
    pname: str
    subpop: str
    pdist: DistributionABC


class AddModifierArg(NamedTuple):
    """Arguments for adding a modifier parameter."""

    pname: str
    ptype: Literal["outcome_modifiers", "seir_modifiers"]
    parameter_config: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """Convert to a dictionary."""
        return {
            "pname": self.pname,
            "ptype": self.ptype,
            "parameter_config": create_confuse_configview_from_dict(self.parameter_config),
        }

    def number_of_subpopulations(self, subpopulations: list[str]) -> int:
        """Return the number of subpopulations."""
        if (
            self.parameter_config.get("method", "SinglePeriodModifier")
            == "MultiPeriodModifier"
        ):
            return sum(
                AddModifierArg(
                    pname=self.pname, ptype=self.ptype, parameter_config=group
                ).number_of_subpopulations(subpopulations)
                for group in self.parameter_config.get("groups", [])
            )
        if subpop_groups := self.parameter_config.get("subpop_groups"):
            return len(subpop_groups)
        if self.parameter_config["subpop"] == "all":
            return len(subpopulations)
        return len(self.parameter_config["subpop"])


@pytest.mark.parametrize(
    "single_parameter_args",
    [
        [
            AddSingleParameterArg(
                ptype="incidH::delay",
                pname="Ro",
                subpop="GA",
                pdist=GammaDistribution(shape=14.0, scale=0.5),
            ),
        ],
        [
            AddSingleParameterArg(
                ptype="seir_modifiers",
                pname="gamma",
                subpop="GA",
                pdist=GammaDistribution(shape=2.0, scale=0.5),
            ),
            AddSingleParameterArg(
                ptype="seir_modifiers",
                pname="beta",
                subpop="GA",
                pdist=GammaDistribution(shape=2.0, scale=0.5),
            ),
        ],
        [
            AddSingleParameterArg(
                ptype="outcome_modifiers",
                pname="incidH::probability",
                subpop="USA",
                pdist=BetaDistribution(alpha=2.0, beta=5.0),
            ),
            AddSingleParameterArg(
                ptype="seir_modifiers",
                pname="Ro",
                subpop="USA",
                pdist=GammaDistribution(shape=1.5, scale=0.5),
            ),
            AddSingleParameterArg(
                ptype="outcome_modifiers",
                pname="incidH::delay",
                subpop="Canada",
                pdist=GammaDistribution(shape=10.0, scale=0.5),
            ),
            AddSingleParameterArg(
                ptype="seir_modifiers",
                pname="beta",
                subpop="Canada",
                pdist=GammaDistribution(shape=2.0, scale=0.5),
            ),
        ],
    ],
)
def test_adding_single_parameters(
    single_parameter_args: list[AddSingleParameterArg],
) -> None:
    """Test adding single parameters to an empty `InferenceParameters` class."""
    inference_params = InferenceParameters(create_confuse_configview_from_dict({}), [])
    for arg in single_parameter_args:
        inference_params.add_single_parameter(**arg)
    assert inference_params.get_dim() == len(single_parameter_args)
    assert inference_params.ptypes == [arg["ptype"] for arg in single_parameter_args]
    assert inference_params.pnames == [arg["pname"] for arg in single_parameter_args]
    assert inference_params.subpops == [arg["subpop"] for arg in single_parameter_args]
    assert inference_params.pdists == [arg["pdist"] for arg in single_parameter_args]


@pytest.mark.parametrize(
    ("modifier_args", "subpopulations"),
    [
        (
            [
                AddModifierArg(
                    pname="Ro_summer",
                    ptype="seir_modifiers",
                    parameter_config={
                        "method": "SinglePeriodModifier",
                        "parameter": "Ro",
                        "subpop": "all",
                        "value": {
                            "distribution": "gamma",
                            "shape": 1.5,
                            "scale": 0.5,
                        },
                    },
                ),
            ],
            ["USA"],
        ),
        (
            [
                AddModifierArg(
                    pname="gamma_humid",
                    ptype="seir_modifiers",
                    parameter_config={
                        "method": "SinglePeriodModifier",
                        "parameter": "gamma",
                        "subpop": "all",
                        "value": {
                            "distribution": "gamma",
                            "shape": 1.5,
                            "scale": 0.5,
                        },
                    },
                ),
            ],
            ["NC", "SC", "GA"],
        ),
        (
            [
                AddModifierArg(
                    pname="incidH_spring_fall",
                    ptype="outcome_modifiers",
                    parameter_config={
                        "method": "MultiPeriodModifier",
                        "parameter": "incidH::probability",
                        "groups": [
                            {
                                "subpop": "all",
                            },
                        ],
                        "value": {
                            "distribution": "beta",
                            "alpha": 2.0,
                            "beta": 5.0,
                        },
                    },
                ),
                AddModifierArg(
                    pname="incidH_winter_delay",
                    ptype="outcome_modifiers",
                    parameter_config={
                        "method": "MultiPeriodModifier",
                        "parameter": "incidH::delay",
                        "groups": [
                            {
                                "subpop": "all",
                                "subpop_groups": [["USA", "Canada"]],
                            }
                        ],
                        "value": {
                            "distribution": "gamma",
                            "shape": 1.5,
                            "scale": 0.5,
                        },
                    },
                ),
            ],
            ["USA", "Canada"],
        ),
        (
            [
                AddModifierArg(
                    pname="Ro_summer_midatlantic",
                    ptype="seir_modifiers",
                    parameter_config={
                        "method": "SinglePeriodModifier",
                        "parameter": "Ro",
                        "subpop": ["VA", "NC"],
                        "subpop_groups": [["VA", "NC"]],
                        "value": {
                            "distribution": "truncnorm",
                            "a": 0.1,
                            "b": 2.1,
                            "mean": 0.9,
                            "sd": 0.5,
                        },
                    },
                ),
                AddModifierArg(
                    pname="Ro_summer_southeast",
                    ptype="seir_modifiers",
                    parameter_config={
                        "method": "SinglePeriodModifier",
                        "parameter": "Ro",
                        "subpop": ["SC", "GA"],
                        "subpop_groups": [["SC", "GA"]],
                        "value": {
                            "distribution": "truncnorm",
                            "a": 0.1,
                            "b": 2.1,
                            "mean": 0.9,
                            "sd": 0.5,
                        },
                    },
                ),
            ],
            ["VA", "NC", "SC", "GA"],
        ),
    ],
)
def test_adding_modifiers(
    modifier_args: list[AddModifierArg], subpopulations: list[str]
) -> None:
    """Test adding modifier parameters to an `InferenceParameters` class."""
    inference_params = InferenceParameters(create_confuse_configview_from_dict({}), [])
    for arg in modifier_args:
        inference_params.add_modifier(**arg.as_dict() | {"subpops": subpopulations})

    assert inference_params.get_dim() == sum(
        arg.number_of_subpopulations(subpopulations) for arg in modifier_args
    )


@pytest.mark.parametrize(
    ("single_parameter_args", "proposal"),
    [
        (
            [
                AddSingleParameterArg(
                    ptype="incidH::delay",
                    pname="Ro",
                    subpop="GA",
                    pdist=GammaDistribution(shape=14.0, scale=0.5),
                ),
            ],
            np.array([7.0], dtype=np.float64),
        ),
        (
            [
                AddSingleParameterArg(
                    ptype="incidH::delay",
                    pname="Ro",
                    subpop="GA",
                    pdist=GammaDistribution(shape=14.0, scale=0.5),
                ),
            ],
            np.array([0.0], dtype=np.float64),
        ),
        (
            [
                AddSingleParameterArg(
                    ptype="incidH::delay",
                    pname="Ro",
                    subpop="GA",
                    pdist=GammaDistribution(shape=14.0, scale=0.5),
                ),
            ],
            np.array([-0.5], dtype=np.float64),
        ),
        (
            [
                AddSingleParameterArg(
                    ptype="seir_modifiers",
                    pname="gamma",
                    subpop="GA",
                    pdist=GammaDistribution(shape=2.0, scale=0.5),
                ),
                AddSingleParameterArg(
                    ptype="seir_modifiers",
                    pname="beta",
                    subpop="GA",
                    pdist=UniformDistribution(low=0.0, high=2.0),
                ),
            ],
            np.array([1.0, 1.0], dtype=np.float64),
        ),
        (
            [
                AddSingleParameterArg(
                    ptype="seir_modifiers",
                    pname="gamma",
                    subpop="GA",
                    pdist=GammaDistribution(shape=2.0, scale=0.5),
                ),
                AddSingleParameterArg(
                    ptype="seir_modifiers",
                    pname="beta",
                    subpop="GA",
                    pdist=UniformDistribution(low=0.0, high=2.0),
                ),
            ],
            np.array([-1.0, 1.0], dtype=np.float64),
        ),
        (
            [
                AddSingleParameterArg(
                    ptype="seir_modifiers",
                    pname="gamma",
                    subpop="GA",
                    pdist=GammaDistribution(shape=2.0, scale=0.5),
                ),
                AddSingleParameterArg(
                    ptype="seir_modifiers",
                    pname="beta",
                    subpop="GA",
                    pdist=UniformDistribution(low=0.0, high=2.0),
                ),
            ],
            np.array([1.0, -1.0], dtype=np.float64),
        ),
        (
            [
                AddSingleParameterArg(
                    ptype="seir_modifiers",
                    pname="gamma",
                    subpop="GA",
                    pdist=GammaDistribution(shape=2.0, scale=0.5),
                ),
                AddSingleParameterArg(
                    ptype="seir_modifiers",
                    pname="beta",
                    subpop="GA",
                    pdist=UniformDistribution(low=0.0, high=2.0),
                ),
            ],
            np.array([-1.0, -1.0], dtype=np.float64),
        ),
        (
            [
                AddSingleParameterArg(
                    ptype="seir_modifiers",
                    pname="gamma",
                    subpop="GA",
                    pdist=GammaDistribution(shape=2.0, scale=0.5),
                ),
                AddSingleParameterArg(
                    ptype="seir_modifiers",
                    pname="beta",
                    subpop="GA",
                    pdist=UniformDistribution(low=0.0, high=2.0),
                ),
            ],
            np.array([0.0, 0.0], dtype=np.float64),
        ),
        (
            [
                AddSingleParameterArg(
                    ptype="seir_modifiers",
                    pname="gamma",
                    subpop="GA",
                    pdist=GammaDistribution(shape=2.0, scale=0.5),
                ),
                AddSingleParameterArg(
                    ptype="seir_modifiers",
                    pname="beta",
                    subpop="GA",
                    pdist=UniformDistribution(low=0.0, high=2.0),
                ),
            ],
            np.array([0.0, 2.0], dtype=np.float64),
        ),
        (
            [
                AddSingleParameterArg(
                    ptype="seir_modifiers",
                    pname="gamma",
                    subpop="GA",
                    pdist=GammaDistribution(shape=2.0, scale=0.5),
                ),
                AddSingleParameterArg(
                    ptype="seir_modifiers",
                    pname="beta",
                    subpop="GA",
                    pdist=UniformDistribution(low=0.0, high=2.0),
                ),
            ],
            np.array([0.0, 3.0], dtype=np.float64),
        ),
    ],
)
def test_check_in_bound(
    single_parameter_args: list[AddSingleParameterArg], proposal: npt.NDArray[np.float64]
) -> None:
    """Test checking if a proposal is within bounds for single parameters."""
    inference_params = InferenceParameters(create_confuse_configview_from_dict({}), [])
    for arg in single_parameter_args:
        inference_params.add_single_parameter(**arg)
    assert proposal.ndim == 1
    assert len(inference_params) == len(proposal)
    lower_bounds = [arg["pdist"].support[0] for arg in single_parameter_args]
    upper_bounds = [arg["pdist"].support[1] for arg in single_parameter_args]
    assert inference_params.check_in_bound(proposal) == np.all(
        (proposal >= lower_bounds) & (proposal <= upper_bounds)
    )


@pytest.mark.parametrize(
    (
        "modifier_args",
        "subpopulations",
        "proposal",
        "snpi_df",
        "hnpi_df",
        "expected_snpi_df",
        "expected_hnpi_df",
    ),
    [
        (
            [
                AddModifierArg(
                    pname="gamma_humid",
                    ptype="seir_modifiers",
                    parameter_config={
                        "method": "SinglePeriodModifier",
                        "parameter": "gamma",
                        "subpop": "all",
                        "value": {
                            "distribution": "gamma",
                            "shape": 1.5,
                            "scale": 0.5,
                        },
                    },
                ),
            ],
            ["GA", "NC", "SC"],
            np.arange(1, 4, dtype=np.float64),
            pd.DataFrame.from_records(
                [
                    {
                        "subpop": "GA",
                        "modifier_name": "gamma_humid",
                        "start_date": "2023-07-01",
                        "end_date": "2023-09-30",
                        "parameter": "gamma",
                        "value": 1.5,
                    },
                    {
                        "subpop": "NC",
                        "modifier_name": "gamma_humid",
                        "start_date": "2023-07-01",
                        "end_date": "2023-09-30",
                        "parameter": "gamma",
                        "value": 1.5,
                    },
                    {
                        "subpop": "SC",
                        "modifier_name": "gamma_humid",
                        "start_date": "2023-07-01",
                        "end_date": "2023-09-30",
                        "parameter": "gamma",
                        "value": 1.5,
                    },
                ]
            ),
            pd.DataFrame(
                columns=[
                    "subpop",
                    "modifier_name",
                    "start_date",
                    "end_date",
                    "parameter",
                    "value",
                ]
            ),
            pd.DataFrame.from_records(
                [
                    {
                        "subpop": "GA",
                        "modifier_name": "gamma_humid",
                        "start_date": "2023-07-01",
                        "end_date": "2023-09-30",
                        "parameter": "gamma",
                        "value": 1.0,
                    },
                    {
                        "subpop": "NC",
                        "modifier_name": "gamma_humid",
                        "start_date": "2023-07-01",
                        "end_date": "2023-09-30",
                        "parameter": "gamma",
                        "value": 2.0,
                    },
                    {
                        "subpop": "SC",
                        "modifier_name": "gamma_humid",
                        "start_date": "2023-07-01",
                        "end_date": "2023-09-30",
                        "parameter": "gamma",
                        "value": 3.0,
                    },
                ]
            ),
            pd.DataFrame(
                columns=[
                    "subpop",
                    "modifier_name",
                    "start_date",
                    "end_date",
                    "parameter",
                    "value",
                ]
            ),
        ),
        (
            [
                AddModifierArg(
                    pname="incidH_spring_delay",
                    ptype="outcome_modifiers",
                    parameter_config={
                        "method": "SinglePeriodModifier",
                        "parameter": "incidH::delay",
                        "subpop": "all",
                        "value": {
                            "distribution": "gamma",
                            "shape": 14,
                            "scale": 0.5,
                        },
                    },
                ),
            ],
            ["USA", "Canada"],
            np.array([7.0, 10.0], dtype=np.float64),
            pd.DataFrame(
                columns=[
                    "subpop",
                    "modifier_name",
                    "start_date",
                    "end_date",
                    "parameter",
                    "value",
                ]
            ),
            pd.DataFrame.from_records(
                [
                    {
                        "subpop": "Canada",
                        "modifier_name": "incidH_spring_delay",
                        "start_date": "2023-03-01",
                        "end_date": "2023-05-31",
                        "parameter": "incidH::delay",
                        "value": 14.0,
                    },
                    {
                        "subpop": "USA",
                        "modifier_name": "incidH_spring_delay",
                        "start_date": "2023-03-01",
                        "end_date": "2023-05-31",
                        "parameter": "incidH::delay",
                        "value": 14.0,
                    },
                ]
            ),
            pd.DataFrame(
                columns=[
                    "subpop",
                    "modifier_name",
                    "start_date",
                    "end_date",
                    "parameter",
                    "value",
                ]
            ),
            pd.DataFrame.from_records(
                [
                    {
                        "subpop": "Canada",
                        "modifier_name": "incidH_spring_delay",
                        "start_date": "2023-03-01",
                        "end_date": "2023-05-31",
                        "parameter": "incidH::delay",
                        "value": 7.0,
                    },
                    {
                        "subpop": "USA",
                        "modifier_name": "incidH_spring_delay",
                        "start_date": "2023-03-01",
                        "end_date": "2023-05-31",
                        "parameter": "incidH::delay",
                        "value": 10.0,
                    },
                ]
            ),
        ),
        (
            [
                AddModifierArg(
                    pname="beta_summer_regional",
                    ptype="seir_modifiers",
                    parameter_config={
                        "method": "SinglePeriodModifier",
                        "parameter": "beta",
                        "subpop": "all",
                        "subpop_groups": [["VA", "NC"], ["SC", "GA"]],
                        "value": {
                            "distribution": "truncnorm",
                            "a": 0.1,
                            "b": 2.1,
                            "mean": 0.9,
                            "sd": 0.5,
                        },
                    },
                ),
                AddModifierArg(
                    pname="incidH_fall_prob",
                    ptype="outcome_modifiers",
                    parameter_config={
                        "method": "SinglePeriodModifier",
                        "parameter": "incidH::probability",
                        "subpop": "all",
                        "value": {
                            "distribution": "beta",
                            "alpha": 2.0,
                            "beta": 5.0,
                        },
                    },
                ),
            ],
            ["VA", "NC", "SC", "GA"],
            np.array([1.0, 2.0, 0.8, 0.4, 0.6, 0.2], dtype=np.float64),
            pd.DataFrame.from_records(
                [
                    {
                        "subpop": "NC,VA",
                        "modifier_name": "beta_summer_regional",
                        "start_date": "2023-07-01",
                        "end_date": "2023-09-30",
                        "parameter": "beta",
                        "value": 1.5,
                    },
                    {
                        "subpop": "GA,SC",
                        "modifier_name": "beta_summer_regional",
                        "start_date": "2023-07-01",
                        "end_date": "2023-09-30",
                        "parameter": "beta",
                        "value": 1.5,
                    },
                    {
                        "subpop": "GA,NC,SC",
                        "modifier_name": "beta_humid",
                        "start_date": "2023-07-01",
                        "end_date": "2023-09-30",
                        "parameter": "beta",
                        "value": 1.5,
                    },
                ]
            ),
            pd.DataFrame.from_records(
                [
                    {
                        "subpop": "VA",
                        "modifier_name": "incidH_fall_prob",
                        "start_date": "2023-07-01",
                        "end_date": "2023-09-30",
                        "parameter": "incidH::probability",
                        "value": 0.05,
                    },
                    {
                        "subpop": "NC",
                        "modifier_name": "incidH_fall_prob",
                        "start_date": "2023-07-01",
                        "end_date": "2023-09-30",
                        "parameter": "incidH::probability",
                        "value": 0.05,
                    },
                    {
                        "subpop": "SC",
                        "modifier_name": "incidH_fall_prob",
                        "start_date": "2023-07-01",
                        "end_date": "2023-09-30",
                        "parameter": "incidH::probability",
                        "value": 0.05,
                    },
                    {
                        "subpop": "GA",
                        "modifier_name": "incidH_fall_prob",
                        "start_date": "2023-07-01",
                        "end_date": "2023-09-30",
                        "parameter": "incidH::probability",
                        "value": 0.05,
                    },
                ]
            ),
            pd.DataFrame.from_records(
                [
                    {
                        "subpop": "NC,VA",
                        "modifier_name": "beta_summer_regional",
                        "start_date": "2023-07-01",
                        "end_date": "2023-09-30",
                        "parameter": "beta",
                        "value": 1.0,
                    },
                    {
                        "subpop": "GA,SC",
                        "modifier_name": "beta_summer_regional",
                        "start_date": "2023-07-01",
                        "end_date": "2023-09-30",
                        "parameter": "beta",
                        "value": 2.0,
                    },
                    {
                        "subpop": "GA,NC,SC",
                        "modifier_name": "beta_humid",
                        "start_date": "2023-07-01",
                        "end_date": "2023-09-30",
                        "parameter": "beta",
                        "value": 1.5,
                    },
                ]
            ),
            pd.DataFrame.from_records(
                [
                    {
                        "subpop": "VA",
                        "modifier_name": "incidH_fall_prob",
                        "start_date": "2023-07-01",
                        "end_date": "2023-09-30",
                        "parameter": "incidH::probability",
                        "value": 0.2,
                    },
                    {
                        "subpop": "NC",
                        "modifier_name": "incidH_fall_prob",
                        "start_date": "2023-07-01",
                        "end_date": "2023-09-30",
                        "parameter": "incidH::probability",
                        "value": 0.4,
                    },
                    {
                        "subpop": "SC",
                        "modifier_name": "incidH_fall_prob",
                        "start_date": "2023-07-01",
                        "end_date": "2023-09-30",
                        "parameter": "incidH::probability",
                        "value": 0.6,
                    },
                    {
                        "subpop": "GA",
                        "modifier_name": "incidH_fall_prob",
                        "start_date": "2023-07-01",
                        "end_date": "2023-09-30",
                        "parameter": "incidH::probability",
                        "value": 0.8,
                    },
                ]
            ),
        ),
    ],
)
def test_inject_proposal(
    modifier_args: list[AddModifierArg],
    subpopulations: list[str],
    proposal: npt.NDArray[np.float64],
    snpi_df: pd.DataFrame,
    hnpi_df: pd.DataFrame,
    expected_snpi_df: pd.DataFrame,
    expected_hnpi_df: pd.DataFrame,
) -> None:
    """Test injecting a proposal into the modifiers with `InferenceParameters`."""
    # if subpopulations == ["USA", "Canada"]:
    #     import pdbpp; pdbpp.set_trace()

    inference_params = InferenceParameters(create_confuse_configview_from_dict({}), [])
    for arg in modifier_args:
        inference_params.add_modifier(**arg.as_dict() | {"subpops": subpopulations})
    modified_snpi_df, modified_hnpi_df = inference_params.inject_proposal(
        proposal, snpi_df, hnpi_df
    )
    pd.testing.assert_frame_equal(
        modified_snpi_df.drop(columns=["value"]), snpi_df.drop(columns=["value"])
    )
    pd.testing.assert_frame_equal(
        modified_hnpi_df.drop(columns=["value"]), hnpi_df.drop(columns=["value"])
    )
    pd.testing.assert_frame_equal(modified_snpi_df, expected_snpi_df, check_like=True)
    pd.testing.assert_frame_equal(modified_hnpi_df, expected_hnpi_df, check_like=True)
