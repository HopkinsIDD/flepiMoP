import pandas as pd
import numpy as np
import typing


# Helper function
def reduce_parameter(
    parameter: np.ndarray,
    modification: typing.Union[pd.DataFrame, float],
    method: str = "prod",
) -> np.ndarray:
    if isinstance(modification, pd.DataFrame):
        modification = modification.T
        modification.index = pd.to_datetime(modification.index.astype(str))
        modification = modification.resample("1D").ffill().to_numpy()  # Type consistency:
    if method == "prod":
        return parameter * (1 - modification)
    elif method == "sum":
        return parameter + modification
    else:
        raise ValueError(f"Unknown method to do NPI reduction, got {method}")


def get_spatial_groups(grp_config, subpop: list) -> dict:
    """
    Spatial groups are defined in the config file as a list (of lists).
    They have the same value.
    grouped is a list of lists of subpop
    ungrouped is a list of subpop
    the list are ordered, and this is important so we can get back and forth
    from the written to disk part that is comma separated
    """

    spatial_groups = {"grouped": [], "ungrouped": []}

    if not grp_config["spatial_groups"].exists():
        spatial_groups["ungrouped"] = subpop
    else:
        if grp_config["spatial_groups"].get() == "all":
            spatial_groups["grouped"] = [subpop]
        else:
            spatial_groups["grouped"] = grp_config["spatial_groups"].get()
            spatial_groups["ungrouped"] = list(
                set(subpop) - set(flatten_list_of_lists(spatial_groups["grouped"]))
            )

    # flatten the list of lists of grouped subpop, so we can do some checks
    flat_grouped_list = flatten_list_of_lists(spatial_groups["grouped"])
    # check that all subpop are either grouped or ungrouped
    if set(flat_grouped_list + spatial_groups["ungrouped"]) != set(subpop):
        print("set of grouped and ungrouped subpop", set(flat_grouped_list + spatial_groups["ungrouped"]))
        print("set of affected subpop             ", set(subpop))
        raise ValueError(f"The two above sets are differs for for intervention with config \n {grp_config}")
    if len(set(flat_grouped_list + spatial_groups["ungrouped"])) != len(
        flat_grouped_list + spatial_groups["ungrouped"]
    ):
        raise ValueError(
            f"spatial_group error. for intervention with config \n {grp_config} \n duplicate entries in the set of grouped and ungrouped subpop"
        )

    spatial_groups["grouped"] = make_list_of_list(spatial_groups["grouped"])

    # sort the lists
    spatial_groups["grouped"] = [sorted(x) for x in spatial_groups["grouped"]]
    spatial_groups["ungrouped"] = sorted(spatial_groups["ungrouped"])

    return spatial_groups


def flatten_list_of_lists(list_of_lists):
    """flatten a list of lists into a single list, or return the original list if it is not a list of lists"""
    if not list_of_lists:
        return list_of_lists  # empty list
    elif not isinstance(list_of_lists[0], list):
        return list_of_lists
    return [item for sublist in list_of_lists for item in sublist]


def make_list_of_list(this_list):
    """if the list contains its' values, nest it into another list"""
    if not this_list:
        return this_list  # empty list
    elif isinstance(this_list[0], list):
        return this_list
    else:
        return [this_list]
