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


def get_spatial_groups(grp_config, affected_geoids: list) -> dict:
    """
    Spatial groups are defined in the config file as a list (of lists).
    They have the same value.
    """
    # grouped is a list of lists of geoids, or just a list of geoids that belong in one group
    # ungrouped is a list of geoids
    spatial_groups = {"grouped": [], "ungrouped": []}

    if not grp_config["spatial_groups"].exists():
        spatial_groups["ungrouped"] = affected_geoids
    else:
        if grp_config["spatial_groups"].get() == "all":
            spatial_groups["grouped"] = [affected_geoids]
        else:
            spatial_groups["grouped"] = grp_config["spatial_groups"].get()

            spatial_groups["ungrouped"] = list(
                set(affected_geoids) - set(flatten_list_of_lists(spatial_groups["grouped"]))
            )

    # flatten the list of lists of grouped geoids, so we can do some checks
    flat_grouped_list = flatten_list_of_lists(spatial_groups["grouped"])
    print("Spatial groups:", spatial_groups)
    # check that all geoids are either grouped or ungrouped
    if set(flat_grouped_list + spatial_groups["ungrouped"]) != set(affected_geoids):
        raise ValueError(
            f"spatial_group error. for intervention with config \n {grp_config} \n The set of grouped and ungrouped geoids is not equal to the set of affected geoids"
        )
    if len(set(flat_grouped_list + spatial_groups["ungrouped"])) != len(
        flat_grouped_list + spatial_groups["ungrouped"]
    ):
        raise ValueError(
            f"spatial_group error. for intervention with config \n {grp_config} \n duplicate entries in the set of grouped and ungrouped geoids"
        )

    return spatial_groups


def flatten_list_of_lists(list_of_lists):
    """flatten a list of lists into a single list, or return the original list if it is not a list of lists"""
    if not list_of_lists:
        return list_of_lists  # empty list
    elif not isinstance(list_of_lists[0], list):
        return list_of_lists
    return [item for sublist in list_of_lists for item in sublist]
