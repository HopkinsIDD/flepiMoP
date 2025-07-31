from pathlib import Path
import pandas as pd
import numpy as np


def build_initial_array(
    ic_file: Path,
    *,
    compartment_labels: list[str],
    n_nodes: int,
    node_col: str = "group",
    comp_col: str = "compartment",
    count_col: str = "N",
    allow_missing_subpops: bool = True,
    allow_missing_compartments: bool = True,
    proportional_ic: bool = False,
    allowed_node_labels: list[str] | None = None,
) -> np.ndarray:
    """
    Reads a tidy-format initial condition CSV file and constructs the initial
    state array y0 with shape (n_compartments, n_nodes).

    Args:
        ic_file: Path to the initial conditions .csv file.
        compartment_labels: Ordered list of compartment names expected in the simulation.
        n_nodes: Number of subpopulations (columns of y0).
        node_col: Column name in the CSV for the subpopulation identifier (default: 'group').
        comp_col: Column name for the compartment label (default: 'compartment').
        count_col: Column name for the count in each compartment (default: 'N').
        allow_missing_subpops: If True, ignore rows with unknown subpop indices.
        allow_missing_compartments: If True, ignore rows with unrecognized compartments.
        proportional_ic: Not yet implemented; if True, will attempt to proportionally fill.

    Returns:
        y0: NumPy array of shape (n_compartments, n_nodes)
    """
    ic_df = pd.read_csv(ic_file)
    return read_initial_condition_from_tidydataframe(
        ic_df=ic_df,
        compartment_labels=compartment_labels,
        n_nodes=n_nodes,
        node_col=node_col,
        comp_col=comp_col,
        count_col=count_col,
        allow_missing_subpops=allow_missing_subpops,
        allow_missing_compartments=allow_missing_compartments,
        proportional_ic=proportional_ic,
        allowed_node_labels=allowed_node_labels,
    )


def read_initial_condition_from_tidydataframe(
    ic_df: pd.DataFrame,
    *,
    compartment_labels: list[str],
    n_nodes: int,
    node_col: str,
    comp_col: str,
    count_col: str,
    allow_missing_subpops: bool = True,
    allow_missing_compartments: bool = True,
    proportional_ic: bool = False,
    allowed_node_labels: list[str] | None = None,
) -> np.ndarray:
    """
    Converts a tidy-format DataFrame into an initial condition array.

    Args:
        ic_df: A pandas DataFrame with at least 3 columns — for node, compartment, and count.
        compartment_labels: List of allowed/expected compartment names.
        n_nodes: Total number of subpopulations.
        node_col: Column in ic_df identifying the subpopulation (e.g., 'subpop').
        comp_col: Column identifying the compartment (e.g., 'mc_name').
        count_col: Column with the number of individuals in that compartment (e.g., 'amount').
        allow_missing_subpops: If True, rows with unknown nodes are skipped silently.
        allow_missing_compartments: If True, rows with unrecognized compartments are skipped.
        proportional_ic: Placeholder for future support; raises NotImplementedError for now.
        allowed_node_labels: Optional list of valid node labels; if provided, only these are accepted.

    Returns:
        y0: A (n_compartments × n_nodes) array populated from the DataFrame.
    """
    required_cols = {node_col, comp_col, count_col}
    actual_cols = set(ic_df.columns)
    if not required_cols.issubset(actual_cols):
        raise ValueError(
            f"Missing required columns in initial condition file: {required_cols - actual_cols}. "
            f"Found columns: {list(ic_df.columns)}"
        )

    comp_idx_map = {name: i for i, name in enumerate(compartment_labels)}

    if allowed_node_labels is not None:
        node_idx_map = {label: i for i, label in enumerate(allowed_node_labels)}
    else:
        unique_nodes = sorted(ic_df[node_col].unique())
        node_idx_map = {label: i for i, label in enumerate(unique_nodes)}

    if len(node_idx_map) > n_nodes:
        raise ValueError(
            f"More unique nodes in data ({len(node_idx_map)}) than n_nodes={n_nodes}"
        )

    y0 = np.zeros((len(compartment_labels), n_nodes), dtype=np.float64)

    for _, row in ic_df.iterrows():
        node_label = row[node_col]
        comp = row[comp_col]
        count = float(row[count_col])

        if node_label not in node_idx_map:
            if not allow_missing_subpops:
                raise ValueError(f"Unknown node label '{node_label}'")
            continue

        node = node_idx_map[node_label]

        if node >= n_nodes:
            if not allow_missing_subpops:
                raise ValueError(
                    f"Node index {node} out of bounds for n_nodes={n_nodes}"
                )
            continue

        if comp not in comp_idx_map:
            if not allow_missing_compartments:
                raise ValueError(
                    f"Compartment '{comp}' not in expected: {compartment_labels}"
                )
            continue

        y0[comp_idx_map[comp], node] += count

    if proportional_ic:
        raise NotImplementedError("Proportional IC support is not yet implemented.")

    return y0
