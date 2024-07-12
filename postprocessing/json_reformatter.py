from typing import Literal

from os.path import basename, getsize, dirname
import time
import json
from pathlib import Path
import argparse

import numpy as np
import numpy.typing as npt
import pandas as pd
import pyarrow.parquet as pq


def get_scenario_name(output_directory: str, scenario_name: str | None) -> str:
    """
    Determines the scenario name from the input directory if it is not explicitly
    provided.

    Parameters
    ----------
    output_directory : str
        The directory path from which the scenario name can be derived.
    scenario_name : str or None
        The explicit scenario name, if provided. If None, the scenario name is derived
        from the input directory's base name.

    Returns
    -------
    str
        The determined scenario name.

    Notes
    -----
    If `scenario_name` is provided, it is returned as is. Otherwise, the function
    returns the base name of the `input_directory`.
    """
    if scenario_name:
        return scenario_name
    scenario_name = basename(output_directory)
    if scenario_name == "":
        scenario_name = basename(dirname(output_directory))
    return scenario_name


def verbose_message(msg: str, start_time: None | float = None) -> None:
    """
    Prints a verbose message with a timestamp and optional elapsed time.

    Parameters
    ----------
    msg : str
        The message to be printed.
    start_time : None or float, optional
        The start time to calculate the elapsed time. If None, the elapsed time will not
        be included in the message. The default is `None`.

    Returns
    -------
    None
        This function does not return any value.

    Notes
    -----
    If the message does not end with a period, one will be added. The elapsed time is
    included if `start_time` is provided.
    """
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    formatted_msg = f"[{timestamp}] {msg}"
    if formatted_msg[-1] != ".":
        formatted_msg += "."
    if start_time:
        diff_time = time.time() - start_time
        formatted_msg += f" {diff_time:.2f} seconds elapsed."
    print(formatted_msg)


def read_sim_parquet_file(parquet_file: Path) -> tuple[int, pd.DataFrame]:
    """
    Reads a Parquet file and extracts the simulation ID and data.

    Parameters
    ----------
    parquet_file : Path
        The path to the Parquet file to be read.

    Returns
    -------
    tuple of (int, pd.DataFrame)
        A tuple containing the simulation ID (extracted from the file name) and a pandas
        DataFrame with the data from the Parquet file.

    Notes
    -----
    The simulation ID is assumed to be the integer part of the file name before the
    first dot.
    """
    sim_id = int(parquet_file.name.split(".", 1)[0])
    df = pq.read_table(str(parquet_file)).to_pandas()
    return (sim_id, df)


def read_sim_csv_file(csv_file: Path) -> tuple[int, pd.DataFrame]:
    """
    Reads a CSV file and extracts the simulation ID and data.

    Parameters
    ----------
    csv_file : Path
        The path to the CSV file to be read.

    Returns
    -------
    tuple of (int, pd.DataFrame)
        A tuple containing the simulation ID (extracted from the file name) and a pandas
        DataFrame with the data from the CSV file.

    Notes
    -----
    The simulation ID is assumed to be the integer part of the file name before the
    first dot.
    """
    sim_id = int(csv_file.name.split(".", 1)[0])
    df = pd.read_csv(filepath_or_buffer=str(csv_file))
    return (sim_id, df)


def read_hosp_parquet_file(file_path: Path, verbose: int) -> pd.DataFrame:
    """
    Reads a hosp Parquet file, transforms the data, and returns it as a pandas
    DataFrame.

    Parameters
    ----------
    file_path : Path
        The path to the hosp Parquet file to be read.
    verbose : int
        The verbosity level for logging messages. If `verbose` is 2 or higher, detailed
        messages will be printed during the process.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame containing the transformed data with columns: 'sim_id',
        'date', 'geoid', 'indicator', 'value'.
    """
    if verbose >= 2:
        start_time = time.time()
        verbose_message(msg=f"Reading {file_path.name}")
    sim_id, df = read_sim_parquet_file(parquet_file=file_path)
    if verbose >= 2:
        verbose_message(
            msg=f"Finished reading {file_path.name}",
            start_time=start_time,
        )
        start_time = time.time()
        verbose_message(msg=f"Transforming {file_path.name}")
    df.columns = df.columns.to_series().replace(
        {
            "subpop": "geoid",
            "place": "geoid",
        }
    )
    df = pd.melt(
        frame=df,
        id_vars=["date", "geoid"],
        var_name="indicator",
        value_name="value",
    )
    df = df[df["indicator"] != "time"]
    df["value"] = df["value"].astype(float)
    df["sim_id"] = sim_id
    df = df[["sim_id", "date", "geoid", "indicator", "value"]]
    if verbose >= 2:
        verbose_message(
            msg=f"Finished transforming {file_path.name}",
            start_time=start_time,
        )
    return df


def read_spar_parquet_file(file_path: Path, verbose: int) -> pd.DataFrame:
    """
    Reads a spar Parquet file, transforms the data, and returns it as a pandas
    DataFrame.

    Parameters
    ----------
    file_path : Path
        The path to the spar Parquet file to be read.
    verbose : int
        The verbosity level for logging messages. If `verbose` is 2 or higher, detailed
        messages will be printed during the process.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame containing the transformed data with columns: 'sim_id',
        'value'.
    """
    if verbose >= 2:
        start_time = time.time()
        verbose_message(msg=f"Reading {file_path.name}")
    sim_id, df = read_sim_parquet_file(parquet_file=file_path)
    if verbose >= 2:
        verbose_message(
            msg=f"Finished reading {file_path.name}",
            start_time=start_time,
        )
        start_time = time.time()
        verbose_message(msg=f"Transforming {file_path.name}")
    df = df[df["parameter"] == "r0"][["value"]]
    df["sim_id"] = sim_id
    df = df[["sim_id", "value"]]
    if verbose >= 2:
        verbose_message(
            msg=f"Finished transforming {file_path.name}",
            start_time=start_time,
        )
    return df


def read_seed_parquet_csv_file(file_path: Path, verbose: int) -> pd.DataFrame:
    """
    Reads a CSV or Parquet file and returns a transformed DataFrame.

    This function reads a file specified by `file_path`, which can be either
    in CSV or Parquet format. It then transforms the DataFrame by renaming
    columns and creating an 'indicator' column based on several other columns.
    If the `verbose` level is set to 2 or higher, it will print timing and
    transformation messages.

    Parameters
    ----------
    file_path : Path
        The path to the file to be read. The file can be in CSV or Parquet format.
    verbose : int
        The verbosity level. If 2 or higher, prints detailed messages about
        reading and transforming the file.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the transformed data with columns 'geoid',
        'date', 'indicator', and 'value'.

    Notes
    -----
    The function performs the following transformations on the DataFrame:
    - Renames 'subpop' and 'place' columns to 'geoid'.
    - Creates an 'indicator' column by concatenating 'source_infection_stage',
      'source_vaccination_stage', 'source_variant_type', and 'source_age_strata'.
    - Selects only the 'geoid', 'date', 'indicator', and 'value' columns
      for the final DataFrame.

    Examples
    --------
    >>> from pathlib import Path
    >>> df = read_seed_parquet_csv_file(Path('data.csv'), verbose=2)
    Reading data.csv
    Finished reading data.csv
    Transforming data.csv
    Finished transforming data.csv
    >>> df.head()
        geoid         dat     indicator  value
    0       1  2021-01-01  incid1_1_1_1     10
    1       2  2021-01-01  incid1_1_1_1     15
    """
    if verbose >= 2:
        start_time = time.time()
        verbose_message(msg=f"Reading {file_path.name}")
    if file_path.suffix == ".csv":
        _, df = read_sim_csv_file(csv_file=file_path)
    else:
        _, df = read_sim_parquet_file(parquet_file=file_path)
    if verbose >= 2:
        verbose_message(
            msg=f"Finished reading {file_path.name}",
            start_time=start_time,
        )
        start_time = time.time()
        verbose_message(msg=f"Transforming {file_path.name}")
    df.columns = df.columns.to_series().replace(
        {
            "subpop": "geoid",
            "place": "geoid",
            "amount": "value",
        }
    )
    df["indicator"] = (
        "incid"
        + df["source_infection_stage"].astype(str)
        + "_"
        + df["source_vaccination_stage"].astype(str)
        + "_"
        + df["source_variant_type"].astype(str)
        + "_"
        + df["source_age_strata"].astype(str)
    )
    df = df[["geoid", "date", "indicator", "value"]]
    if verbose >= 2:
        verbose_message(
            msg=f"Finished transforming {file_path.name}",
            start_time=start_time,
        )
    return df


def read_dataframe_from_folder(
    input_directory: str,
    file_type: Literal["spar", "hosp", "seed"],
    verbose: int,
) -> pd.DataFrame:
    """
    Reads and concatenates spar or hosp Parquet files from a specified directory into a
    pandas DataFrame.

    Parameters
    ----------
    input_directory : str
        The directory containing the Parquet files.
    file_type : {'spar', 'hosp', 'seed'}
        The type of files to read. Must be either 'spar', 'hosp', or 'seed'.
    verbose : int
        The verbosity level for logging messages. If `verbose` is 1 or higher, progress
        messages will be printed.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame containing the concatenated data from all specified Parquet
        files. The DataFrame will have columns 'sim_id', 'date', 'geoid', 'indicator', &
        'value' if file_type is 'hosp' or if the file_type is 'spar' the columns will be
        'sim_id' & 'value'.

    Notes
    -----
    If `verbose` is set to 1 or higher the function logs messages about the number of files found and the progress of reading and concatenating these files.
    """
    spar_hosp_seed_path = Path(input_directory)
    spar_hosp_seed_files = [
        x for x in spar_hosp_seed_path.glob("**/*.parquet") if x.is_file()
    ]
    if file_type == "seed":
        spar_hosp_seed_files += [
            x for x in spar_hosp_seed_path.glob("**/*.csv") if x.is_file()
        ]
    if verbose >= 1:
        start_time = time.time()
        verbose_message(
            msg=(
                f"Reading the {file_type} files in {str(spar_hosp_seed_path)}, "
                f"found {len(spar_hosp_seed_files)} parquet files"
            ),
        )
    file_reader = (
        read_hosp_parquet_file if file_type == "hosp" else read_spar_parquet_file
    )
    match file_type:
        case "spar":
            file_reader = read_spar_parquet_file
        case "hosp":
            file_reader = read_hosp_parquet_file
        case "seed":
            file_reader = read_seed_parquet_csv_file
        case _:
            raise ValueError(f"A file_type of '{file_type}' is not recognized.")
    df = pd.concat(
        objs=[
            file_reader(file_path=fp, verbose=verbose) for fp in spar_hosp_seed_files
        ],
    )
    if file_type == "seed":
        df = df.drop_duplicates(subset=["geoid", "date", "indicator"])
    if verbose >= 1:
        verbose_message(
            msg=(
                f"Finished reading the {file_type} files in "
                f"{str(spar_hosp_seed_path)}"
            ),
            start_time=start_time,
        )
    return df


def calculate_dates_series(
    hosp_df: pd.DataFrame,
) -> tuple[pd.Series, list[str]]:
    """
    Calculate a series of unique, sorted dates and their formatted string
    representations from a hosp DataFrame.

    Parameters
    ----------
    hosp_df : pd.DataFrame
        The DataFrame containing the hosp data with a 'date' column.

    Returns
    -------
    tuple of (pd.Series, list of str)
        A tuple containing:
        - dates_series: pd.Series
            A pandas Series of unique, sorted dates.
        - formatted_dates_list: list of str
            A list of formatted date strings in the 'YYYY-MM-DD' format.

    Notes
    -----
    The function extracts unique dates from the 'date' column of the input DataFrame,
    sorts them, and then returns them both as a pandas Series and as a list of formatted
    date strings.
    """
    dates_array = hosp_df["date"].unique()
    dates_array = dates_array[dates_array.argsort()]
    formatted_dates = dates_array.strftime("%Y-%m-%d").tolist()
    dates_series = pd.Series(
        data=dates_array,
        name="date",
    )
    return (dates_series, formatted_dates)


def calculate_indicator_array(
    hosp_df: pd.DataFrame,
) -> npt.NDArray[object]:
    """
    Calculate a sorted array of unique indicators from a DataFrame.

    This function extracts the unique values from the "indicator" column of the
    provided DataFrame, converts them to an object dtype, sorts them, and
    returns the resulting array.

    Parameters
    ----------
    hosp_df : pandas.DataFrame
        The DataFrame containing the "indicator" column from which unique
        indicators are to be extracted.

    Returns
    -------
    numpy.ndarray
        A sorted array of unique indicators with object dtype.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> hosp_df = pd.DataFrame({
    ...     "indicator": ["A", "B", "A", "C", "B", "C", "D"]
    ... })
    >>> calculate_indicator_array(hosp_df)
    array(['A', 'B', 'C', 'D'], dtype=object)
    """
    indicators = hosp_df["indicator"].unique().astype(object)
    indicators.sort()
    return indicators


def calculate_summary_from_hosp_dataframe(
    hosp_df: pd.DataFrame,
    func: callable = np.mean,
) -> pd.DataFrame:
    """
    Calculate a summary DataFrame from hosp data using a specified aggregation function.

    Parameters
    ----------
    hosp_df : pd.DataFrame
        The DataFrame containing the hosp data with columns 'date', 'geoid',
        'indicator', and 'value'.
    func : callable, optional
        The function to apply for aggregation. The default is `np.mean`.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the aggregated summary with columns 'date',
        'state_geoid', 'geoid', 'indicator', & 'value'.

    Notes
    -----
    The function groups the input DataFrame by 'date', 'geoid', and 'indicator', applies
    the specified aggregation function, resets the index, and adds a 'state_geoid'
    column by slicing the 'geoid' column.
    """
    summary_df = hosp_df[["date", "geoid", "indicator", "value"]].groupby(
        ["date", "geoid", "indicator"]
    )
    if func == np.mean:
        summary_df = summary_df.mean()
    else:
        summary_df = summary_df.apply(func=func)
    summary_df = summary_df.reset_index()
    summary_df["state_geoid"] = summary_df["geoid"].str.slice(stop=2)
    summary_df = summary_df[["date", "state_geoid", "geoid", "indicator", "value"]]
    return summary_df


def json_dump_efficient(obj: dict, file_path: Path) -> int:
    """
    Efficiently serialize an object to a JSON formatted stream.

    Parameters
    ----------
    obj : dict
        The dictionary to serialize.
    fp : Path
        The path which the JSON data will be written.

    Returns
    -------
    int
        The size of the file after writing the JSON data in bytes.

    Notes
    -----
    This function uses `json.dump` with optimizations:
    - `ensure_ascii=False` to allow non-ASCII characters, making the process faster.
    - `check_circular=False` to skip circular reference checking, which is valid for the
      data produced by this script.
    - `separators=(',', ':')` to minimize whitespace and trim the file size.
    """
    if obj:
        with file_path.open(mode="w") as fp:
            json.dump(
                obj=obj,
                fp=fp,
                ensure_ascii=False,
                check_circular=False,
                separators=(",", ":"),
            )
        return getsize(file_path)
    return 0


def write_outcomes_json(
    output_directory: str,
    indicator_array: npt.NDArray[object],
    verbose: int,
) -> str:
    """
    Write outcomes from a hosp DataFrame to a JSON file in the specified directory.

    Parameters
    ----------
    output_directory : str
        The directory where the output JSON file will be written.
    hosp_df : pd.DataFrame
        The DataFrame containing hosp data, which includes an 'indicator' column.
    indicator_array: npt.NDArray[object]
        A numpy array of indicator names.
    verbose : int
        The verbosity level for logging messages. If `verbose` is 1 or higher, progress
        messages will be printed.

    Returns
    -------
    str
        The path to the output JSON file as a string.

    Notes
    -----
    This function generates a JSON file named 'outcomes.json' in the specified output
    directory. It creates a dictionary of outcomes from the unique values in the
    'indicator' column of `hosp_df`, with each outcome assigned a unique ID. The JSON
    file is written using `json_dump_efficient`, which optimizes the file size and
    writing speed.

    If `verbose` is 1 or higher, the function logs messages indicating the start and
    completion of the writing process, including the size of the written file.
    """
    output_path = Path(output_directory, "outcomes.json")
    if verbose >= 1:
        start_time = time.time()
        verbose_message(msg=f"Writing outcomes to {output_path.name}.")
    outcomes = {}
    for i, indicator in enumerate(indicator_array):
        j = i + 1
        outcomes[str(j)] = {
            "id": j,
            "key": indicator,
            "name": indicator,
        }
    file_size = json_dump_efficient(obj=outcomes, file_path=output_path)
    if verbose >= 1:
        verbose_message(
            msg=f"Finished writing {file_size} bytes to {output_path.name}",
            start_time=start_time,
        )
    return str(output_path)


def write_geo_jsons(
    output_directory: str,
    hosp_df: pd.DataFrame,
    spar_df: pd.DataFrame,
    dates_series: pd.Series,
    formatted_dates: list[str],
    indicator_array: npt.NDArray[object],
    scenario_name: str,
    severity_name: str,
    verbose: int,
) -> list[str]:
    """
    Write geographical JSON files from hosp and spar data.

    Parameters
    ----------
    output_directory : str
        The directory where the output JSON files will be written.
    hosp_df : pd.DataFrame
        The DataFrame containing hosp data.
    spar_df : pd.DataFrame
        The DataFrame containing spar data.
    dates_series : pd.Series
        A Series of dates used for merging data.
    formatted_dates: list[str]
        A list of dates formatted in "YYYY-MM-DD" format.
    indicator_array: npt.NDArray[object]
        A numpy array of indicator names.
    scenario_name : str
        The name of the scenario.
    severity_name : str
        The name of the severity level.
    verbose : int
        The verbosity level for logging messages. If `verbose` is 1 or higher, progress
        messages will be printed.

    Returns
    -------
    list of str
        A list of paths to the written JSON files.

    Notes
    -----
    This function generates JSON files for each unique geoid in `hosp_df`. It groups the
    data by 'geoid' and writes each group to a separate JSON file. The JSON files
    include nested structures for scenarios, severity levels, and indicators, with each
    indicator containing data from multiple simulations. The function logs progress
    messages if `verbose` is set to 1 or higher.
    """
    if verbose >= 1:
        files_start_time = time.time()
        verbose_message(msg=f"Writing geo JSONs.")
    written_files = []
    total_file_size = 0
    for geoid, geoid_df in hosp_df.groupby("geoid"):
        if geoid[2:] == "000":
            geoid = geoid[:2]
        output_path = Path(output_directory, f"{geoid}.json")
        if verbose >= 2:
            file_start_time = time.time()
            verbose_message(
                msg=f"Writing geo JSON for {output_path.name}.",
            )
        data = {}
        data[scenario_name] = {}
        data[scenario_name]["dates"] = formatted_dates
        data[scenario_name][severity_name] = {}
        for indicator in indicator_array:
            data[scenario_name][severity_name][indicator] = []
            indicator_df = pd.merge(
                left=dates_series,
                right=geoid_df[geoid_df["indicator"] == indicator][
                    ["date", "sim_id", "value"]
                ],
                how="left",
                on="date",
            )
            # Loop over the sims
            for sim_id, sim_df in indicator_df.groupby("sim_id"):
                r0 = float(spar_df[spar_df["sim_id"] == sim_id]["value"].iloc[0])
                data[scenario_name][severity_name][indicator].append(
                    {
                        "name": str(sim_id),
                        "max": sim_df["value"].max(),
                        "vals": sim_df["value"].tolist(),
                        "over": True,
                        "r0": r0,
                    }
                )
        file_size = json_dump_efficient(obj=data, file_path=output_path)
        total_file_size += file_size
        written_files.append(str(output_path))
        if verbose >= 2:
            verbose_message(
                msg=f"Finished writing {file_size} bytes to {output_path.name}",
                start_time=file_start_time,
            )
    if verbose >= 1:
        verbose_message(
            msg=(
                f"Finished writing {total_file_size} bytes to {len(written_files)} "
                "geo JSONs."
            ),
            start_time=files_start_time,
        )
    return written_files


def write_actuals_jsons(
    output_directory: str,
    seed_df: pd.DataFrame,
    indicator_array: npt.NDArray[object],
    verbose: int,
) -> list[str]:
    """
    Write actuals JSON files from seed data.

    Parameters
    ----------
    output_directory : str
        The directory where the output JSON files will be written.
    seed_df : pd.DataFrame
        The DataFrame containing seed data.
    indicator_array: npt.NDArray[object]
        A numpy array of indicator names.
    verbose : int
        The verbosity level for logging messages. If `verbose` is 1 or higher, progress
        messages will be printed.

    Returns
    -------
    list of str
        A list of paths to the written JSON files.

    Notes
    -----
    This function generates JSON files for each unique geoid in `seed_df`. It groups the
    data by 'geoid' and writes each group to a separate JSON file. The JSON files
    include for each indicator a list of dates and values. The function logs progress
    messages if `verbose` is set to 1 or higher.
    """
    if verbose >= 1:
        files_start_time = time.time()
        verbose_message(msg=f"Writing actuals JSONs.")
    written_files = []
    total_file_size = 0
    seed_df = seed_df[seed_df["indicator"].isin(indicator_array)]
    seed_df["date"] = pd.to_datetime(seed_df["date"]).dt.strftime("%Y-%m-%d")
    for geoid, geoid_df in seed_df.groupby("geoid"):
        if geoid[2:] == "000":
            geoid = geoid[:2]
        output_path = Path(output_directory, f"{geoid}.json")
        if verbose >= 2:
            file_start_time = time.time()
            verbose_message(
                msg=f"Writing actuals JSON for {output_path.name}.",
            )
        data = {}
        for indicator, indicator_df in geoid_df.groupby("indicator"):
            data[indicator] = indicator_df[["date", "value"]].to_dict("records")
        file_size = json_dump_efficient(obj=data, file_path=output_path)
        total_file_size += file_size
        written_files.append(str(output_path))
        if verbose >= 2:
            verbose_message(
                msg=f"Finished writing {file_size} bytes to {output_path.name}",
                start_time=file_start_time,
            )
    if verbose >= 1:
        verbose_message(
            msg=(
                f"Finished writing {total_file_size} bytes to {len(written_files)} "
                "actuals JSONs."
            ),
            start_time=files_start_time,
        )
    return written_files


def write_stats_for_map_json(
    output_directory: str,
    summary_df: pd.DataFrame,
    dates_series: pd.Series,
    indicator_array: npt.NDArray[object],
    scenario_name: str,
    verbose: int,
) -> str:
    """
    Write statistics for the map visual to a JSON file in the specified output
    directory.

    Parameters
    ----------
    output_directory : str
        The directory where the output JSON file will be written.
    summary_df : pd.DataFrame
        The DataFrame containing the summary data.
    dates_series : pd.Series
        A Series of dates used for merging data.
    indicator_array: npt.NDArray[object]
        A numpy array of indicator names.
    scenario_name : str
        The name of the scenario.
    verbose : int
        The verbosity level for logging messages. If `verbose` is 1 or higher, progress
        messages will be printed.

    Returns
    -------
    str
        The path to the output JSON file as a string.

    Notes
    -----
    This function generates a JSON file named 'statsForMap.json' in the specified output
    directory. It groups the data by 'state_geoid' and 'geoid', then organizes it by
    scenario and indicator. Each indicator's values are merged with a series of dates
    and stored in the JSON structure. The function logs progress messages if `verbose`
    is set to 1 or higher.
    """
    output_path = Path(output_directory, "statsForMap.json")
    if verbose >= 1:
        start_time = time.time()
        verbose_message(msg=f"Writing outcomes to {output_path.name}.")
    stats_for_map = {}
    for state_geoid, state_df in summary_df.groupby("state_geoid"):
        stats_for_map[state_geoid] = {}
        for county_geoid, county_df in state_df.groupby("geoid"):
            if county_geoid[2:] == "000":
                # Misnomer, county is actually the state
                county_geoid = state_geoid
            stats_for_map[state_geoid][county_geoid] = {}
            county_map = stats_for_map[state_geoid][county_geoid]
            county_map[scenario_name] = {}
            for indicator in indicator_array:
                indicator_df = pd.merge(
                    left=dates_series,
                    right=county_df,
                    how="left",
                    on="date",
                )
                county_map[scenario_name][indicator] = indicator_df["value"].tolist()
    file_size = json_dump_efficient(obj=stats_for_map, file_path=output_path)
    if verbose >= 1:
        verbose_message(
            msg=f"Finished writing {file_size} bytes to {output_path.name}.",
            start_time=start_time,
        )
    return str(output_path)


def write_valid_geoids_json(
    output_directory: str,
    summary_df: pd.DataFrame,
    verbose: int,
) -> str:
    """
    Write unique, valid geoids to a JSON file.

    This function extracts unique geoids from the provided DataFrame,
    processes them to ensure validity, sorts them, and writes them to a JSON
    file in the specified output directory. The verbosity level controls the
    amount of progress information printed.

    Parameters
    ----------
    output_directory : str
        The directory where the JSON file will be saved.
    summary_df : pandas.DataFrame
        The DataFrame containing the "geoid" column from which unique geoids
        are to be extracted.
    verbose : int
        The verbosity level. If >= 1, progress information is printed.

    Returns
    -------
    str
        The path to the created JSON file as a string.

    Examples
    --------
    >>> import pandas as pd
    >>> summary_df = pd.DataFrame({
    ...     'geoid': ['123000', '456000', '789000', '123456']
    ... })
    >>> write_valid_geoids_json('output', summary_df, verbose=0)
    'output/validGeoids.json'
    """
    output_path = Path(output_directory, "validGeoids.json")
    if verbose >= 1:
        start_time = time.time()
        verbose_message(msg=f"Writing valid geoids to {output_path.name}.")
    geoids = summary_df["geoid"].unique().tolist()
    geoids = [g[:2] if g[2:] == "000" else g for g in geoids]
    geoids.sort()
    data = {
        "geoids": geoids,
    }
    file_size = json_dump_efficient(obj=data, file_path=output_path)
    if verbose >= 1:
        verbose_message(
            msg=f"Finished writing {file_size} bytes to {output_path.name}.",
            start_time=start_time,
        )
    return str(output_path)


def main() -> None:
    # Parse given arguments
    parser = argparse.ArgumentParser(
        prog="flepiMoP Parquet To JSON",
        description=(
            "Convert the outputs of a flepiMoP run from parquet to JSON for use by the "
            "dashboard app."
        ),
    )
    parser.add_argument(
        "--hosp-directory",
        action="store",
        type=str,
        help="The directory to read hosp input from.",
        required=True,
    )
    parser.add_argument(
        "--spar-directory",
        action="store",
        type=str,
        help="The directory to read spar input from.",
        required=True,
    )
    parser.add_argument(
        "--seed-directory",
        action="store",
        type=str,
        help="The directory to read seed input from.",
    )
    parser.add_argument(
        "--output-directory",
        action="store",
        type=str,
        help="The directory to output the JSON files to.",
        required=True,
    )
    parser.add_argument(
        "--scenario-name",
        action="store",
        type=str,
        help=(
            "The scenario short name to use for display in the dashboard app. If not "
            "given the basename of the output directory will be used."
        ),
    )
    parser.add_argument(
        "--severity-name",
        action="store",
        nargs=1,
        type=str,
        default="low",
        choices=["low", "med", "high"],
        help="The severity name to use for display in the dashboard app.",
    )
    parser.add_argument(
        "--rounding",
        action="store",
        type=int,
        help=(
            "Number of digits to round numbers to in output JSON, -1 for no rounding."
            " Can significantly reduce file size and dashboard latency."
        ),
        default=-1,
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        help=(
            "Verbose flag for printing information on input data sizes, timing, and "
            "output file sizes to stdout."
        ),
        default=0,
    )
    args = parser.parse_args()
    # First, get the scenario name of the flepiMoP run
    args.scenario_name = get_scenario_name(
        output_directory=args.output_directory,
        scenario_name=args.scenario_name,
    )
    if args.verbose >= 1:
        verbose_message(msg=f"Using '{args.scenario_name}' as the scenario name.")
    # Next process our raw data in
    spar_df = read_dataframe_from_folder(
        input_directory=args.spar_directory,
        file_type="spar",
        verbose=args.verbose,
    )
    hosp_df = read_dataframe_from_folder(
        input_directory=args.hosp_directory,
        file_type="hosp",
        verbose=args.verbose,
    )
    if args.seed_directory:
        seed_df = read_dataframe_from_folder(
            input_directory=args.seed_directory,
            file_type="seed",
            verbose=args.verbose,
        )
    # Then calculate some data based on our raw hosp_df as well as some light
    # rounding
    if args.verbose >= 1:
        start_time = time.time()
        if args.rounding > -1:
            msg = "Calculating dates series, summary dataframe, and applying rounding"
        else:
            msg = "Calculating dates series and summary dataframe"
        verbose_message(msg=msg)
    dates_series, formatted_dates = calculate_dates_series(hosp_df=hosp_df)
    summary_df = calculate_summary_from_hosp_dataframe(hosp_df=hosp_df)
    indicator_array = calculate_indicator_array(hosp_df=hosp_df)
    if args.rounding > -1:
        # Calculate the rounding *after* calculating the summary to not
        # propagate rounding errors
        hosp_df["value"] = hosp_df["value"].round(decimals=args.rounding)
        summary_df["value"] = summary_df["value"].round(decimals=args.rounding)
        if args.seed_directory:
            seed_df["value"] = seed_df["value"].round(decimals=args.rounding)
    if args.verbose >= 1:
        if args.rounding > -1:
            msg = (
                "Finished calculating dates series, summary dataframe, and applying "
                "rounding"
            )
        else:
            msg = "Finished calculating dates series and summary dataframe"
        verbose_message(msg=msg, start_time=start_time)
    # Write the JSONs out
    write_outcomes_json(
        output_directory=args.output_directory,
        indicator_array=indicator_array,
        verbose=args.verbose,
    )
    write_geo_jsons(
        output_directory=args.output_directory,
        hosp_df=hosp_df,
        spar_df=spar_df,
        dates_series=dates_series,
        formatted_dates=formatted_dates,
        indicator_array=indicator_array,
        scenario_name=args.scenario_name,
        severity_name=args.severity_name,
        verbose=args.verbose,
    )
    if args.seed_directory:
        write_actuals_jsons(
            output_directory=args.output_directory,
            seed_df=seed_df,
            indicator_array=indicator_array,
            verbose=args.verbose,
        )
    write_stats_for_map_json(
        output_directory=args.output_directory,
        summary_df=summary_df,
        dates_series=dates_series,
        indicator_array=indicator_array,
        scenario_name=args.scenario_name,
        verbose=args.verbose,
    )
    write_valid_geoids_json(
        output_directory=args.output_directory,
        summary_df=summary_df,
        verbose=args.verbose,
    )
    if args.verbose >= 1:
        verbose_message(
            msg="Finished transforming flepiMoP output for use in the dashboard",
        )


if __name__ == "__main__":
    main()
