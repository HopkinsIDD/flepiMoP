"""
Defines class, methods, and functions necessary to establising compartments in the model.

Classes:
    Compartments: An object to handle compartment data for the model.

Functions:
    get_list_dimension: Returns length of object passed (if a list); otherwise returns 1.
    list_access_element_safe: Attempts to access something from the given object at specified index.
    list_access_element: Attempts to access soemthing from the given object at specified index.
    list_recursive_convert_to_string: Convert item(s) in object to str(s).
    compartments: A container for subcommands related to the compartmental model.
    plot: Generate a plot representing transition between compartments.
    export: Export compartment data to a CSV file.
"""

import logging
from functools import reduce
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from click import pass_context, Context

from .utils import config, Timer, as_list
from .shared_cli import config_files_argument, config_file_options, parse_config_files, cli

logger = logging.getLogger(__name__)


class Compartments:
    """
    An object to handle compartment data for the model.

    Attributes:
        times_set: Counter to track succesful data intialization from config.
    """

    # Minimal object to be easily picklable for // runs
    def __init__(
        self,
        seir_config=None,
        compartments_config=None,
        compartments_file=None,
        transitions_file=None,
    ):
        """
        Initializes a `Compartments` object.

        Args:
            seir_config: Config file for SEIR model construction information.
            compartments_config: Config file for compartment information.
            compartments_file: File to specify compartment information.
            transitions_file: File to specify transition information.
        """
        self.times_set = 0

        ## Something like this is needed for check script:
        if (not compartments_file is None) and (not transitions_file is None):
            self.fromFile(compartments_file, transitions_file)
            self.times_set += 1
        if (self.times_set == 0) and (compartments_config is not None):
            self.constructFromConfig(seir_config, compartments_config)
            self.times_set += 1

        if self.times_set == 0:
            raise ValueError("Compartments object not set, no config or file provided.")
        return

    def constructFromConfig(self, seir_config, compartment_config) -> None:
        """
        This method is called by the constructor if the compartments are not loaded from a file.
        It will parse the compartments and transitions from the configuration files.
        It will populate dynamic class attributes `compartments` and `transitions`.
        """
        self.compartments = self.parse_compartments(seir_config, compartment_config)
        self.transitions = self.parse_transitions(seir_config, False)

    def __eq__(self, other):
        return (self.transitions == other.transitions).all().all() and (
            self.compartments == other.compartments
        ).all().all()

    def parse_compartments(self, seir_config: dict, compartment_config) -> pd.DataFrame:
        """
        Parses compartment configurations and returns a DataFrame of compartment names.

        Args:
            seir_config: Configuraton information for model.
            compartment_config: Configuration information for model comartments.

        Returns:
            A DataFrame where each row is a unique compartment and columns
            correspond to attributes of the compartments.
        """
        compartment_df = None
        for compartment_name, compartment_value in compartment_config.get().items():
            tmp = pd.DataFrame({"key": 1, compartment_name: compartment_value})
            if compartment_df is None:
                compartment_df = tmp
            else:
                compartment_df = pd.merge(compartment_df, tmp, on="key")
        compartment_df = compartment_df.drop(["key"], axis=1)
        compartment_df["name"] = compartment_df.apply(
            lambda x: reduce(lambda a, b: a + "_" + b, x), axis=1
        )
        return compartment_df

    def parse_transitions(
        self, seir_config: dict, fake_config: bool = False
    ) -> pd.DataFrame:
        """
        Parses the transitions defined in config and returns a concatenated DataFrame.

        Args:
            seir_config: Configuraton information for model.
            fake_config:
                Flag indicating whether or not transitions provied are placeholders.
                Default value is False.
        Returns:
            A DataFrame containing all transitions from the config.
        """
        rc = reduce(
            lambda a, b: pd.concat(
                [a, self.parse_single_transition(seir_config, b, fake_config)]
            ),
            seir_config["transitions"],
            pd.DataFrame(),
        )
        rc = rc.reset_index(drop=True)
        return rc

    def check_transition_element(self, single_transition_config, problem_dimension=None):
        return True

    def check_transition_elements(self, single_transition_config, problem_dimension):
        return True

    def access_original_config_by_multi_index(
        self, config_piece, index, dimension=None, encapsulate_as_list=False
    ):
        if dimension is None:
            dimension = [None for i in index]
        tmp = [y for y in zip(index, range(len(index)), dimension)]
        tmp = zip(index, range(len(index)), dimension)
        tmp = [
            list_access_element_safe(config_piece[x[1]], x[0], x[2], encapsulate_as_list)
            for x in tmp
        ]
        return tmp

    def expand_transition_elements(self, single_transition_config, problem_dimension):
        proportion_size = get_list_dimension(single_transition_config["proportional_to"])
        new_transition_config = single_transition_config.copy()

        # replace "source" by the actual source from the config
        for p_idx in range(proportion_size):
            if new_transition_config["proportional_to"][p_idx] == "source":
                new_transition_config["proportional_to"][p_idx] = new_transition_config[
                    "source"
                ]

        temp_array = np.zeros(problem_dimension)

        new_transition_config["source"] = np.zeros(problem_dimension, dtype=object)
        new_transition_config["destination"] = np.zeros(problem_dimension, dtype=object)
        new_transition_config["rate"] = np.zeros(problem_dimension, dtype=object)

        new_transition_config["proportional_to"] = np.zeros(problem_dimension, dtype=object)
        new_transition_config["proportion_exponent"] = np.zeros(
            problem_dimension, dtype=object
        )

        it = np.nditer(
            temp_array, flags=["multi_index"]
        )  # it is an iterator that will go through all the indexes of the array
        for x in it:
            try:
                new_transition_config["source"][it.multi_index] = (
                    list_recursive_convert_to_string(
                        self.access_original_config_by_multi_index(
                            single_transition_config["source"], it.multi_index
                        )
                    )
                )
            except Exception as e:
                print(f"Error {e}:")
                print(
                    f">>> in expand_transition_elements for `source:` at index '{it.multi_index}'"
                )
                print(
                    f">>> this transition source is: '{single_transition_config['source']}'"
                )
                print(
                    f">>> this transition destination is: '{single_transition_config['destination']}'"
                )
                print(f"transition_dimension: '{problem_dimension}'")
                raise e

            try:
                new_transition_config["destination"][it.multi_index] = (
                    list_recursive_convert_to_string(
                        self.access_original_config_by_multi_index(
                            single_transition_config["destination"], it.multi_index
                        )
                    )
                )
            except Exception as e:
                print(f"Error {e}:")
                print(
                    f">>> in expand_transition_elements for `destination:` at index '{it.multi_index}'"
                )
                print(
                    f">>> this transition source is: '{single_transition_config['source']}'"
                )
                print(
                    f">>> this transition destination is: '{single_transition_config['destination']}'"
                )
                print(f"transition_dimension: '{problem_dimension}'")
                raise e

            try:
                new_transition_config["rate"][it.multi_index] = (
                    list_recursive_convert_to_string(
                        self.access_original_config_by_multi_index(
                            single_transition_config["rate"], it.multi_index
                        )
                    )
                )
            except Exception as e:
                print(f"Error {e}:")
                print(
                    f">>> in expand_transition_elements for `rate:` at index '{it.multi_index}'"
                )
                print(
                    f">>> this transition source is: '{single_transition_config['source']}'"
                )
                print(
                    f">>> this transition destination is: '{single_transition_config['destination']}'"
                )
                print(f"transition_dimension: '{problem_dimension}'")
                raise e

            try:
                new_transition_config["proportional_to"][it.multi_index] = as_list(
                    list_recursive_convert_to_string(
                        [
                            self.access_original_config_by_multi_index(
                                single_transition_config["proportional_to"][p_idx],
                                it.multi_index,
                                problem_dimension,
                                True,
                            )
                            for p_idx in range(proportion_size)
                        ]
                    )
                )
            except Exception as e:
                print(f"Error {e}:")
                print(
                    f">>> in expand_transition_elements for `proportional_to:` at index '{it.multi_index}'"
                )
                print(
                    f">>> this transition source is: '{single_transition_config['source']}'"
                )
                print(
                    f">>> this transition destination is: '{single_transition_config['destination']}'"
                )
                print(f"transition_dimension: '{problem_dimension}'")
                raise e

            if (
                "proportion_exponent" in single_transition_config
            ):  # if proportion_exponent is not defined, it is set to 1
                try:
                    self.access_original_config_by_multi_index(
                        single_transition_config["proportion_exponent"][0],
                        it.multi_index,
                        problem_dimension,
                    )
                    new_transition_config["proportion_exponent"][it.multi_index] = (
                        list_recursive_convert_to_string(
                            [
                                self.access_original_config_by_multi_index(
                                    single_transition_config["proportion_exponent"][p_idx],
                                    it.multi_index,
                                    problem_dimension,
                                )
                                for p_idx in range(proportion_size)
                            ]
                        )
                    )
                except Exception as e:
                    print(f"Error {e}:")
                    print(
                        f">>> in expand_transition_elements for `proportion_exponent:` at index '{it.multi_index}'"
                    )
                    print(
                        f">>> this transition source is: '{single_transition_config['source']}'"
                    )
                    print(
                        f">>> this transition destination is: '{single_transition_config['destination']}'"
                    )
                    print(f"transition_dimension: '{problem_dimension}'")
                    raise e
            else:
                new_transition_config["proportion_exponent"][it.multi_index] = [
                    "1"
                ] * proportion_size

        return new_transition_config

    def format_source(self, source_column):
        rc = [
            y
            for y in map(
                lambda x: reduce(lambda a, b: str(a) + "_" + str(b), x), source_column
            )
        ]
        return rc

    def unformat_source(self, source_column):
        rc = [x.split("_") for x in source_column]
        return rc

    def format_destination(self, destination_column):
        rc = [
            y
            for y in map(
                lambda x: reduce(lambda a, b: str(a) + "_" + str(b), x),
                destination_column,
            )
        ]
        return rc

    def unformat_destination(self, destination_column):
        rc = [x.split("_") for x in destination_column]
        return rc

    def format_rate(self, rate_column):
        rc = [
            y
            for y in map(
                lambda x: reduce(lambda a, b: str(a) + "%*%" + str(b), x), rate_column
            )
        ]
        return rc

    def unformat_rate(self, rate_column, compartment_dimension):
        rc = [x.split("%*%", maxsplit=compartment_dimension - 1) for x in rate_column]
        for row in rc:
            while len(row) < compartment_dimension:
                row.append(1)
        return rc

    def format_proportional_to(self, proportional_to_column):
        rc = [
            y
            for y in map(
                lambda x: reduce(
                    lambda a, b: str(a) + "*" + str(b),
                    map(
                        lambda x: reduce(
                            lambda a, b: str(a) + "_" + str(b),
                            map(
                                lambda x: reduce(
                                    lambda a, b: str(a) + "+" + str(b), as_list(x)
                                ),
                                x,
                            ),
                        ),
                        x,
                    ),
                ),
                proportional_to_column,
            )
        ]
        return rc

    def unformat_proportional_to(self, proportional_to_column):
        rc = [x.split("*") for x in proportional_to_column]
        for row in range(len(rc)):
            rc[row] = [x.split("_") for x in rc[row]]
            for elem in range(len(rc[row])):
                rc[row][elem] = [x.split("+") for x in as_list(rc[row][elem])]
        return rc

    def format_proportion_exponent(self, proportion_exponent_column):
        rc = [
            y
            for y in map(
                lambda x: reduce(
                    lambda a, b: str(a) + "%*%" + str(b),
                    map(lambda x: reduce(lambda a, b: str(a) + "*" + str(b), x), x),
                ),
                proportion_exponent_column,
            )
        ]
        return rc

    def unformat_proportion_exponent(
        self, proportion_exponent_column, compartment_dimension
    ):
        rc = [x.split("%*%") for x in proportion_exponent_column]
        for row in range(len(rc)):
            rc[row] = [x.split("*", maxsplit=compartment_dimension - 1) for x in rc[row]]
            for elem in rc[row]:
                while len(elem) < compartment_dimension:
                    elem.append(1)
        return rc

    def parse_single_transition(
        self, seir_config, single_transition_config, fake_config=False
    ):
        ## This method relies on having run parse_compartments
        if not fake_config:
            single_transition_config = single_transition_config.get()
        self.check_transition_element(single_transition_config["source"])
        self.check_transition_element(single_transition_config["destination"])
        source_dimension = [
            get_list_dimension(x) for x in single_transition_config["source"]
        ]
        destination_dimension = [
            get_list_dimension(x) for x in single_transition_config["destination"]
        ]
        problem_dimension = reduce(
            lambda x, y: max(x, y), (source_dimension, destination_dimension)
        )
        self.check_transition_elements(single_transition_config, problem_dimension)
        transitions = self.expand_transition_elements(
            single_transition_config, problem_dimension
        )

        tmp_array = np.zeros(problem_dimension)
        it = np.nditer(tmp_array, flags=["multi_index"])
        rc = reduce(
            lambda a, b: pd.concat([a, b]),
            [
                pd.DataFrame(
                    {
                        "source": [transitions["source"][it.multi_index]],
                        "destination": [transitions["destination"][it.multi_index]],
                        "rate": [transitions["rate"][it.multi_index]],
                        "proportional_to": [transitions["proportional_to"][it.multi_index]],
                        "proportion_exponent": [
                            transitions["proportion_exponent"][it.multi_index]
                        ],
                    },
                    index=[0],
                )
                for x in it
            ],
        )

        return rc

    def toFile(
        self,
        compartments_file="compartments.parquet",
        transitions_file="transitions.parquet",
        write_parquet=True,
    ):
        out_df = self.compartments.copy()
        if write_parquet:
            pa_df = pa.Table.from_pandas(out_df, preserve_index=False)
            pa.parquet.write_table(pa_df, compartments_file)
        else:
            out_df.to_csv(compartments_file, index=False)

        out_df = self.transitions.copy()
        out_df["source"] = self.format_source(out_df["source"])
        out_df["destination"] = self.format_destination(out_df["destination"])
        out_df["rate"] = self.format_rate(out_df["rate"])
        out_df["proportional_to"] = self.format_proportional_to(out_df["proportional_to"])
        out_df["proportion_exponent"] = self.format_proportion_exponent(
            out_df["proportion_exponent"]
        )
        if write_parquet:
            pa_df = pa.Table.from_pandas(out_df, preserve_index=False)
            pa.parquet.write_table(pa_df, transitions_file)
        else:
            out_df.to_csv(transitions_file, index=False)
        return

    def fromFile(self, compartments_file, transitions_file):
        self.compartments = pq.read_table(compartments_file).to_pandas()
        self.transitions = pq.read_table(transitions_file).to_pandas()
        compartment_dimension = self.compartments.shape[1] - 1
        self.transitions["source"] = self.unformat_source(self.transitions["source"])
        self.transitions["destination"] = self.unformat_destination(
            self.transitions["destination"]
        )
        self.transitions["rate"] = self.unformat_rate(
            self.transitions["rate"], compartment_dimension
        )
        self.transitions["proportional_to"] = self.unformat_proportional_to(
            self.transitions["proportional_to"]
        )
        self.transitions["proportion_exponent"] = self.unformat_proportion_exponent(
            self.transitions["proportion_exponent"], compartment_dimension
        )

        return

    def get_comp_idx(self, comp_dict: dict, error_info: str = "no information") -> int:
        """
        Return the index of a compartiment given a filter. The filter has to isolate a compartment,
        but it ignores columns that don't exist:

        Args:
            comp_dict:
                A dictionary where keys are compartment names and
                values are values to filter by for each column.
            error_info:
                A message providing additional context about where the
                the method was called from. Default value is "no information".

        Returns:
            Index of the comaprtment that matches the filter.

        Raises:
            ValueError: Filter results in more than one or zero matches.
        """
        mask = pd.concat(
            [self.compartments[k] == v for k, v in comp_dict.items()], axis=1
        ).all(axis=1)
        comp_idx = self.compartments[mask].index.values
        if len(comp_idx) != 1:
            raise ValueError(
                f"The provided dictionary does not allow an isolated compartment: '{comp_dict}'. "
                f"Isolate '{self.compartments[mask]}'. "
                f"The `get_comp_idx` function was called by '{error_info}'."
            )
        return comp_idx[0]

    def get_ncomp(self) -> int:
        return len(self.compartments)

    def get_transition_array(self) -> tuple:
        """
        Constructs the transition matrix for the model.

        Returns:
            tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
                - unique_strings: unique strings from `proportion_exponent` and `rate`
                - transition_array: array representing transitions and corresponding compartment indices.
                - proportion_array: array representing proportion compartment indices
                - proportion_info: array containing start and end indices for proportions

        Raises:
            ValueError: If term is not found in list of valid compartments.
            ValueErrror: If any string in `rate` or `proportional_to` is an invalid candidate.
        """
        with Timer("SEIR.compartments"):
            transition_array = np.zeros(
                (self.transitions.shape[1], self.transitions.shape[0]), dtype="int64"
            )
            for cit, colname in enumerate(("source", "destination")):
                for it, elem in enumerate(self.transitions[colname]):
                    elem = reduce(lambda a, b: a + "_" + b, elem)
                    rc = -1
                    for compartment in range(self.compartments.shape[0]):
                        if self.compartments["name"][compartment] == elem:
                            rc = compartment
                    if rc == -1:
                        raise ValueError(
                            f"Could not find '{colname}' defined by '{elem}' in '{self.compartments}'."
                        )
                    transition_array[cit, it] = rc

            unique_strings = []
            for x in self.transitions["proportion_exponent"]:
                for y in x:
                    candidate = reduce(lambda a, b: a + "*" + b, y)
                    candidate = candidate.replace(" ", "")
                    # candidate = candidate.replace("*1", "")
                    if not candidate in unique_strings:
                        unique_strings.append(candidate)

            for x in self.transitions["rate"]:
                candidate = reduce(lambda a, b: a + "*" + b, x)
                candidate = candidate.replace(" ", "")
                # candidate = candidate.replace("*1", "")
                if not candidate in unique_strings:
                    unique_strings.append(candidate)

            # parenthesis are now supported
            # assert reduce(lambda a, b: a and b, [(x.find("(") == -1) for x in unique_strings])
            # assert reduce(lambda a, b: a and b, [(x.find(")") == -1) for x in unique_strings])
            assert reduce(
                lambda a, b: a and b, [(x.find("%") == -1) for x in unique_strings]
            )
            assert reduce(
                lambda a, b: a and b, [(x.find(" ") == -1) for x in unique_strings]
            )

            for it, elem in enumerate(self.transitions["rate"]):
                candidate = reduce(lambda a, b: a + "*" + b, elem)
                candidate = candidate.replace(" ", "")
                # candidate = candidate.replace("*1", "")
                if candidate not in unique_strings:
                    raise ValueError(
                        f"Candidate '{candidate}' from 'rate' column is not in the list of unique strings: {unique_strings}."
                    )
                rc = [it for it, x in enumerate(unique_strings) if x == candidate][0]
                transition_array[2][it] = rc

            current_proportion_start = 0
            for it, elem in enumerate(self.transitions["proportional_to"]):
                transition_array[3][it] = current_proportion_start
                transition_array[4][it] = current_proportion_start + len(elem)
                current_proportion_start += len(elem)

            proportion_info = np.zeros((3, transition_array[4].max()), dtype="int64")
            current_proportion_sum_start = 0
            current_proportion_sum_it = 0
            for it, elem in enumerate(self.transitions["proportional_to"]):
                for it2, elem2 in enumerate(elem):
                    elem_tmp = [
                        w
                        for w in pd.DataFrame(index=pd.MultiIndex.from_product(elem2))
                        .reset_index()
                        .apply(lambda z: reduce(lambda x, y: f"{x}_{y}", z), axis=1)
                    ]

                    # for it3, elem3 in enumerate(elem_tmp):
                    #     rc = -1
                    #     for compartment in range(self.compartments.shape[0]):
                    #         if self.compartments["name"][compartment] == elem3:
                    #             rc = compartment
                    #     if rc == -1:
                    #         raise ValueError(f"Could not find match for {elem3} in compartments")
                    proportion_info[0][
                        current_proportion_sum_it
                    ] = current_proportion_sum_start
                    proportion_info[1][current_proportion_sum_it] = (
                        current_proportion_sum_start + len(elem_tmp)
                    )
                    current_proportion_sum_it += 1
                    current_proportion_sum_start += len(elem_tmp)
            proportion_compartment_index = 0
            for it, elem in enumerate(self.transitions["proportion_exponent"]):
                for y in elem:
                    candidate = reduce(lambda a, b: a + "*" + b, y)
                    candidate = candidate.replace(" ", "")
                    # candidate = candidate.replace("*1", "")
                    if candidate not in unique_strings:
                        raise ValueError(
                            f"Proportion exponent '{candidate}' is not found in the list of unique strings: '{unique_strings}'."
                        )
                    rc = [it for it, x in enumerate(unique_strings) if x == candidate][0]
                    proportion_info[2][proportion_compartment_index] = rc
                    proportion_compartment_index += 1

            assert proportion_compartment_index == current_proportion_sum_it

            proportion_array = np.zeros((current_proportion_sum_start), dtype="int64")

            proportion_index = 0
            for it, elem in enumerate(self.transitions["proportional_to"]):
                for it2, elem2 in enumerate(elem):
                    elem_tmp = [
                        w
                        for w in pd.DataFrame(index=pd.MultiIndex.from_product(elem2))
                        .reset_index()
                        .apply(lambda z: reduce(lambda x, y: f"{x}_{y}", z), axis=1)
                    ]

                    for it3, elem3 in enumerate(elem_tmp):
                        rc = -1
                        for compartment in range(self.compartments.shape[0]):
                            if self.compartments["name"][compartment] == elem3:
                                rc = compartment
                        if rc == -1:
                            raise ValueError(
                                f"Could not find `proportional_to` '{elem3}' in compartments. "
                                f"Available compartments: '{self.compartments}'."
                            )

                        proportion_array[proportion_index] = rc
                        proportion_index += 1

            ## This will need to be reworked to deal with the summing bit
            ## There will be changes needed in the steps_source too
            ## They are doable though
            # for it, elem in enumerate(self.transitions['proportional_to']):
            #     elem = [y for y in map(
            #         lambda x: reduce(
            #             lambda a, b: str(a) + "_" + str(b),
            #             map(
            #                 lambda x: reduce(
            #                     lambda a, b: str(a) + "+" + str(b),
            #                     as_list(x)
            #                 ),
            #                 x
            #             )
            #         ),
            #         elem
            #     )]
            #     for it2, elem2 in enumerate(elem):
            #         rc = -1
            #         for compartment in range(self.compartments.shape[0]):
            #             if self.compartments["name"][compartment] == elem2:
            #                 rc = compartment
            #         proportion_array[it]

        return (
            unique_strings,
            transition_array,
            proportion_array,
            proportion_info,
        )

    def parse_parameters(
        self, parameters: np.ndarray, parameter_names: list, unique_strings: list
    ) -> np.ndarray:
        """
        Parses provided parameters and stores them in NumPy arrays.

        Args:
            parameters: Input parameters to be parsed.
            parameter_names: List of all parameter names.
            unique_strings:
                List of unique values from `proportion_exponent`
                and `rate` columns of transitions data.

        Returns:
            Array of parsed parameters.
        """
        # parsed_parameters_old = self.parse_parameter_strings_to_numpy_arrays(parameters, parameter_names, unique_strings)
        parsed_parameters = self.parse_parameter_strings_to_numpy_arrays_v2(
            parameters, parameter_names, unique_strings
        )
        # for i in range(len(unique_strings)):
        #    print(unique_strings[i], (parsed_parameters[i]==parsed_parameters_old[i]).all())
        return parsed_parameters

    def parse_parameter_strings_to_numpy_arrays_v2(
        self, parameters, parameter_names, string_list
    ):
        # is using eval a better way ???
        import sympy as sp

        # Validate input lengths
        if len(parameters) != len(parameter_names):
            raise ValueError(
                "Number of parameter values does not match the number of parameter names."
            )

        # Define the symbols used in the formulas
        symbolic_parameters_namespace = {name: sp.symbols(name) for name in parameter_names}

        symbolic_parameters = [sp.symbols(name) for name in parameter_names]

        parsed_formulas = []
        for formula in string_list:
            try:
                # here it is very important to pass locals so that e.g if the  gamma parameter
                # is defined, it is not converted into the gamma scipy function
                f = sp.sympify(formula, locals=symbolic_parameters_namespace)
                parsed_formulas.append(f)
            except Exception as e:
                print(
                    f"Cannot parse formula '{formula}' from parameters: '{parameter_names}'."
                )
                raise (e)  # Print the error message for debugging

        # the list order needs to be right.
        parameter_values = {
            param: value for param, value in zip(symbolic_parameters, parameters)
        }
        parameter_values_list = [parameter_values[param] for param in symbolic_parameters]

        # Create a lambdify function for substitution
        substitution_function = sp.lambdify(symbolic_parameters, parsed_formulas)

        # Apply the lambdify function with parameter values as a list
        substituted_formulas = substitution_function(*parameter_values_list)
        for i in range(len(substituted_formulas)):
            # sometime it's "1" or "1*1*1*..." which produce an int or float instead of an array
            # in this case we find the next array and set it to that size,
            # TODO: instead of searching for the next array, better to just use the parameter shape.
            if not isinstance(substituted_formulas[i], np.ndarray):
                for k in range(len(substituted_formulas)):
                    if isinstance(substituted_formulas[k], np.ndarray):
                        substituted_formulas[i] = substituted_formulas[i] * np.ones_like(
                            substituted_formulas[k]
                        )

        return np.array(substituted_formulas)

    def parse_parameter_strings_to_numpy_arrays(
        self,
        parameters,
        parameter_names,
        string_list,
        operator_reduce_lambdas={
            "*": lambda a, b: a * b,
            "/": lambda a, b: a / b,
            "+": lambda a, b: a + b,
            "-": lambda a, b: a - b,
            "^": lambda a, b: a**b,
        },
        operators=["^", "*", "/", "+", "-"],
    ):
        if (
            not operators
        ):  # empty list means all have been tried. Usually there just remains one string in string_list at that time.
            raise ValueError(
                f"Could not parse string '{string_list}'. "
                f"This usually means that '{string_list[0]}' is a parameter name that is not defined "
                f"or that it contains an operator that is not in the list of supported operators: '{operators}'. "
                f"The defined parameters are '{parameter_names}'."
            )

        split_strings = [x.split(operators[0]) for x in string_list]
        rc_size = [len(string_list)]
        for x in parameters.shape[1:]:
            rc_size.append(x)
        rc = np.zeros(rc_size, dtype="float64")
        for sit, string in enumerate(split_strings):
            tmp_rc_size = [len(string)]
            for x in parameters.shape[1:]:
                tmp_rc_size.append(x)
            tmp_rc = np.zeros(tmp_rc_size, dtype="float64")
            is_numeric = [x.isnumeric() for x in string]
            is_parameter = [x in parameter_names for x in string]
            is_resolvable = [x[0] or x[1] for x in zip(is_numeric, is_parameter)]
            is_totally_resolvable = reduce(lambda a, b: a and b, is_resolvable)
            if not is_totally_resolvable:
                not_resolvable_indices = [it for it, x in enumerate(is_resolvable) if not x]

                tmp_rc[not_resolvable_indices] = (
                    self.parse_parameter_strings_to_numpy_arrays(
                        parameters,
                        parameter_names,
                        [string[not is_resolvable]],
                        operator_reduce_lambdas,
                        operators[1:],
                    )
                )
            for numeric_index in [x for x in range(len(is_numeric)) if is_numeric[x]]:
                tmp_rc[numeric_index] = parameters[0] * 0 + float(string[numeric_index])
            for parameter_index in [x for x in range(len(is_parameter)) if is_parameter[x]]:
                parameter_name_index = [
                    it
                    for it, x in enumerate(parameter_names)
                    if x == string[parameter_index]
                ]
                tmp_rc[parameter_index] = parameters[parameter_name_index]
            rc[sit] = reduce(operator_reduce_lambdas[operators[0]], tmp_rc)

        return rc

    def get_compartments_explicitDF(self) -> pd.DataFrame:
        """
        Returns a copy of the compartments information DataFrame.

        All columns receive a 'mc_' prefix.

        Returns:
            A copy of the compartments DataFrame.
        """
        df: pd.DataFrame = self.compartments.copy(
            deep=True
        )  # .melt(id_vars='name', var_name='meta_compartment', value_name='sub_compartment')
        # add prefix mc to all columns, even name
        rename_dict = {cn: f"mc_{cn}" for cn in df.columns}
        df = df.rename(columns=rename_dict)
        return df

    def plot(
        self, output_file="transition_graph", source_filters=[], destination_filters=[]
    ):
        import graphviz
        from functools import reduce, partial

        some_graph = self.parse_transitions(config["seir"])

        def filter_func(lst, this_filter=[]):
            for must in this_filter:
                if any(x in lst for x in must):
                    pass
                else:
                    return False
            return True

        some_graph = some_graph[
            [
                x
                for x in map(
                    partial(filter_func, this_filter=source_filters),
                    some_graph["source"],
                )
            ]
        ]
        some_graph = some_graph[
            [
                x
                for x in map(
                    partial(filter_func, this_filter=destination_filters),
                    some_graph["destination"],
                )
            ]
        ]

        graph_description = (
            "digraph {\n  overlap = false;"
            + reduce(
                lambda a, b: a + "\n" + b,
                some_graph.apply(
                    lambda x: f"""{reduce(lambda a,b : a + "_" + b, x["source"])} -> {reduce(lambda a,b: a + "_" + b, x["destination"])} [label="{reduce(lambda a,b: a + "*" + b, x["rate"])}"];""",
                    axis=1,
                ),
            )
            + "\n}"
        )

        src = graphviz.Source(graph_description)
        src.render(output_file)


def get_list_dimension(thing: Any) -> int:
    """
    Returns the dimension of a given object.

    Args:
        thing: Object whose dimension needs to be determined.

    Returns:
        Length of the object if a list, otherwise 1.
    """
    if type(thing) == list:
        return len(thing)
    return 1


def list_access_element_safe(
    thing: Any, idx: int, dimension=None, encapsulate_as_list=False
) -> Any | list[Any]:
    """
    Attempts to access an element from the given object `thing` at the specified index `idx`.

    Args:
        thing: Object to be accessed from.
        idx: Index of object you would like to access.
        dimension: Dimension or shape of the object.
        encapsulate_as_list: If `True`, the accessed element will be returned as a list. Default is False.

    Raises:
        Exception: If `thing` is not iterable or `idx ` is out of range.

    Returns:
        Item at `idx` if `thing` is list, or
        element itself if `thing` is not a list.
        Returned item will be a list if `encapsulate_as_list` is `True`.
    """
    try:
        return list_access_element(thing, idx, dimension, encapsulate_as_list)
    except Exception as e:
        raise Exception(
            f"Error {e}: "
            f"in list_access_element_safe for '{thing}' at index '{idx}'. "
            f"This is often, but not always because the object above is a list (there are brackets around it). "
            f"and in this case it is not broadcast, so if you want to it to be broadcasted, you need remove the brackets around it. "
            f"dimension: '{dimension}'."
        )


def list_access_element(
    thing: Any, idx: int, dimension=None, encapsulate_as_list=False
) -> Any | list[Any]:
    """
    Access an element from a list or return the input itself if not a list.
    If input `thing` is a list, the function will return the element at the specified index (`idx`).
    If input `thing` is not a list, the function will return the element itself, regardless of the
    `idx` value.

    Args:
        thing: Object to be accessed from.
        idx: Index of object you would like to access.
        dimension: Dimension or shape of the object.
        encapsulate_as_list: If `True`, the accessed element will be returned as a list. Default is False.

    Returns:
        Item at `idx` if `thing` is list, or
        element itself if `thing` is not a list.
        Returned item will be a list if `encapsulate_as_list` is `True`.
    """
    if not dimension is None:
        if dimension == 1:
            rc = as_list(thing)
        if dimension != 1:
            rc = as_list(list_access_element(thing, idx, None))
    if type(thing) == list:
        rc = thing[idx]
    else:
        rc = thing
    if encapsulate_as_list:
        return as_list(rc)
    else:
        return rc


def list_recursive_convert_to_string(thing: Any) -> str:
    """
    Return given object as a str,
    or recursively convert elements of given list to strs.

    Args:
        thing: Object to be converted to a str or series of strs.

    Returns:
        Value(s) in `thing` as str(s).
    """
    if type(thing) == list:
        return [list_recursive_convert_to_string(x) for x in thing]
    return str(thing)


@cli.group()
@pass_context
def compartments(ctx: Context):
    """
    Add commands for working with FlepiMoP compartments.
    """
    pass


@compartments.command(params=[config_files_argument] + list(config_file_options.values()))
@pass_context
def plot(ctx: Context, **kwargs):
    """
    Command to generate a plot representing transitions between compartments.
    """
    parse_config_files(config, ctx, **kwargs)
    assert config["compartments"].exists()
    assert config["seir"].exists()
    comp = Compartments(
        seir_config=config["seir"], compartments_config=config["compartments"]
    )

    # TODO: this should be a command like build compartments.
    (
        unique_strings,
        transition_array,
        proportion_array,
        proportion_info,
    ) = comp.get_transition_array()

    comp.plot(output_file="transition_graph", source_filters=[], destination_filters=[])


@compartments.command(params=[config_files_argument] + list(config_file_options.values()))
@pass_context
def export(ctx: Context, **kwargs):
    """
    Export compartment information to a CSV file.
    """
    parse_config_files(config, ctx, **kwargs)
    assert config["compartments"].exists()
    assert config["seir"].exists()
    comp = Compartments(
        seir_config=config["seir"], compartments_config=config["compartments"]
    )
    (
        unique_strings,
        transition_array,
        proportion_array,
        proportion_info,
    ) = comp.get_transition_array()
    comp.toFile("compartments_file.csv", "transitions_file.csv", write_parquet=False)
    print("wrote files 'compartments_file.csv', 'transitions_file.csv' ")


cli.add_command(compartments)
