"""
A script to introspect gempyor's click-based CLI.

Extracts help information for all available flepiMoP commands, subcommands, and options and formats
the information into a markdown file in the gitbook documentation. 

Functions:
    record_click_help: recursively retrieves help info and stores in a data structure
    output_help_table: converts help info into markdown tables
    append_md_file: overwrites markdown file after a given point with help info tables
    get_dict_depth: determines the nested depth of dictionaries for proper handling in output_help_table()
"""

import os
import sys

import click
from tabulate import tabulate

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from gempyor.shared_cli import cli


def record_click_help(click_object: click.Command | click.Group, command_info: dict) -> dict:
    """
    Creates a hierarchical dictionary containing subcommand and option information for a click.Command or a click.Group.

    Args:
        click_object: The click object containing the click.Command/click.Group
        command_info: An empty dictionary to populate with help info.

    Returns: 
        A dictionary containing help info for all subcommands and options of that particularly click.Command or click.Group.
    """

    if isinstance(click_object, (click.Command, click.Group)):
        command_info[click_object.name] = {}
    else:
        raise TypeError(f"Error, click_object must be either click.Group or click.Command: {type(click_object)}")
    
    if isinstance(click_object, click.Group):
        for _, subcommand in click_object.commands.items():
            subcommand_dict = record_click_help(subcommand, {})
            command_info[click_object.name][subcommand.name] = subcommand_dict[subcommand.name]
    elif isinstance(click_object, click.Command):
        for param in click_object.params:
            if isinstance(param, click.Argument):
                continue  # skip click.Arguments for now 
            command_info[click_object.name][param.name] = {
                "option": param.opts,
                "type": param.type,
                "description": param.help,
                "default": param.default
            } 
   
    return command_info


def output_help_table(help_dict: dict) -> dict[str: str]:  
    """
    Populates a markdown table with a click object's help information.
    
    Args:
        help_dict: A hierarchical dictionary populated with help information for a click object.
        
    Returns:
        A dictionary of strs containing click object help info (formatted as a markdown table) for each subcommand. 
    """

    return_dict = {}
    depth = get_dict_depth(help_dict)

    if depth == 4:
        for command in help_dict.keys():
            for subcommand in help_dict[command].keys():
                data_structure = [["name", "option", "type", "description", "default"]]
                for option in help_dict[command][subcommand].keys():
                    data_structure.append([
                        option, 
                        help_dict[command][subcommand][option]['option'], 
                        help_dict[command][subcommand][option]['type'], 
                        help_dict[command][subcommand][option]['description'], 
                        help_dict[command][subcommand][option]['default']
                        ])
                markdown_table = tabulate(data_structure, headers="firstrow", tablefmt="pipe")
                return_dict[subcommand] = markdown_table
                

    elif depth == 3:
        for command in help_dict.keys():
            data_structure = [["name", "option", "type", "description", "default"]]
            for option in help_dict[command].keys():
                data_structure.append([
                        option, 
                        help_dict[command][option]['option'], 
                        help_dict[command][option]['type'], 
                        help_dict[command][option]['description'], 
                        help_dict[command][option]['default']
                    ])
            markdown_table = tabulate(data_structure, headers="firstrow", tablefmt="pipe")
            return_dict[command] = markdown_table

    else:
        raise Exception(f"help_dict has depth of {depth}, meaning there is more than one layer of subcommands (no framework to support)") # expand to make more dynamic 
    
        
    return return_dict


def append_md_file(help_dict: dict[str: list[dict, dict]]) -> None:
    """
    Put help info tables into the correct place in the documentation markdown file.

    Args:
        help_dict: A dictionary containing a markdown table(s) for each command/subcommand. 

    Returns:
        None
    """

    file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../documentation/gitbook/gempyor/model-implementation/flepimop-click-commands.md')) 

    with open(file_path, "r") as file:
        lines = file.readlines()
    # Find the line where the replacement should begin
    start_point = "to see CLI help in terminal."  # change if the last line of the file changes (otherwise an error will get thrown)
    start_index = None
    for index, line in enumerate(lines):
        if start_point in line:
            start_index = index
            break

    if start_index is not None:
        new_file_content = lines[:start_index + 1]  # Keep content before start point, append after
        for command in help_dict:
            new_content = f"\n\n## {command}:"
            for table in help_dict[command][1]:
                new_content += f"\n\n### {table}\n{help_dict[command][1][table]}"
            new_file_content.append(new_content) 
        with open(file_path, "w") as file:
            file.writelines(new_file_content)
    else:
        print(f"Could not find the section '{start_point}' in the file.")


def get_dict_depth(d: dict) -> int:
    """
    Returns the depth of a python dict object.

    Arguments:
        d: a dict

    Returns:
        The number of levels deep `d` is.
    """

    if isinstance(d, dict) and d:
        return 1 + max(get_dict_depth(value) for value in d.values())
    
    return 0


def main():
    """
    The main execution function. 
    """

    flepimop_commands = {} # dict[str, list[str, dict]]
    for command in cli.commands:
        flepimop_commands[cli.commands[command].name] = [record_click_help(cli.commands[command], {})]
        
    for command in flepimop_commands: # populate data structure with markdown tables
        flepimop_commands[command].append(output_help_table(flepimop_commands[command][0]))

    append_md_file(flepimop_commands)



if __name__ == "__main__":
    main()


