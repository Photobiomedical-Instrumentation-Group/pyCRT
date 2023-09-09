"""
Functions related to processing TOML files containing camera settings or pyCRT
configurations.
"""

from typing import Union, Any

import tomli

# Type aliases for commonly used types
# {{{
# Dictionary created from a TOML configuration file
TOMLDict = dict[str, Union[list, int, dict[str, Any]]]
# }}}


def loadTOML(filePath: str) -> TOMLDict:
    # {{{
    """
    Reads a TOML config file and returns its contents as a dictionary.

    Parameters
    ----------
    filePath : str
        The path for the TOML file (be it a camera settings specification file
        or a pyCRT configuration file).

    Returns
    -------
    tomlDict : dict of str keys and list, int or dict values
        The dictionary containing with the TOML file contents.

    """

    with open(filePath, mode="rb") as arq:
        tomlDict = tomli.load(arq)

    return tomlDict


# }}}
