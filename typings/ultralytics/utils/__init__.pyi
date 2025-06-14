"""
This type stub file was generated by pyright.
"""

import contextlib
import importlib.metadata
import inspect
import json
import logging.config
import os
import platform
import re
import subprocess
import sys
import threading
import time
import uuid
import warnings
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
import yaml
from pathlib import Path
from threading import Lock
from types import SimpleNamespace
from typing import Union
from urllib.parse import unquote
from ultralytics import __version__
from tqdm import rich
from ultralytics.utils.patches import imread, imshow, imwrite, torch_load, torch_save

RANK = ...
LOCAL_RANK = ...
ARGV = ...
FILE = ...
ROOT = ...
ASSETS = ...
ASSETS_URL = ...
DEFAULT_CFG_PATH = ...
DEFAULT_SOL_CFG_PATH = ...
NUM_THREADS = ...
AUTOINSTALL = ...
VERBOSE = ...
TQDM_BAR_FORMAT = ...
LOGGING_NAME = ...
ARM64 = ...
PYTHON_VERSION = ...
TORCH_VERSION = ...
TORCHVISION_VERSION = ...
IS_VSCODE = ...
RKNN_CHIPS = ...
HELP_MSG = ...
if TQDM_RICH := str(os.getenv("YOLO_TQDM_RICH", False)).lower() == "true":
    ...
class TQDM(rich.tqdm if TQDM_RICH else tqdm.tqdm):
    """
    A custom TQDM progress bar class that extends the original tqdm functionality.

    This class modifies the behavior of the original tqdm progress bar based on global settings and provides
    additional customization options.

    Attributes:
        disable (bool): Whether to disable the progress bar. Determined by the global VERBOSE setting and
            any passed 'disable' argument.
        bar_format (str): The format string for the progress bar. Uses the global TQDM_BAR_FORMAT if not
            explicitly set.

    Methods:
        __init__: Initializes the TQDM object with custom settings.

    Examples:
        >>> from ultralytics.utils import TQDM
        >>> for i in TQDM(range(100)):
        ...     # Your processing code here
        ...     pass
    """
    def __init__(self, *args, **kwargs) -> None:
        """
        Initializes a custom TQDM progress bar.

        This class extends the original tqdm class to provide customized behavior for Ultralytics projects.

        Args:
            *args (Any): Variable length argument list to be passed to the original tqdm constructor.
            **kwargs (Any): Arbitrary keyword arguments to be passed to the original tqdm constructor.

        Notes:
            - The progress bar is disabled if VERBOSE is False or if 'disable' is explicitly set to True in kwargs.
            - The default bar format is set to TQDM_BAR_FORMAT unless overridden in kwargs.

        Examples:
            >>> from ultralytics.utils import TQDM
            >>> for i in TQDM(range(100)):
            ...     # Your code here
            ...     pass
        """
        ...
    


class SimpleClass:
    """
    A simple base class for creating objects with string representations of their attributes.

    This class provides a foundation for creating objects that can be easily printed or represented as strings,
    showing all their non-callable attributes. It's useful for debugging and introspection of object states.

    Methods:
        __str__: Returns a human-readable string representation of the object.
        __repr__: Returns a machine-readable string representation of the object.
        __getattr__: Provides a custom attribute access error message with helpful information.

    Examples:
        >>> class MyClass(SimpleClass):
        ...     def __init__(self):
        ...         self.x = 10
        ...         self.y = "hello"
        >>> obj = MyClass()
        >>> print(obj)
        __main__.MyClass object with attributes:

        x: 10
        y: 'hello'

    Notes:
        - This class is designed to be subclassed. It provides a convenient way to inspect object attributes.
        - The string representation includes the module and class name of the object.
        - Callable attributes and attributes starting with an underscore are excluded from the string representation.
    """
    def __str__(self) -> str:
        """Return a human-readable string representation of the object."""
        ...
    
    def __repr__(self): # -> str:
        """Return a machine-readable string representation of the object."""
        ...
    
    def __getattr__(self, attr):
        """Custom attribute access error message with helpful information."""
        ...
    


class IterableSimpleNamespace(SimpleNamespace):
    """
    An iterable SimpleNamespace class that provides enhanced functionality for attribute access and iteration.

    This class extends the SimpleNamespace class with additional methods for iteration, string representation,
    and attribute access. It is designed to be used as a convenient container for storing and accessing
    configuration parameters.

    Methods:
        __iter__: Returns an iterator of key-value pairs from the namespace's attributes.
        __str__: Returns a human-readable string representation of the object.
        __getattr__: Provides a custom attribute access error message with helpful information.
        get: Retrieves the value of a specified key, or a default value if the key doesn't exist.

    Examples:
        >>> cfg = IterableSimpleNamespace(a=1, b=2, c=3)
        >>> for k, v in cfg:
        ...     print(f"{k}: {v}")
        a: 1
        b: 2
        c: 3
        >>> print(cfg)
        a=1
        b=2
        c=3
        >>> cfg.get("b")
        2
        >>> cfg.get("d", "default")
        'default'

    Notes:
        This class is particularly useful for storing configuration parameters in a more accessible
        and iterable format compared to a standard dictionary.
    """
    def __iter__(self): # -> Iterator[tuple[str, Any]]:
        """Return an iterator of key-value pairs from the namespace's attributes."""
        ...
    
    def __str__(self) -> str:
        """Return a human-readable string representation of the object."""
        ...
    
    def __getattr__(self, attr):
        """Custom attribute access error message with helpful information."""
        ...
    
    def get(self, key, default=...): # -> Any | None:
        """Return the value of the specified key if it exists; otherwise, return the default value."""
        ...
    


def plt_settings(rcparams=..., backend=...): # -> Callable[..., Callable[..., Any]]:
    """
    Decorator to temporarily set rc parameters and the backend for a plotting function.

    Example:
        decorator: @plt_settings({"font.size": 12})
        context manager: with plt_settings({"font.size": 12}):

    Args:
        rcparams (dict): Dictionary of rc parameters to set.
        backend (str, optional): Name of the backend to use. Defaults to 'Agg'.

    Returns:
        (Callable): Decorated function with temporarily set rc parameters and backend. This decorator can be
            applied to any function that needs to have specific matplotlib rc parameters and backend for its execution.
    """
    ...

def set_logging(name=..., verbose=...): # -> Logger:
    """
    Sets up logging with UTF-8 encoding and configurable verbosity.

    This function configures logging for the Ultralytics library, setting the appropriate logging level and
    formatter based on the verbosity flag and the current process rank. It handles special cases for Windows
    environments where UTF-8 encoding might not be the default.

    Args:
        name (str): Name of the logger. Defaults to "LOGGING_NAME".
        verbose (bool): Flag to set logging level to INFO if True, ERROR otherwise. Defaults to True.

    Examples:
        >>> set_logging(name="ultralytics", verbose=True)
        >>> logger = logging.getLogger("ultralytics")
        >>> logger.info("This is an info message")

    Notes:
        - On Windows, this function attempts to reconfigure stdout to use UTF-8 encoding if possible.
        - If reconfiguration is not possible, it falls back to a custom formatter that handles non-UTF-8 environments.
        - The function sets up a StreamHandler with the appropriate formatter and level.
        - The logger's propagate flag is set to False to prevent duplicate logging in parent loggers.
    """
    ...

LOGGER = ...
def emojis(string=...): # -> str:
    """Return platform-dependent emoji-safe version of string."""
    ...

class ThreadingLocked:
    """
    A decorator class for ensuring thread-safe execution of a function or method. This class can be used as a decorator
    to make sure that if the decorated function is called from multiple threads, only one thread at a time will be able
    to execute the function.

    Attributes:
        lock (threading.Lock): A lock object used to manage access to the decorated function.

    Example:
        ```python
        from ultralytics.utils import ThreadingLocked

        @ThreadingLocked()
        def my_function():
            # Your code here
        ```
    """
    def __init__(self) -> None:
        """Initializes the decorator class for thread-safe execution of a function or method."""
        ...
    
    def __call__(self, f): # -> _Wrapped[Callable[..., Any], Any, Callable[..., Any], Any]:
        """Run thread-safe execution of function or method."""
        ...
    


def yaml_save(file=..., data=..., header=...): # -> None:
    """
    Save YAML data to a file.

    Args:
        file (str, optional): File name. Default is 'data.yaml'.
        data (dict): Data to save in YAML format.
        header (str, optional): YAML header to add.

    Returns:
        (None): Data is saved to the specified file.
    """
    ...

def yaml_load(file=..., append_filename=...): # -> Any | dict[Any, Any]:
    """
    Load YAML data from a file.

    Args:
        file (str, optional): File name. Default is 'data.yaml'.
        append_filename (bool): Add the YAML filename to the YAML dictionary. Default is False.

    Returns:
        (dict): YAML data and file name.
    """
    ...

def yaml_print(yaml_file: Union[str, Path, dict]) -> None:
    """
    Pretty prints a YAML file or a YAML-formatted dictionary.

    Args:
        yaml_file: The file path of the YAML file or a YAML-formatted dictionary.

    Returns:
        (None)
    """
    ...

DEFAULT_CFG_DICT = ...
DEFAULT_SOL_DICT = ...
DEFAULT_CFG_KEYS = ...
DEFAULT_CFG = ...
def read_device_model() -> str:
    """
    Reads the device model information from the system and caches it for quick access. Used by is_jetson() and
    is_raspberrypi().

    Returns:
        (str): Kernel release information.
    """
    ...

def is_ubuntu() -> bool:
    """
    Check if the OS is Ubuntu.

    Returns:
        (bool): True if OS is Ubuntu, False otherwise.
    """
    ...

def is_colab(): # -> bool:
    """
    Check if the current script is running inside a Google Colab notebook.

    Returns:
        (bool): True if running inside a Colab notebook, False otherwise.
    """
    ...

def is_kaggle(): # -> bool:
    """
    Check if the current script is running inside a Kaggle kernel.

    Returns:
        (bool): True if running inside a Kaggle kernel, False otherwise.
    """
    ...

def is_jupyter(): # -> bool:
    """
    Check if the current script is running inside a Jupyter Notebook.

    Returns:
        (bool): True if running inside a Jupyter Notebook, False otherwise.

    Note:
        - Only works on Colab and Kaggle, other environments like Jupyterlab and Paperspace are not reliably detectable.
        - "get_ipython" in globals() method suffers false positives when IPython package installed manually.
    """
    ...

def is_runpod(): # -> bool:
    """
    Check if the current script is running inside a RunPod container.

    Returns:
        (bool): True if running in RunPod, False otherwise.
    """
    ...

def is_docker() -> bool:
    """
    Determine if the script is running inside a Docker container.

    Returns:
        (bool): True if the script is running inside a Docker container, False otherwise.
    """
    ...

def is_raspberrypi() -> bool:
    """
    Determines if the Python environment is running on a Raspberry Pi by checking the device model information.

    Returns:
        (bool): True if running on a Raspberry Pi, False otherwise.
    """
    ...

def is_jetson() -> bool:
    """
    Determines if the Python environment is running on an NVIDIA Jetson device by checking the device model information.

    Returns:
        (bool): True if running on an NVIDIA Jetson device, False otherwise.
    """
    ...

def is_online() -> bool:
    """
    Check internet connectivity by attempting to connect to a known online host.

    Returns:
        (bool): True if connection is successful, False otherwise.
    """
    ...

def is_pip_package(filepath: str = ...) -> bool:
    """
    Determines if the file at the given filepath is part of a pip package.

    Args:
        filepath (str): The filepath to check.

    Returns:
        (bool): True if the file is part of a pip package, False otherwise.
    """
    ...

def is_dir_writeable(dir_path: Union[str, Path]) -> bool:
    """
    Check if a directory is writeable.

    Args:
        dir_path (str | Path): The path to the directory.

    Returns:
        (bool): True if the directory is writeable, False otherwise.
    """
    ...

def is_pytest_running(): # -> bool:
    """
    Determines whether pytest is currently running or not.

    Returns:
        (bool): True if pytest is running, False otherwise.
    """
    ...

def is_github_action_running() -> bool:
    """
    Determine if the current environment is a GitHub Actions runner.

    Returns:
        (bool): True if the current environment is a GitHub Actions runner, False otherwise.
    """
    ...

def get_git_dir(): # -> Path | None:
    """
    Determines whether the current file is part of a git repository and if so, returns the repository root directory. If
    the current file is not part of a git repository, returns None.

    Returns:
        (Path | None): Git root directory if found or None if not found.
    """
    ...

def is_git_dir(): # -> bool:
    """
    Determines whether the current file is part of a git repository. If the current file is not part of a git
    repository, returns None.

    Returns:
        (bool): True if current file is part of a git repository.
    """
    ...

def get_git_origin_url(): # -> str | None:
    """
    Retrieves the origin URL of a git repository.

    Returns:
        (str | None): The origin URL of the git repository or None if not git directory.
    """
    ...

def get_git_branch(): # -> str | None:
    """
    Returns the current git branch name. If not in a git repository, returns None.

    Returns:
        (str | None): The current git branch name or None if not a git directory.
    """
    ...

def get_default_args(func): # -> dict[str, Any]:
    """
    Returns a dictionary of default arguments for a function.

    Args:
        func (callable): The function to inspect.

    Returns:
        (dict): A dictionary where each key is a parameter name, and each value is the default value of that parameter.
    """
    ...

def get_ubuntu_version(): # -> str | Any | None:
    """
    Retrieve the Ubuntu version if the OS is Ubuntu.

    Returns:
        (str): Ubuntu version or None if not an Ubuntu OS.
    """
    ...

def get_user_config_dir(sub_dir=...): # -> Path:
    """
    Return the appropriate config directory based on the environment operating system.

    Args:
        sub_dir (str): The name of the subdirectory to create.

    Returns:
        (Path): The path to the user config directory.
    """
    ...

DEVICE_MODEL = ...
ONLINE = ...
IS_COLAB = ...
IS_KAGGLE = ...
IS_DOCKER = ...
IS_JETSON = ...
IS_JUPYTER = ...
IS_PIP_PACKAGE = ...
IS_RASPBERRYPI = ...
GIT_DIR = ...
IS_GIT_DIR = ...
USER_CONFIG_DIR = ...
SETTINGS_FILE = ...
def colorstr(*input): # -> str:
    r"""
    Colors a string based on the provided color and style arguments. Utilizes ANSI escape codes.
    See https://en.wikipedia.org/wiki/ANSI_escape_code for more details.

    This function can be called in two ways:
        - colorstr('color', 'style', 'your string')
        - colorstr('your string')

    In the second form, 'blue' and 'bold' will be applied by default.

    Args:
        *input (str | Path): A sequence of strings where the first n-1 strings are color and style arguments,
                      and the last string is the one to be colored.

    Supported Colors and Styles:
        Basic Colors: 'black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white'
        Bright Colors: 'bright_black', 'bright_red', 'bright_green', 'bright_yellow',
                       'bright_blue', 'bright_magenta', 'bright_cyan', 'bright_white'
        Misc: 'end', 'bold', 'underline'

    Returns:
        (str): The input string wrapped with ANSI escape codes for the specified color and style.

    Examples:
        >>> colorstr("blue", "bold", "hello world")
        >>> "\033[34m\033[1mhello world\033[0m"
    """
    ...

def remove_colorstr(input_string): # -> str:
    """
    Removes ANSI escape codes from a string, effectively un-coloring it.

    Args:
        input_string (str): The string to remove color and style from.

    Returns:
        (str): A new string with all ANSI escape codes removed.

    Examples:
        >>> remove_colorstr(colorstr("blue", "bold", "hello world"))
        >>> "hello world"
    """
    ...

class TryExcept(contextlib.ContextDecorator):
    """
    Ultralytics TryExcept class. Use as @TryExcept() decorator or 'with TryExcept():' context manager.

    Examples:
        As a decorator:
        >>> @TryExcept(msg="Error occurred in func", verbose=True)
        >>> def func():
        >>> # Function logic here
        >>>     pass

        As a context manager:
        >>> with TryExcept(msg="Error occurred in block", verbose=True):
        >>> # Code block here
        >>>     pass
    """
    def __init__(self, msg=..., verbose=...) -> None:
        """Initialize TryExcept class with optional message and verbosity settings."""
        ...
    
    def __enter__(self): # -> None:
        """Executes when entering TryExcept context, initializes instance."""
        ...
    
    def __exit__(self, exc_type, value, traceback): # -> Literal[True]:
        """Defines behavior when exiting a 'with' block, prints error message if necessary."""
        ...
    


class Retry(contextlib.ContextDecorator):
    """
    Retry class for function execution with exponential backoff.

    Can be used as a decorator to retry a function on exceptions, up to a specified number of times with an
    exponentially increasing delay between retries.

    Examples:
        Example usage as a decorator:
        >>> @Retry(times=3, delay=2)
        >>> def test_func():
        >>> # Replace with function logic that may raise exceptions
        >>>     return True
    """
    def __init__(self, times=..., delay=...) -> None:
        """Initialize Retry class with specified number of retries and delay."""
        ...
    
    def __call__(self, func): # -> Callable[..., Any | None]:
        """Decorator implementation for Retry with exponential backoff."""
        ...
    


def threaded(func): # -> Callable[..., Thread | Any]:
    """
    Multi-threads a target function by default and returns the thread or function result.

    Use as @threaded decorator. The function runs in a separate thread unless 'threaded=False' is passed.
    """
    ...

def set_sentry(): # -> None:
    """
    Initialize the Sentry SDK for error tracking and reporting. Only used if sentry_sdk package is installed and
    sync=True in settings. Run 'yolo settings' to see and update settings.

    Conditions required to send errors (ALL conditions must be met or no errors will be reported):
        - sentry_sdk package is installed
        - sync=True in YOLO settings
        - pytest is not running
        - running in a pip package installation
        - running in a non-git directory
        - running with rank -1 or 0
        - online environment
        - CLI used to run package (checked with 'yolo' as the name of the main CLI command)

    The function also configures Sentry SDK to ignore KeyboardInterrupt and FileNotFoundError exceptions and to exclude
    events with 'out of memory' in their exception message.

    Additionally, the function sets custom tags and user information for Sentry events.
    """
    ...

class JSONDict(dict):
    """
    A dictionary-like class that provides JSON persistence for its contents.

    This class extends the built-in dictionary to automatically save its contents to a JSON file whenever they are
    modified. It ensures thread-safe operations using a lock.

    Attributes:
        file_path (Path): The path to the JSON file used for persistence.
        lock (threading.Lock): A lock object to ensure thread-safe operations.

    Methods:
        _load: Loads the data from the JSON file into the dictionary.
        _save: Saves the current state of the dictionary to the JSON file.
        __setitem__: Stores a key-value pair and persists it to disk.
        __delitem__: Removes an item and updates the persistent storage.
        update: Updates the dictionary and persists changes.
        clear: Clears all entries and updates the persistent storage.

    Examples:
        >>> json_dict = JSONDict("data.json")
        >>> json_dict["key"] = "value"
        >>> print(json_dict["key"])
        value
        >>> del json_dict["key"]
        >>> json_dict.update({"new_key": "new_value"})
        >>> json_dict.clear()
    """
    def __init__(self, file_path: Union[str, Path] = ...) -> None:
        """Initialize a JSONDict object with a specified file path for JSON persistence."""
        ...
    
    def __setitem__(self, key, value): # -> None:
        """Store a key-value pair and persist to disk."""
        ...
    
    def __delitem__(self, key): # -> None:
        """Remove an item and update the persistent storage."""
        ...
    
    def __str__(self) -> str:
        """Return a pretty-printed JSON string representation of the dictionary."""
        ...
    
    def update(self, *args, **kwargs): # -> None:
        """Update the dictionary and persist changes."""
        ...
    
    def clear(self): # -> None:
        """Clear all entries and update the persistent storage."""
        ...
    


class SettingsManager(JSONDict):
    """
    SettingsManager class for managing and persisting Ultralytics settings.

    This class extends JSONDict to provide JSON persistence for settings, ensuring thread-safe operations and default
    values. It validates settings on initialization and provides methods to update or reset settings.

    Attributes:
        file (Path): The path to the JSON file used for persistence.
        version (str): The version of the settings schema.
        defaults (Dict): A dictionary containing default settings.
        help_msg (str): A help message for users on how to view and update settings.

    Methods:
        _validate_settings: Validates the current settings and resets if necessary.
        update: Updates settings, validating keys and types.
        reset: Resets the settings to default and saves them.

    Examples:
        Initialize and update settings:
        >>> settings = SettingsManager()
        >>> settings.update(runs_dir="/new/runs/dir")
        >>> print(settings["runs_dir"])
        /new/runs/dir
    """
    def __init__(self, file=..., version=...) -> None:
        """Initializes the SettingsManager with default settings and loads user settings."""
        ...
    
    def __setitem__(self, key, value): # -> None:
        """Updates one key: value pair."""
        ...
    
    def update(self, *args, **kwargs): # -> None:
        """Updates settings, validating keys and types."""
        ...
    
    def reset(self): # -> None:
        """Resets the settings to default and saves them."""
        ...
    


def deprecation_warn(arg, new_arg=...): # -> None:
    """Issue a deprecation warning when a deprecated argument is used, suggesting an updated argument."""
    ...

def clean_url(url): # -> str:
    """Strip auth from URL, i.e. https://url.com/file.txt?auth -> https://url.com/file.txt."""
    ...

def url2file(url): # -> str:
    """Convert URL to filename, i.e. https://url.com/file.txt?auth -> file.txt."""
    ...

def vscode_msg(ext=...) -> str:
    """Display a message to install Ultralytics-Snippets for VS Code if not already installed."""
    ...

PREFIX = ...
SETTINGS = ...
PERSISTENT_CACHE = ...
DATASETS_DIR = ...
WEIGHTS_DIR = ...
RUNS_DIR = ...
ENVIRONMENT = ...
TESTS_RUNNING = ...
if WINDOWS:
    ...
