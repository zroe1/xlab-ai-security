"""
Core functionality for the xlab package.
"""

from . import __version__


def hello_world():
    """
    Print a hello world message with the current package version.
    
    Returns:
        str: The hello world message with version information.
    """
    message = f"Hello world! You are using version {__version__} of the package"
    print(message)
    return message 