#!/usr/bin/env python3
"""
Example usage of the xlab package.

This script demonstrates how to import and use the xlab package.
"""

# Import the package
import xlab

# Print package information
print(f"XLab Package Version: {xlab.__version__}")
print("Available functions:", xlab.__all__)
print()

# Use the hello_world function
print("Calling xlab.hello_world():")
message = xlab.hello_world()
print()

# You can also import specific functions
from xlab import hello_world

print("Calling hello_world() directly:")
hello_world()
print()

print("Example completed successfully!") 