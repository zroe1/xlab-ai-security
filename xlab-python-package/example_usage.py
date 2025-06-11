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

# --- Example of using the student-facing tests ---
print("--- Testing with xlab.tests ---")

# Import the tests module
from xlab import tests

def student_code_fail(input_str):
    """A sample student function that fails the test."""
    return "some other output"

print("Running test with a function that should fail:")
result_fail = tests.section1_0.task1(student_code_fail)
print(f"Result: {result_fail}")
print()
# -----------------------------------------

print("Example completed successfully!") 