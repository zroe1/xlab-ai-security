"""
Tests for section 1.0 of the AI Security course.
"""

import pytest

class TestTarget:
    func = None

target = TestTarget()

@pytest.fixture
def student_function():
    """A pytest fixture that provides the student's function to any test."""
    if target.func is None:
        pytest.skip("Student function not provided.")
    return target.func

@pytest.mark.task1
def test_function_runs_without_crashing(student_function):
    """Test that the function can be called without raising an exception."""
    try:
        student_function("test input")
    except Exception as e:
        pytest.fail(f"Function crashed with error: {e}")

@pytest.mark.task1
@pytest.mark.parametrize("input_text,expected_output,description", [
    ("Hello there gpt-2", ['Hello', ' there', ' g', 'pt', '-', '2'], "basic text with hyphen"),
    ("??!hello--*- world#$", ['??', '!', 'hello', '--', '*', '-', ' world', '#$'], "special characters and symbols"),
    ("https://xrisk.uchicago.edu/fellowship/", ['https', '://', 'x', 'risk', '.', 'uch', 'icago', '.', 'edu', '/', 'fell', 'owship', '/'], "URL with dots and slashes"),
    ("", [], "empty string"),
    (".,.,.,.,.,.,.,", ['.,', '.,', '.,', '.,', '.,', '.,', '.,'], "repeated punctuation"),
])
def test_tokenization_cases(student_function, input_text, expected_output, description):
    """Test tokenization function with various input cases."""
    result = student_function(input_text)
    assert result == expected_output, f"Test case '{description}' failed. Expected: {expected_output}, Got: {result}"

def task1(student_func):
    """Runs all 'task1' tests against the provided student function."""
    target.func = student_func
    result_code = pytest.main([__file__, "-v", "--no-header", "-m", "task1", "-W", "ignore"])
    if result_code == pytest.ExitCode.OK:
        print("✅ All checks passed!")