"""
Tests for section 2.1.2 of the AI Security course.
"""

import pytest
import sys
from typing import Any

# Global variables to store test parameters passed from notebook
_test_config = {
    'array1': None,
    'array2': None,
    'array3': None,
    'target': None
}

# Task 4 Tests
class TestTask4:
    """Tests for Task 4: Array comparison with target value counting."""
    
    def test_arrays_have_equal_length(self):
        """Test that all arrays have equal length."""
        array1 = _test_config['array1']
        array2 = _test_config['array2']
        array3 = _test_config['array3']
        
        assert array1 is not None, "Array1 not configured for testing"
        assert array2 is not None, "Array2 not configured for testing"
        assert array3 is not None, "Array3 not configured for testing"
        
        len1, len2, len3 = len(array1), len(array2), len(array3)
        assert len1 == len2 == len3, f"Arrays must have equal length for valid comparison. " \
                                    f"Got lengths: {len1}, {len2}, {len3}"
    
    def test_third_array_has_most_occurrences(self):
        """Test that the third array has the most occurrences of the target value."""
        array1 = _test_config['array1']
        array2 = _test_config['array2']
        array3 = _test_config['array3']
        target = _test_config['target']
        
        assert array1 is not None, "Array1 not configured for testing"
        assert array2 is not None, "Array2 not configured for testing"
        assert array3 is not None, "Array3 not configured for testing"
        assert target is not None, "Target value not configured for testing"
        
        # Count occurrences in each array
        count1 = array1.count(target) if hasattr(array1, 'count') else sum(1 for x in array1 if x == target)
        count2 = array2.count(target) if hasattr(array2, 'count') else sum(1 for x in array2 if x == target)
        count3 = array3.count(target) if hasattr(array3, 'count') else sum(1 for x in array3 if x == target)
        
        assert count3 > count1 and count3 > count2, \
            f"Third array does not have the most occurrences of target {target}. " \
            f"Counts: Array1={count1}, Array2={count2}, Array3={count3}"

def _run_pytest_with_capture(test_class_or_function, verbose=True):
    """
    Run pytest on a specific test class or function and capture results.
    Runs pytest programmatically within the same process to preserve global state.
    
    Returns:
        dict: Summary of test results with total, passed, failed counts and score.
    """
    import io
    from contextlib import redirect_stdout, redirect_stderr
    
    # Get test class name
    if hasattr(test_class_or_function, '__name__'):
        test_name = test_class_or_function.__name__
    else:
        test_name = test_class_or_function
    
    # Get the current file path to specify which tests to run
    current_file = __file__
    
    # Capture stdout and stderr
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    
    try:
        # Run pytest programmatically within the same process
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            # Import pytest and run it programmatically
            exit_code = pytest.main([
                f'{current_file}::{test_name}',
                '-v' if verbose else '-q',
                '--tb=short',
                '--no-header'  # Cleaner output
            ])
        
        stdout_output = stdout_capture.getvalue()
        stderr_output = stderr_capture.getvalue()
        
        # Parse pytest output
        lines = stdout_output.split('\n')
        
        # Count passed and failed tests
        passed = 0
        failed = 0
        
        for line in lines:
            if '::' in line:
                if 'PASSED' in line:
                    passed += 1
                elif 'FAILED' in line or 'ERROR' in line:
                    failed += 1
        
        total_tests = passed + failed
        
        return {
            "total_tests": total_tests,
            "passed": passed,
            "failed": failed,
            "score": round((passed / total_tests) * 100) if total_tests > 0 else 0,
            "stdout": stdout_output,
            "stderr": stderr_output,
            "exit_code": exit_code
        }
    
    except Exception as e:
        return {
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "score": 0,
            "stdout": stdout_capture.getvalue(),
            "stderr": f"Error running pytest: {e}",
            "exit_code": 1
        }

def _print_test_summary(result_dict, task_name):
    """Print a formatted summary of test results."""
    # ANSI color codes
    GREEN = '\033[92m'
    RED = '\033[91m'
    RESET = '\033[0m'
    
    passed = result_dict["passed"]
    failed = result_dict["failed"]
    total = result_dict["total_tests"]
    
    print(f"\nRunning tests for Section 2.1.2, {task_name}...")
    print("=" * 70)
    
    if failed == 0:
        print(f"🎉 All tests passed! ({passed}/{total})")
    else:
        print(f"📊 Results: {passed} passed, {failed} failed out of {total} total")
    
    print("=" * 70)
    
    # Print pytest output for detailed feedback
    if result_dict.get("stdout"):
        print("\nDetailed output:")
        print(result_dict["stdout"])
    
    if result_dict.get("stderr") and result_dict["stderr"].strip():
        print(f"\n{RED}Errors:{RESET}")
        print(result_dict["stderr"])

# Notebook interface function (maintaining backward compatibility)
def task4(array1, array2, array3, target):
    """
    Run Task 4 tests using pytest.
    
    Tests that the third array has the most occurrences of the target value
    compared to the first two arrays.

    Args:
        array1 (list): First array to compare.
        array2 (list): Second array to compare.
        array3 (list): Third array to compare.
        target (int): The target value to count in each array.

    Returns:
        dict: A summary dictionary with the total number of tests, the
              number passed/failed, and a final score.
    """
    # Configure global test parameters
    _test_config['array1'] = array1
    _test_config['array2'] = array2
    _test_config['array3'] = array3
    _test_config['target'] = target
    
    # Run pytest tests
    result = _run_pytest_with_capture(TestTask4)
    _print_test_summary(result, "Task 4")
    
    return result
