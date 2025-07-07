"""
Tests for section 2.3 of the AI Security course.
"""

import pickle
import torch
import torch.nn.functional as F
import os
import pytest
import numpy as np
import sys
import xlab
from typing import List, Callable, Any

# Global variables to store test parameters passed from notebook
_test_config = {
    'model': None,
    'student_function': None,
    'loss_fn': None,
    'label': None,
}


def demo_l_inf_dist(epsilon, h, w, c):
    """Calculates the delta update for l_inf distribution.
    
    Args:
        epsilon: Small number used for perturbation
        h: Dimension of square
        w: Image dimension (assumes image tensor is square)
        c: Number of colour channels (RGB is 3)
    Returns:
        delta: Tensor of the same size as input, containing updates for each channel.
    """
    delta = torch.zeros(c, w, w)
    r,s = np.random.randint(w-h, size = (2))
    for channel in range(c):
        unif = np.random.uniform(-2*epsilon, 2*epsilon)
        delta[channel][r:r+h, s:s+h] = unif
    return delta
    return None


class TestTask1:
    """Tests for the l_inf_dist function."""

    def test_output_shape_and_type(self):
        """Tests that the output tensor has the correct shape and data type."""
        dist_func = _test_config['student_function']

        # Test parameters
        epsilon, h, w, c = 0.1, 8, 32, 3

        # Execution
        result_tensor = dist_func(epsilon, h, w, c)

        # Assertions
        assert isinstance(result_tensor, torch.Tensor)
        assert result_tensor.shape == (c, w, w)
        assert result_tensor.dtype == torch.float32  # torch.zeros defaults to float32

    def test_values_in_range(self):
        """Tests that all modified values are within the [-2*epsilon, 2*epsilon] range."""
        dist_func = _test_config['student_function']

        # Test parameters
        epsilon, h, w, c = 0.05, 5, 32, 1

        # Execution
        result_tensor = dist_func(epsilon, h, w, c)

        # Find non-zero elements to check their values
        non_zero_elements = result_tensor[result_tensor != 0]

        # Assertions
        assert torch.all(non_zero_elements >= -2 * epsilon)
        assert torch.all(non_zero_elements <= 2 * epsilon)
        # Ensure the correct number of elements were modified
        assert non_zero_elements.numel() == h * h

    def test_patch_structure(self):
        """Tests that a single h x h patch is modified consistently across channels."""
        dist_func = _test_config['student_function']

        # Test parameters
        epsilon, h, w, c = 0.1, 4, 32, 3

        # Seed for reproducibility to determine the patch location ahead of time
        np.random.seed(42)
        expected_r, expected_s = np.random.randint(32 - h, size=(2))

        # Reset seed and execute the function
        np.random.seed(42)
        result_tensor = dist_func(epsilon, h, w, c)

        # Create a boolean mask for the expected patch location
        mask = torch.zeros_like(result_tensor, dtype=torch.bool)
        mask[:, expected_r:expected_r + h, expected_s:expected_s + h] = True

        # Assertions
        # 1. Values outside the patch must be zero.
        assert torch.all(result_tensor[~mask] == 0)

        # 2. Values inside the patch for a given channel must all be the same.
        for i in range(c):
            patch = result_tensor[i, expected_r:expected_r + h, expected_s:expected_s + h]
            assert torch.all(patch == patch[0, 0])

        # 3. Values for different channels should be different (highly probable).
        patch_vals = [result_tensor[i, expected_r, expected_s].item() for i in range(c)]
        assert len(set(patch_vals)) == c

    def compare_to_solution(self):
        """Tests that the output tensor has the correct shape and data type."""
        dist_func = _test_config['student_function']

        # Test parameters
        epsilon, h, w, c = 0.1, 8, 32, 3

        # Execution
        result_tensor = dist_func(epsilon, h, w, c)
        ans_tensor = demo_l_inf_dist(epsilon, h, w, c)

        # Assertions
        assert torch.equal(result_tensor, ans_tensor)


    def test_reproducibility_with_seed(self):
        """Tests that seeding numpy's RNG makes the function's output reproducible."""
        dist_func = _test_config['student_function']

        # Test parameters
        epsilon, h, w, c = 0.1, 8, 32, 3
        seed = 123

        # Execution - First run
        np.random.seed(seed)
        result1 = dist_func(epsilon, h, w, c)

        # Execution - Second run with the same seed
        np.random.seed(seed)
        result2 = dist_func(epsilon, h, w, c)

        # Execution - Third run with a different seed
        np.random.seed(seed + 1)
        result3 = dist_func(epsilon, h, w, c)

        # Assertions
        assert torch.equal(result1, result2)
        assert not torch.equal(result1, result3)
            

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
    
    print(f"\nRunning tests for Section 2.4.2, {task_name}...")
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

# Notebook interface functions (maintaining backward compatibility)
def task1(student_function):
    """
    Run Task 1 tests using pytest.
    
    Args:
        student_function: The function to test with.
    
    Returns:
        dict: A summary dictionary with test results.
    """
    # Configure global test parameters
    _test_config['student_function'] = student_function
    
    # Run pytest tests
    result = _run_pytest_with_capture(TestTask1)
    _print_test_summary(result, "Task 1")
    
    return result


def task2(student_function):
    """
    Run Task 2 tests using pytest.
    
    Args:
        student_function: The function to test with.
    
    Returns:
        dict: A summary dictionary with test results.
    """
    # Configure global test parameters
    _test_config['student_function'] = student_function
    
    # Run pytest tests
    result = _run_pytest_with_capture(TestTask2)
    _print_test_summary(result, "Task 2")
    
    return result

def task3(student_function):
    """
    Run Task 3 tests using pytest.
    
    Args:
        student_function: The function to test with.
    
    Returns:
        dict: A summary dictionary with test results.
    """
    # Configure global test parameters
    _test_config['student_function'] = student_function
    
    # Run pytest tests
    result = _run_pytest_with_capture(TestTask3)
    _print_test_summary(result, "Task 3")
    
    return result

def task4(student_function):
    """
    Run Task 4 tests using pytest.
    
    Args:
        student_function: The function to test with.
    
    Returns:
        dict: A summary dictionary with test results.
    """
    # Configure global test parameters
    _test_config['student_function'] = student_function
    
    # Run pytest tests
    result = _run_pytest_with_capture(TestTask4)
    _print_test_summary(result, "Task 4")
    
    return result