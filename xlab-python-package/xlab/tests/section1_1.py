"""
Tests for section 1.1 of the AI Security course.
"""

import pickle
import torch
import torch.nn.functional as F
import os
import pytest
import sys
import xlab
from typing import List, Callable, Any

# Global variables to store test parameters passed from notebook
_test_config = {
    'model': None,
    'student_function': None,
}

def get_100_examples():
    """Load test data for accuracy testing."""
    current_dir = os.path.dirname(__file__)
    data_path = os.path.join(current_dir, 'data', 'cifar10_data.pkl')
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
        x_test_loaded = data['x_test']
        y_test_loaded = data['y_test']

    return x_test_loaded, y_test_loaded

# Task 1 Tests
class TestTask1:
    """Tests for Task 1: Model accuracy evaluation."""
    
    def test_model_prediction(self):
        """Test that the model achieves >90% accuracy on test data."""
        model = _test_config['model']
        assert model is not None, "Model not configured for testing"
        
        x, y = get_100_examples()
        
        logits = model(x)
        predictions = torch.argmax(logits, axis=1)

        cnn = xlab.SimpleCNN
        cnn.eval()
        logits = cnn(x)
        cnn_pred = torch.argmax(logits, axis=1)
        
        assert predictions == logits
    

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
def task1(model):
    """
    Run Task 1 tests using pytest.
    
    Args:
        student_function (function): The student's function to test.
        model: The model to test with.
    
    Returns:
        dict: A summary dictionary with test results.
    """
    # Configure global test parameters
    _test_config['model'] = model
    
    # Run pytest tests
    result = _run_pytest_with_capture(TestTask1)
    _print_test_summary(result, "Task 1")
    
    return result

