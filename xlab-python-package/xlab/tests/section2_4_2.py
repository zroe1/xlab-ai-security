"""
Tests for section 2.4.2 of the AI Security course.
"""

import pickle
import torch
import torch.nn.functional as F
import os
import pytest
import sys
from typing import List, Callable, Any

# Global variables to store test parameters passed from notebook
_test_config = {
    'model': None,
    'student_function': None,
    'task_2_imgs': None,
    'student_wiggle_relu': None,
    'student_wiggle_relu_grad': None
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

def reference_wiggle_relu(x, amplitude=0.1, frequency=150):
    """Reference implementation of wiggle_ReLU."""
    return F.relu(x) + amplitude * torch.sin(x * frequency)

def reference_wiggle_relu_grad(x, amplitude=0.1, frequency=150):
    """Reference implementation of wiggle_ReLU gradient using autograd."""
    x_cloned = x.clone().requires_grad_(True)
    out = reference_wiggle_relu(x_cloned, amplitude, frequency)
    torch.sum(out).backward()
    return x_cloned.grad

# Task 1 Tests
class TestTask1:
    """Tests for Task 1: Model accuracy evaluation."""
    
    def test_model_accuracy_above_90_percent(self):
        """Test that the model achieves >90% accuracy on test data."""
        model = _test_config['model']
        assert model is not None, "Model not configured for testing"
        
        device = next(model.parameters()).device
        x, y = get_100_examples()
        x, y = x.to(device), y.to(device)
        
        logits = model(x)
        predictions = torch.argmax(logits, axis=1)
        correct_predictions = predictions == y
        accuracy = torch.mean(correct_predictions.float())
        
        assert accuracy > 0.9, f"Model accuracy {accuracy:.4f} is not above 0.9. " \
                              f"Try loading the model with load_model(model_name='Standard', threat_model='Linf')"
    
    def test_student_function_accuracy_matches_model(self):
        """Test that the student function returns the same accuracy as direct model evaluation."""
        model = _test_config['model']
        student_function = _test_config['student_function']
        assert model is not None, "Model not configured for testing"
        assert student_function is not None, "Student function not configured for testing"
        
        device = next(model.parameters()).device
        x, y = get_100_examples()
        x, y = x.to(device), y.to(device)
        
        # Calculate expected accuracy
        logits = model(x)
        predictions = torch.argmax(logits, axis=1)
        correct_predictions = predictions == y
        expected_accuracy = torch.mean(correct_predictions.float())
        
        # Get student function result
        student_accuracy = student_function(model, x, y)
        
        accuracy_diff = abs(float(expected_accuracy) - float(student_accuracy))
        assert accuracy_diff <= 0.01, f"Student function accuracy {student_accuracy:.4f} " \
                                     f"differs from expected {expected_accuracy:.4f} by {accuracy_diff:.4f} " \
                                     f"(threshold: 0.01)"

# Task 2 Tests
class TestTask2:
    """Tests for Task 2: Image classification as frogs."""
    
    def test_all_images_classify_as_frog(self):
        """Test that all provided images classify as class 6 (frog)."""
        model = _test_config['model']
        task_2_imgs = _test_config['task_2_imgs']
        assert model is not None, "Model not configured for testing"
        assert task_2_imgs is not None, "Task 2 images not configured for testing"
        
        device = next(model.parameters()).device
        target_class = 6  # frog class
        failed_images = []
        
        for i, img in enumerate(task_2_imgs):
            # Move image to device if not already there
            if img.device != device:
                img = img.to(device)
            
            # Get model prediction
            logits = model(img)
            predicted_class = torch.argmax(logits, dim=1).item()
            
            if predicted_class != target_class:
                failed_images.append((i, predicted_class))
        
        assert len(failed_images) == 0, f"Expected all {len(task_2_imgs)} images to classify as class 6 (frog), " \
                                       f"but {len(failed_images)} failed: {failed_images[:5]}"

# Task 3 Tests
class TestTask3:
    """Tests for Task 3: wiggle_ReLU function implementation."""
    
    @pytest.mark.parametrize("input_tensor,amplitude,frequency,description", [
        (torch.tensor([1.0, -1.0, 0.0, 2.5, -0.5]), 0.1, 150, "1D tensor, default params"),
        (torch.tensor([[1.0, -2.0], [0.5, -0.3]]), 0.1, 150, "2D tensor, default params"),
        (torch.tensor([[[1.0, -1.0], [0.0, 2.0]], [[0.5, -0.5], [1.5, -1.5]]]), 0.1, 150, "3D tensor, default params"),
        (torch.tensor([1.0, -1.0, 0.0, 2.5]), 0.05, 150, "1D tensor, amplitude=0.05"),
        (torch.tensor([1.0, -1.0, 0.0, 2.5]), 0.2, 150, "1D tensor, amplitude=0.2"),
        (torch.tensor([1.0, -1.0, 0.0, 2.5]), 0.1, 100, "1D tensor, frequency=100"),
        (torch.tensor([1.0, -1.0, 0.0, 2.5]), 0.1, 200, "1D tensor, frequency=200"),
        (torch.tensor([0.0, 0.0, 0.0]), 0.1, 150, "all zeros"),
        (torch.tensor([-1.0, -2.0, -0.5]), 0.1, 150, "all negative values"),
        (torch.tensor([10.0, 5.0, -3.0, 7.5]), 0.05, 75, "custom amplitude and frequency"),
    ])
    def test_wiggle_relu_implementation(self, input_tensor, amplitude, frequency, description):
        """Test wiggle_ReLU function against reference implementation."""
        student_function = _test_config['student_wiggle_relu']
        assert student_function is not None, "Student wiggle_ReLU function not configured for testing"
        
        # Get expected output from reference implementation
        expected_output = reference_wiggle_relu(input_tensor, amplitude, frequency)
        
        # Get student's output
        student_output = student_function(input_tensor, amplitude, frequency)
        
        # Compare outputs with tolerance for floating point precision
        tolerance = 1e-6
        assert torch.allclose(expected_output, student_output, atol=tolerance), \
            f"Test case '{description}' failed. " \
            f"Expected (first few): {expected_output.flatten()[:5].tolist()}, " \
            f"Got (first few): {student_output.flatten()[:5].tolist()}, " \
            f"Max difference: {torch.max(torch.abs(expected_output - student_output)).item():.8f} " \
            f"(tolerance: {tolerance})"

# Task 4 Tests
class TestTask4:
    """Tests for Task 4: wiggle_ReLU gradient function implementation."""
    
    @pytest.mark.parametrize("input_tensor,amplitude,frequency,description", [
        (torch.tensor([1.0, -1.0, 0.0, 2.5, -0.5]), 0.1, 150, "1D tensor, default params"),
        (torch.tensor([[1.0, -2.0], [0.5, -0.3]]), 0.1, 150, "2D tensor, default params"),
        (torch.tensor([[[1.0, -1.0], [0.0, 2.0]], [[0.5, -0.5], [1.5, -1.5]]]), 0.1, 150, "3D tensor, default params"),
        (torch.tensor([1.0, -1.0, 0.0, 2.5]), 0.05, 150, "1D tensor, amplitude=0.05"),
        (torch.tensor([1.0, -1.0, 0.0, 2.5]), 0.2, 150, "1D tensor, amplitude=0.2"),
        (torch.tensor([1.0, -1.0, 0.0, 2.5]), 0.1, 100, "1D tensor, frequency=100"),
        (torch.tensor([1.0, -1.0, 0.0, 2.5]), 0.1, 200, "1D tensor, frequency=200"),
        (torch.tensor([0.0, 0.0, 0.0]), 0.1, 150, "all zeros"),
        (torch.tensor([-1.0, -2.0, -0.5]), 0.1, 150, "all negative values"),
        (torch.tensor([10.0, 5.0, -3.0, 7.5]), 0.05, 75, "custom amplitude and frequency"),
        (torch.tensor([0.001, -0.001, 0.1, -0.1]), 0.1, 150, "small values near zero"),
        (torch.tensor([3.14159, -3.14159, 1.5708, -1.5708]), 0.1, 150, "special values (pi, pi/2)"),
    ])
    def test_wiggle_relu_grad_implementation(self, input_tensor, amplitude, frequency, description):
        """Test wiggle_ReLU gradient function against reference implementation."""
        student_function = _test_config['student_wiggle_relu_grad']
        assert student_function is not None, "Student wiggle_ReLU gradient function not configured for testing"
        
        # Get expected output from reference implementation
        expected_grad = reference_wiggle_relu_grad(input_tensor, amplitude, frequency)
        
        # Get student's output
        student_grad = student_function(input_tensor, amplitude, frequency)
        
        # Compare outputs with higher tolerance for different implementation approaches
        tolerance = 1e-4
        assert torch.allclose(expected_grad, student_grad, atol=tolerance, rtol=tolerance), \
            f"Test case '{description}' failed. " \
            f"Expected grad (first few): {expected_grad.flatten()[:5].tolist()}, " \
            f"Got grad (first few): {student_grad.flatten()[:5].tolist()}, " \
            f"Max difference: {torch.max(torch.abs(expected_grad - student_grad)).item():.8f} " \
            f"(tolerance: {tolerance})"

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
def task1(student_function, model):
    """
    Run Task 1 tests using pytest.
    
    Args:
        student_function (function): The student's function to test.
        model: The model to test with.
    
    Returns:
        dict: A summary dictionary with test results.
    """
    # Configure global test parameters
    _test_config['student_function'] = student_function
    _test_config['model'] = model
    
    # Run pytest tests
    result = _run_pytest_with_capture(TestTask1)
    _print_test_summary(result, "Task 1")
    
    return result

def task2(task_2_imgs, model):
    """
    Run Task 2 tests using pytest.
    
    Args:
        task_2_imgs (list): List of tensors, each of shape [1, 3, 32, 32].
        model: The model to test with.
    
    Returns:
        dict: A summary dictionary with test results.
    """
    # Configure global test parameters
    _test_config['task_2_imgs'] = task_2_imgs
    _test_config['model'] = model
    
    # Run pytest tests
    result = _run_pytest_with_capture(TestTask2)
    _print_test_summary(result, "Task 2")
    
    return result

def task3(student_function):
    """
    Run Task 3 tests using pytest.
    
    Args:
        student_function (function): The student's wiggle_ReLU function to test.
    
    Returns:
        dict: A summary dictionary with test results.
    """
    # Configure global test parameters
    _test_config['student_wiggle_relu'] = student_function
    
    # Run pytest tests
    result = _run_pytest_with_capture(TestTask3)
    _print_test_summary(result, "Task 3")
    
    return result

def task4(student_function):
    """
    Run Task 4 tests using pytest.
    
    Args:
        student_function (function): The student's wiggle_Relu_grad function to test.
    
    Returns:
        dict: A summary dictionary with test results.
    """
    # Configure global test parameters
    _test_config['student_wiggle_relu_grad'] = student_function
    
    # Run pytest tests
    result = _run_pytest_with_capture(TestTask4)
    _print_test_summary(result, "Task 4")
    
    return result

