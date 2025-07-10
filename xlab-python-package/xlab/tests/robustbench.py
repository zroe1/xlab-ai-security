"""
Tests for section 2.4.2 of the AI Security course, refactored for robust testing.
"""
import io
import os
import pickle
import sys
from contextlib import redirect_stderr, redirect_stdout
from typing import List, Callable, Any

import pytest
import torch
import torch.nn.functional as F

# A configuration class to hold objects passed from the notebook to the tests.
class TestConfig:
    model: torch.nn.Module = None
    student_function_task1: Callable = None
    task_2_imgs: List[torch.Tensor] = None
    student_wiggle_relu: Callable = None
    student_wiggle_relu_grad: Callable = None

_config = TestConfig()

# --- Pytest Fixtures ---
# These fixtures provide the tests with the necessary objects, like the model
# or the student's functions. If an object is not provided, the test is skipped.

@pytest.fixture
def model() -> torch.nn.Module:
    """Fixture to provide the model for testing."""
    if _config.model is None:
        pytest.skip("Model not provided for testing.")
    return _config.model

@pytest.fixture
def student_function_task1() -> Callable:
    """Fixture for the student's function in Task 1."""
    if _config.student_function_task1 is None:
        pytest.skip("Student function for Task 1 not provided.")
    return _config.student_function_task1

@pytest.fixture
def task_2_imgs() -> List[torch.Tensor]:
    """Fixture for the images in Task 2."""
    if _config.task_2_imgs is None:
        pytest.skip("Images for Task 2 not provided.")
    return _config.task_2_imgs

@pytest.fixture
def student_wiggle_relu() -> Callable:
    """Fixture for the student's wiggle_ReLU function in Task 3."""
    if _config.student_wiggle_relu is None:
        pytest.skip("Student wiggle_ReLU function not provided.")
    return _config.student_wiggle_relu

@pytest.fixture
def student_wiggle_relu_grad() -> Callable:
    """Fixture for the student's wiggle_ReLU_grad function in Task 4."""
    if _config.student_wiggle_relu_grad is None:
        pytest.skip("Student wiggle_ReLU_grad function not provided.")
    return _config.student_wiggle_relu_grad

# --- Helper Functions ---

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

# --- Task 1 Tests: Model Accuracy ---

@pytest.mark.task1
def test_model_accuracy_above_80_percent(model):
    """Test that the model achieves >80% accuracy on test data."""
    device = next(model.parameters()).device
    x, y = get_100_examples()
    x, y = x.to(device), y.to(device)
    
    logits = model(x)
    predictions = torch.argmax(logits, axis=1)
    correct_predictions = predictions == y
    accuracy = torch.mean(correct_predictions.float())
    
    assert accuracy > 0.8, f"Model accuracy {accuracy:.4f} is not above 0.8." \
                        f"Check how you are loading your model"

@pytest.mark.task1
def test_student_function_accuracy_matches_model(model, student_function_task1):
    """Test that the student function returns the same accuracy as direct model evaluation."""
    device = next(model.parameters()).device
    x, y = get_100_examples()
    x, y = x.to(device), y.to(device)
    
    # Calculate expected accuracy
    logits = model(x)
    predictions = torch.argmax(logits, axis=1)
    correct_predictions = predictions == y
    expected_accuracy = torch.mean(correct_predictions.float())
    
    # Get student function result
    student_accuracy = student_function_task1(model, x, y)
    
    accuracy_diff = abs(float(expected_accuracy) - float(student_accuracy))
    assert accuracy_diff <= 0.01, f"Student function accuracy {student_accuracy:.4f} " \
                                 f"differs from expected {expected_accuracy:.4f} by {accuracy_diff:.4f} " \
                                 f"(threshold: 0.01)"

# --- Task 2 Tests: Image Classification as Frogs ---

@pytest.mark.task2
def test_all_images_classify_as_frog(model, task_2_imgs):
    """Test that all provided images classify as class 6 (frog)."""
    device = next(model.parameters()).device
    target_class = 6  # frog class
    failed_images = []
    
    for i, img in enumerate(task_2_imgs):
        if img.device != device:
            img = img.to(device)
        
        logits = model(img)
        predicted_class = torch.argmax(logits, dim=1).item()
        
        if predicted_class != target_class:
            failed_images.append((i, predicted_class))
    
    assert len(failed_images) == 0, f"Expected all {len(task_2_imgs)} images to classify as class 6 (frog), " \
                                   f"but {len(failed_images)} failed: {failed_images[:5]}"

# --- Task 3 Tests: wiggle_ReLU Implementation ---

@pytest.mark.task3
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
def test_wiggle_relu_implementation(student_wiggle_relu, input_tensor, amplitude, frequency, description):
    """Test wiggle_ReLU function against reference implementation."""
    expected_output = reference_wiggle_relu(input_tensor, amplitude, frequency)
    student_output = student_wiggle_relu(input_tensor, amplitude, frequency)
    
    tolerance = 1e-6
    assert torch.allclose(expected_output, student_output, atol=tolerance), \
        f"Test case '{description}' failed. " \
        f"Expected (first few): {expected_output.flatten()[:5].tolist()}, " \
        f"Got (first few): {student_output.flatten()[:5].tolist()}, " \
        f"Max difference: {torch.max(torch.abs(expected_output - student_output)).item():.8f} " \
        f"(tolerance: {tolerance})"

# --- Task 4 Tests: wiggle_ReLU Gradient Implementation ---

@pytest.mark.task4
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
def test_wiggle_relu_grad_implementation(student_wiggle_relu_grad, input_tensor, amplitude, frequency, description):
    """Test wiggle_ReLU gradient function against reference implementation."""
    expected_grad = reference_wiggle_relu_grad(input_tensor, amplitude, frequency)
    student_grad = student_wiggle_relu_grad(input_tensor, amplitude, frequency)
    
    tolerance = 1e-4
    assert torch.allclose(expected_grad, student_grad, atol=tolerance, rtol=tolerance), \
        f"Test case '{description}' failed. " \
        f"Expected grad (first few): {expected_grad.flatten()[:5].tolist()}, " \
        f"Got grad (first few): {student_grad.flatten()[:5].tolist()}, " \
        f"Max difference: {torch.max(torch.abs(expected_grad - student_grad)).item():.8f} " \
        f"(tolerance: {tolerance})"

# --- Pytest Runner and Summary ---

def _run_pytest_with_capture(marker: str, verbose: bool = True):
    """
    Run pytest on tests with a specific marker and capture results.
    """
    current_file = __file__
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    
    try:
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            exit_code = pytest.main([
                f'{current_file}',
                '-m', marker,
                '-v' if verbose else '-q',
                '--tb=short',
                '--no-header'
            ])
        
        stdout_output = stdout_capture.getvalue()
        passed = stdout_output.count("PASSED")
        failed = stdout_output.count("FAILED")
        skipped = stdout_output.count("SKIPPED")

        total_tests = passed + failed
        
        return {
            "total_tests": total_tests,
            "passed": passed,
            "failed": failed,
            "score": round((passed / total_tests) * 100) if total_tests > 0 else 0,
            "stdout": stdout_output,
            "stderr": stderr_capture.getvalue(),
            "exit_code": exit_code
        }
    
    except Exception as e:
        return {
            "total_tests": 0, "passed": 0, "failed": 0, "score": 0,
            "stdout": stdout_capture.getvalue(),
            "stderr": f"Error running pytest: {e}",
            "exit_code": 1
        }

def _print_test_summary(result_dict, task_name):
    """Print a formatted summary of test results."""
    GREEN, RED, RESET = '\033[92m', '\033[91m', '\033[0m'
    
    passed, failed, total = result_dict["passed"], result_dict["failed"], result_dict["total_tests"]
    
    print(f"\nRunning tests for Section 2.4.2, {task_name}...")
    print("=" * 70)
    
    if failed == 0 and total > 0:
        print(f"🎉 All tests passed! ({passed}/{total})")
    elif total == 0:
        print("No tests were run. Check if functions/data were passed correctly.")
    else:
        print(f"📊 Results: {passed} passed, {failed} failed out of {total} total")
    
    print("=" * 70)
    
    if result_dict.get("stdout"):
        print("\nDetailed output:")
        print(result_dict["stdout"])
    
    if result_dict.get("stderr") and result_dict["stderr"].strip():
        print(f"\n{RED}Errors:{RESET}")
        print(result_dict["stderr"])

# --- Notebook Interface Functions ---

def task1(student_function: Callable, model: torch.nn.Module):
    """Run Task 1 tests."""
    _config.student_function_task1 = student_function
    _config.model = model
    result = _run_pytest_with_capture("task1")
    _print_test_summary(result, "Task 1")
    return result

def task2(task_2_imgs: List[torch.Tensor], model: torch.nn.Module):
    """Run Task 2 tests."""
    _config.task_2_imgs = task_2_imgs
    _config.model = model
    result = _run_pytest_with_capture("task2")
    _print_test_summary(result, "Task 2")
    return result

def task3(student_function: Callable):
    """Run Task 3 tests."""
    _config.student_wiggle_relu = student_function
    result = _run_pytest_with_capture("task3")
    _print_test_summary(result, "Task 3")
    return result

def task4(student_function: Callable):
    """Run Task 4 tests."""
    _config.student_wiggle_relu_grad = student_function
    result = _run_pytest_with_capture("task4")
    _print_test_summary(result, "Task 4")
    return result

