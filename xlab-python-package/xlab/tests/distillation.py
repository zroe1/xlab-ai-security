"""
Tests for defensive distillation section of the AI Security course, refactored for robust testing.
"""
import io
import sys
from contextlib import redirect_stderr, redirect_stdout
from typing import Callable, Optional

import pytest
import torch
import torch.nn.functional as F

# A configuration class to hold objects passed from the notebook to the tests.
class TestConfig:
    student_softmax_with_temp: Optional[Callable] = None
    student_get_batch_labels: Optional[Callable] = None
    student_cross_entropy_loss_soft: Optional[Callable] = None
    model: Optional[torch.nn.Module] = None

_config = TestConfig()

# --- Pytest Fixtures ---
# These fixtures provide the tests with the necessary objects, like the student's functions.
# If an object is not provided, the test is skipped.

@pytest.fixture
def student_softmax_with_temp() -> Callable:
    """Fixture for the student's softmax_with_temp function."""
    if _config.student_softmax_with_temp is None:
        pytest.skip("Student softmax_with_temp function not provided.")
    return _config.student_softmax_with_temp

@pytest.fixture
def student_get_batch_labels() -> Callable:
    """Fixture for the student's get_batch_labels function."""
    if _config.student_get_batch_labels is None:
        pytest.skip("Student get_batch_labels function not provided.")
    return _config.student_get_batch_labels

@pytest.fixture
def model() -> torch.nn.Module:
    """Fixture to provide the model for testing."""
    if _config.model is None:
        pytest.skip("Model not provided for testing.")
    return _config.model

@pytest.fixture
def student_cross_entropy_loss_soft() -> Callable:
    """Fixture for the student's cross_entropy_loss_soft function."""
    if _config.student_cross_entropy_loss_soft is None:
        pytest.skip("Student cross_entropy_loss_soft function not provided.")
    return _config.student_cross_entropy_loss_soft

# --- Helper Functions ---

def reference_softmax_with_temp(inputs, T):
    """Reference implementation of temperature-scaled softmax."""
    out = inputs / T
    return F.softmax(out, dim=1)

def reference_cross_entropy_loss_soft(soft_labels, probs):
    """Reference implementation of soft cross-entropy loss."""
    assert soft_labels.shape == probs.shape
    batch_size = soft_labels.shape[0]
    log_probs = torch.log(probs)
    return torch.sum(-1 * log_probs * soft_labels) / batch_size

# --- Task 1 Tests: softmax_with_temp Implementation ---

@pytest.mark.task1
@pytest.mark.parametrize("input_tensor,temperature,description", [
    (torch.tensor([[1.0, 2.0, 3.0]]), 1.0, "basic functionality, T=1.0"),
    (torch.tensor([[1.0, 2.0, 3.0], [0.5, 1.5, 2.5]]), 2.0, "multi-batch, T=2.0"),
    (torch.tensor([[1.0, 2.0, 3.0]]), 0.5, "temperature sharpening, T=0.5"),
    (torch.tensor([[0.0, 0.0, 0.0]]), 1.0, "edge case: all zeros"),
])
def test_softmax_with_temp_implementation(student_softmax_with_temp, input_tensor, temperature, description):
    """Test softmax_with_temp function against reference implementation."""
    expected_output = reference_softmax_with_temp(input_tensor, temperature)
    student_output = student_softmax_with_temp(input_tensor, temperature)
    
    tolerance = 1e-6
    assert torch.allclose(expected_output, student_output, atol=tolerance, rtol=tolerance), \
        f"Test case '{description}' failed. " \
        f"Expected (first batch): {expected_output[0].tolist()}, " \
        f"Got (first batch): {student_output[0].tolist()}, " \
        f"Max difference: {torch.max(torch.abs(expected_output - student_output)).item():.8f} " \
        f"(tolerance: {tolerance})"

@pytest.mark.task1
def test_softmax_properties(student_softmax_with_temp):
    """Test that softmax_with_temp maintains essential softmax properties."""
    input_tensor = torch.tensor([[1.0, 2.0, 3.0], [0.5, 1.5, 2.5]])
    temperature = 2.0
    output = student_softmax_with_temp(input_tensor, temperature)
    
    # Test 1: All values should be non-negative
    assert torch.all(output >= 0), f"Softmax output contains negative values: {output}"
    
    # Test 2: Each row should sum to 1 (probability distribution)
    row_sums = torch.sum(output, dim=1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6), \
        f"Softmax rows don't sum to 1: {row_sums.tolist()}"
    
    # Test 3: All values should be <= 1
    assert torch.all(output <= 1), f"Softmax output contains values > 1: {output}"

# --- Task 2 Tests: get_batch_labels Implementation ---

@pytest.mark.task2
@pytest.mark.parametrize("batch_size,temperature,description", [
    (1, 1.0, "single sample, T=1.0"),
    (3, 2.0, "small batch, T=2.0"), 
    (2, 0.5, "batch of 2, T=0.5"),
])
def test_get_batch_labels_basic(student_get_batch_labels, model, batch_size, temperature, description):
    """Test get_batch_labels function with different batch sizes and temperatures."""
    device = next(model.parameters()).device
    
    # Try to determine the correct input shape by examining the first layer
    batch = None
    model_output = None
    
    # Get the first layer to determine expected input size
    first_layer = None
    for module in model.modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
            first_layer = module
            break
    
    if first_layer is None:
        pytest.skip("Cannot determine model input requirements")
    
    try:
        if isinstance(first_layer, torch.nn.Linear):
            # For linear layers, input_features tells us the expected input size
            input_size = first_layer.in_features
            batch = torch.randn(batch_size, input_size).to(device)
        elif isinstance(first_layer, torch.nn.Conv2d):
            # For conv layers, try common image sizes
            in_channels = first_layer.in_channels
            # Try 28x28 (MNIST), then 32x32 (CIFAR)
            for img_size in [28, 32]:
                try:
                    batch = torch.randn(batch_size, in_channels, img_size, img_size).to(device)
                    with torch.no_grad():
                        model_output = model(batch)
                    break
                except RuntimeError:
                    continue
        
        # If we haven't tested the model yet, do it now
        if model_output is None:
            with torch.no_grad():
                model_output = model(batch)
                
    except RuntimeError as e:
        pytest.skip(f"Cannot create compatible input for model: {e}")
    
    # Get output from student function
    output = student_get_batch_labels(batch, temperature)
    
    # Check output shape matches model output shape
    expected_shape = (batch_size, model_output.shape[1])
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"
    
    # Check softmax properties
    assert torch.all(output >= 0), f"Output contains negative values: {output}"
    assert torch.all(output <= 1), f"Output contains values > 1: {output}"
    
    # Check each batch sums to 1
    row_sums = torch.sum(output, dim=1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6), \
        f"Rows don't sum to 1: {row_sums.tolist()}"


@pytest.mark.task2
def test_get_batch_labels_consistency(student_get_batch_labels, model):
    """Test that the function is consistent across multiple calls."""
    device = next(model.parameters()).device
    
    # Get the first layer to determine expected input size
    first_layer = None
    for module in model.modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
            first_layer = module
            break
    
    if first_layer is None:
        pytest.skip("Cannot determine model input requirements")
    
    try:
        if isinstance(first_layer, torch.nn.Linear):
            input_size = first_layer.in_features
            batch = torch.randn(2, input_size).to(device)
        elif isinstance(first_layer, torch.nn.Conv2d):
            in_channels = first_layer.in_channels
            # Try 28x28 (MNIST), then 32x32 (CIFAR)
            for img_size in [28, 32]:
                try:
                    batch = torch.randn(2, in_channels, img_size, img_size).to(device)
                    with torch.no_grad():
                        model(batch)  # Test if it works
                    break
                except RuntimeError:
                    continue
    except RuntimeError as e:
        pytest.skip(f"Cannot create compatible input for model: {e}")
    
    temperature = 1.5
    
    # Call function multiple times with same input
    output1 = student_get_batch_labels(batch, temperature)
    output2 = student_get_batch_labels(batch, temperature)
    
    # Should get identical results
    assert torch.allclose(output1, output2, atol=1e-8), \
        "Function should be deterministic and give identical results for same inputs"

# --- Task 3 Tests: cross_entropy_loss_soft Implementation ---

@pytest.mark.task3
@pytest.mark.parametrize("batch_size,num_classes,description", [
    (1, 3, "single sample, 3 classes"),
    (5, 10, "small batch, 10 classes"),
    (2, 5, "batch of 2, 5 classes"),
])
def test_cross_entropy_loss_soft_implementation(student_cross_entropy_loss_soft, batch_size, num_classes, description):
    """Test cross_entropy_loss_soft function against reference implementation."""
    # Create random soft labels and predictions (both should be probability distributions)
    soft_labels = torch.rand(batch_size, num_classes)
    soft_labels = soft_labels / torch.sum(soft_labels, dim=1, keepdim=True)  # Normalize to sum to 1
    
    probs = torch.rand(batch_size, num_classes)
    probs = probs / torch.sum(probs, dim=1, keepdim=True)  # Normalize to sum to 1
    
    # Add small epsilon to avoid log(0)
    probs = torch.clamp(probs, min=1e-8)
    
    expected_loss = reference_cross_entropy_loss_soft(soft_labels, probs)
    student_loss = student_cross_entropy_loss_soft(soft_labels, probs)
    
    tolerance = 1e-6
    assert torch.allclose(expected_loss, student_loss, atol=tolerance, rtol=tolerance), \
        f"Test case '{description}' failed. " \
        f"Expected loss: {expected_loss.item():.6f}, " \
        f"Got loss: {student_loss.item():.6f}, " \
        f"Difference: {abs(expected_loss.item() - student_loss.item()):.8f} " \
        f"(tolerance: {tolerance})"

# @pytest.mark.task3
# def test_cross_entropy_loss_properties(student_cross_entropy_loss_soft):
#     """Test that cross_entropy_loss_soft has expected mathematical properties."""
#     batch_size, num_classes = 3, 4
    
#     # Test 1: Perfect prediction (soft_labels == probs) should give low loss
#     perfect_probs = torch.tensor([[0.7, 0.2, 0.05, 0.05],
#                                   [0.1, 0.8, 0.05, 0.05],
#                                   [0.05, 0.05, 0.9, 0.0]])
#     perfect_loss = student_cross_entropy_loss_soft(perfect_probs, perfect_probs)
    
#     # Test 2: Worst prediction (uniform when target is one-hot) should give higher loss
#     one_hot_labels = torch.tensor([[1.0, 0.0, 0.0, 0.0],
#                                    [0.0, 1.0, 0.0, 0.0],
#                                    [0.0, 0.0, 1.0, 0.0]])
#     uniform_probs = torch.ones_like(one_hot_labels) / num_classes
#     worst_loss = student_cross_entropy_loss_soft(one_hot_labels, uniform_probs)
    
#     # Perfect prediction should have lower loss than uniform prediction
#     assert perfect_loss < worst_loss, \
#         f"Perfect prediction loss {perfect_loss.item():.4f} should be < uniform prediction loss {worst_loss.item():.4f}"
    
#     # Loss should be non-negative
#     assert perfect_loss >= 0, f"Loss should be non-negative, got {perfect_loss.item():.4f}"
#     assert worst_loss >= 0, f"Loss should be non-negative, got {worst_loss.item():.4f}"

@pytest.mark.task3
def test_cross_entropy_loss_one_hot_labels(student_cross_entropy_loss_soft):
    """Test cross_entropy_loss_soft with one-hot (hard) labels."""
    # Create one-hot labels (simulating hard labels)
    one_hot_labels = torch.tensor([[1.0, 0.0, 0.0],
                                   [0.0, 1.0, 0.0]])
    
    # Create some predictions
    probs = torch.tensor([[0.8, 0.15, 0.05],
                          [0.2, 0.7, 0.1]])
    
    loss = student_cross_entropy_loss_soft(one_hot_labels, probs)
    
    # Verify it's a scalar
    assert loss.dim() == 0, f"Loss should be a scalar, got tensor with shape {loss.shape}"
    
    # Verify it's positive (should be since predictions aren't perfect)
    assert loss > 0, f"Loss should be positive for imperfect predictions, got {loss.item():.4f}"

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
    
    print(f"\nRunning tests for Defensive Distillation, {task_name}...")
    print("=" * 70)
    
    if failed == 0 and total > 0:
        print(f"🎉 All tests passed! ({passed}/{total})")
    elif total == 0:
        print("No tests were run. Check if functions were passed correctly.")
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

def task1(student_function: Callable):
    """Run Task 1 tests."""
    _config.student_softmax_with_temp = student_function
    result = _run_pytest_with_capture("task1")
    _print_test_summary(result, "Task 1")
    return result

def task2(student_function: Callable, model: torch.nn.Module):
    """Run Task 2 tests."""
    _config.student_get_batch_labels = student_function
    _config.model = model
    result = _run_pytest_with_capture("task2")
    _print_test_summary(result, "Task 2")
    return result

def task3(student_function: Callable):
    """Run Task 3 tests."""
    _config.student_cross_entropy_loss_soft = student_function
    result = _run_pytest_with_capture("task3")
    _print_test_summary(result, "Task 3")
    return result
