"""
Tests for section 2.3 of the AI Security course.
"""


import pytest
import sys
import torch
import numpy as np
import random
from typing import Any, Callable

# ==============================================================================
# Global Configuration and Mocks
# ==============================================================================

# Global variable to store test parameters passed from the notebook
_test_config = {
    'student_function': None,
    'model': None,
    'loss': None
}


from xlab.utils import prediction, process_image, show_image, SimpleCNN, CIFAR10
import torch
import numpy as np
import random

classes = CIFAR10().classes

IMG_PATH = 'data/car.jpg'
from huggingface_hub import hf_hub_download
from xlab.models import MiniWideResNet, BasicBlock
import torch

def _get_default_model():
    """Lazy load the default model to avoid heavy operations during import."""
    try:
        model = MiniWideResNet()
        model_path = hf_hub_download(
            repo_id="uchicago-xlab-ai-security/tiny-wideresnet-cifar10",
            filename="adversarial_basics_cnn.pth"
        )
        model = torch.load(model_path, map_location='cpu', weights_only=False)
        return model
    except Exception as e:
        print(f"Warning: Could not load default model: {e}")
        return None



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

# ==============================================================================
# Pytest Test Classes
# ==============================================================================

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
    
class TestTask2:
    """Tests for Task 2: l_inf_square_attack"""

    def test_output(self):
        """Tests that the output image tensor values are adversarial"""
        attack_func = _test_config['student_function']
        model = _test_config['model'] or _get_default_model()
        IMG_PATH = 'car.jpg'
        img = process_image(IMG_PATH)
        label = prediction(model, process_image(IMG_PATH))[0]
        loss_fn = torch.nn.CrossEntropyLoss()
        x_adv = attack_func(model, loss_fn, img, label, 500)
        pred = prediction(model, x_adv)[0]
        assert not torch.equal(label, pred)
                
        

class TestTask3:
    """Tests for Task 3: M (helper function)"""

    def test_return_type(self):
        """Tests that the function returns a numerical type."""
        M_func = _test_config['student_function']
        result = M_func(r=1, s=1, h1=4, h2=4)
        assert isinstance(result, (int, np.integer))

    def test_known_values(self):
        """Tests the output for a few known inputs based on the implementation."""
        M_func = _test_config['student_function']
        # Test case 1: 5x5 square
        # M(r,s,5,5) = max(abs(r - 5//2 - 1), abs(s - 5//2 - 1)) = max(abs(r-3), abs(s-3))
        assert M_func(r=0, s=0, h1=5, h2=5) == 3
        assert M_func(r=2, s=2, h1=5, h2=5) == 1
        assert M_func(r=4, s=4, h1=5, h2=5) == 1
        
        # Test case 2: Rectangular 10x4
        # M(r,s,10,4) = max(abs(r - 10//2 - 1), abs(s - 4//2 - 1)) = max(abs(r-6), abs(s-3))
        assert M_func(r=0, s=0, h1=10, h2=4) == 6
        assert M_func(r=5, s=2, h1=10, h2=4) == 1


class TestTask4:
    """Tests for Task 4: eta (helper function)"""

    def setup_method(self):
        """Make student's M function available for eta to call."""
        # This assumes M is defined in the global scope of the student's file
        # and pytest can discover it. This is a reasonable assumption for this setup.
        pass

    def test_output_shape_and_type(self):
        """Tests that the output tensor has the correct shape and type."""
        eta_func = _test_config['student_function']
        h1, h2 = 8, 6
        result = eta_func(h1, h2)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (h1, h2)

    def test_known_value_simple(self):
        """Tests the output for a simple, manually calculable case."""
        eta_func = _test_config['student_function']
        # For h1=2, h2=1, the logic is as follows:
        # n = 1. M_matrix is 2x1. M values are all 2.
        # max_value = 2.
        # cache will be [1/(1+1-0), 1/2 + 1/(1+1-1)] = [0.5, 1.5]
        # eta_matrix is cache[M-1] = cache[1] = 1.5
        expected = torch.tensor([[1.5], [1.5]])
        result = eta_func(h1=2, h2=1)
        assert torch.allclose(result, expected)

    def test_no_zero_division(self):
        """Tests that the `if n+1-i != 0` check prevents division by zero."""
        eta_func = _test_config['student_function']
        try:
            # A large value for h1 could potentially trigger the edge case
            # if the logic was flawed.
            eta_func(h1=25, h2=15)
        except ZeroDivisionError:
            pytest.fail("ZeroDivisionError was raised in the eta function.")


class TestTask5:
    """Tests for Task 5: l_2_dist"""

    def test_output_shape_and_type(self):
        """Tests that the output delta has the correct shape and type."""
        l2_dist_func = _test_config['student_function']
        c, w, h = 3, 32, 10
        x = torch.rand(1, c, w, w)
        x_hat = x.clone() # Start with zero perturbation
        
        delta = l2_dist_func(x_hat, x, h=h, w=w, c=c)
        assert isinstance(delta, torch.Tensor)
        assert delta.shape == (1, c, w, w)

    def test_norm_of_first_step(self):
        """Tests that the norm of the first perturbation (delta) is equal to epsilon."""
        l2_dist_func = _test_config['student_function']
        c, w, h, epsilon = 3, 32, 10, 5.0
        
        # When x_hat = x, the perturbation `v` is zero.
        # The algorithm should create a new perturbation `delta` with L2 norm of epsilon.
        x = torch.zeros(1, c, w, w)
        x_hat = x.clone()
        
        delta = l2_dist_func(x_hat, x, epsilon=epsilon, h=h, w=w, c=c)
        
        delta_norm = torch.norm(delta)
        assert torch.allclose(delta_norm, torch.tensor(epsilon), atol=1e-5)


# ==============================================================================
# Pytest Runner and Reporter
# ==============================================================================

def _run_pytest_with_capture(test_class_or_function: Any, verbose: bool = True) -> dict:
    """
    Run pytest on a specific test class or function and capture results.
    """
    import io
    from contextlib import redirect_stdout, redirect_stderr
    
    test_name = test_class_or_function.__name__
    current_file = __file__
    
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    
    try:
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            exit_code = pytest.main([
                f'{current_file}::{test_name}',
                '-v' if verbose else '-q',
                '--tb=short',
                '--no-header'
            ])
        
        stdout_output = stdout_capture.getvalue()
        lines = stdout_output.split('\n')
        
        passed = sum(1 for line in lines if '::' in line and 'PASSED' in line)
        failed = sum(1 for line in lines if '::' in line and ('FAILED' in line or 'ERROR' in line))
        total_tests = passed + failed
        
        return {
            "total_tests": total_tests, "passed": passed, "failed": failed,
            "score": round((passed / total_tests) * 100) if total_tests > 0 else 0,
            "stdout": stdout_output, "stderr": stderr_capture.getvalue(),
            "exit_code": exit_code
        }
    
    except Exception as e:
        return {
            "total_tests": 0, "passed": 0, "failed": 0, "score": 0,
            "stdout": stdout_capture.getvalue(), "stderr": f"Error running pytest: {e}",
            "exit_code": 1
        }

def _print_test_summary(result_dict: dict, task_name: str):
    """Print a formatted summary of test results."""
    GREEN, RED, RESET = '\033[92m', '\033[91m', '\033[0m'
    
    passed, failed, total = result_dict["passed"], result_dict["failed"], result_dict["total_tests"]
    
    print(f"\nRunning tests for {task_name}...")
    print("=" * 70)
    
    if failed == 0 and total > 0:
        print(f"🎉 {GREEN}All tests passed! ({passed}/{total}){RESET}")
    elif total > 0:
        print(f"📊 {RED}Results: {passed} passed, {failed} failed out of {total} total{RESET}")
    else:
        print(f"🤷 {RED}No tests were found or run for {task_name}.{RESET}")

    print("=" * 70)
    
    if result_dict.get("stdout"):
        print("\nDetailed output:")
        print(result_dict["stdout"])
    
    if result_dict.get("stderr") and result_dict["stderr"].strip():
        print(f"\n{RED}Errors:{RESET}")
        print(result_dict["stderr"])

# ==============================================================================
# Notebook Interface Functions
# ==============================================================================

def task2(student_function: Callable):
    """Runs tests for Task 2: l_inf_square_attack."""
    _test_config['student_function'] = student_function
    result = _run_pytest_with_capture(TestTask1)
    _print_test_summary(result, "Task 2: l_inf_square_attack")
    return result

def task3(student_function: Callable):
    """Runs tests for Task 3: M helper function."""
    _test_config['student_function'] = student_function
    result = _run_pytest_with_capture(TestTask2)
    _print_test_summary(result, "Task 3: M function")
    return result

def task4(student_function: Callable):
    """Runs tests for Task 4: eta helper function."""
    _test_config['student_function'] = student_function
    result = _run_pytest_with_capture(TestTask3)
    _print_test_summary(result, "Task 4: eta function")
    return result

def task5(student_function: Callable):
    """Runs tests for Task 5: l_2_dist function."""
    _test_config['student_function'] = student_function
    result = _run_pytest_with_capture(TestTask4)
    _print_test_summary(result, "Task 4: l_2_dist function")
    return result

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


def task2(model, student_function):
    """
    Run Task 2 tests using pytest.
    
    Args:
        student_function: The function to test with.
    
    Returns:
        dict: A summary dictionary with test results.
    """
    # Configure global test parameters
    _test_config['model'] = model
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


def task5(student_function):
    """
    Run Task 5 tests using pytest.
    
    Args:
        student_function: The function to test with.
    
    Returns:
        dict: A summary dictionary with test results.
    """
    # Configure global test parameters
    _test_config['student_function'] = student_function
    
    # Run pytest tests
    result = _run_pytest_with_capture(TestTask5)
    _print_test_summary(result, "Task 5")
    
    return result