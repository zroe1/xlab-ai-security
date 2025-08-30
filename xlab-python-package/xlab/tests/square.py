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
_test_config: dict[str, Any] = {
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
    """Tests for get_rand_square_coordinates."""

    def test_output_type_and_bounds(self):
        """Returns (y, x) as ints within valid bounds."""
        coords_func = _test_config['student_function']

        w, h = 32, 5
        y, x = coords_func(w, h)

        assert isinstance(y, int) and isinstance(x, int)
        assert 0 <= y < (w - h)
        assert 0 <= x < (w - h)

    def test_randomness_across_calls(self):
        """Multiple calls produce differing coordinates (high probability)."""
        coords_func = _test_config['student_function']

        w, h = 32, 5
        results = [coords_func(w, h) for _ in range(5)]

        all_equal = all(results[0] == r for r in results[1:])
        assert not all_equal

class TestTask2:
    """Tests for the l_inf_dist function."""

    def test_output_shape_and_type(self):
        """Output tensor has correct shape and dtype."""
        dist_func = _test_config['student_function']

        # Test parameters
        epsilon, h, w, c = 0.1, 8, 32, 3

        # Execution
        result_tensor = dist_func(epsilon, h, w, c)

        # Assertions
        assert isinstance(result_tensor, torch.Tensor)
        assert result_tensor.shape == (c, w, w)
        assert result_tensor.dtype == torch.float32

    def test_randomness_across_runs(self):
        """Multiple runs with same parameters should produce different outputs."""
        dist_func = _test_config['student_function']

        # Test parameters
        epsilon, h, w, c = 0.1, 8, 32, 3

        # Execute multiple times
        results = [dist_func(epsilon, h, w, c) for _ in range(5)]

        # Assert that not all results are identical
        all_equal = True
        for r in results[1:]:
            if not torch.equal(results[0], r):
                all_equal = False
                break
        assert not all_equal
    
class TestTask3:
    """Tests for square_schedule."""

    def test_basic_no_halving(self):
        """At i=0 and default p, returns expected floor of w*sqrt(p)."""
        schedule_func = _test_config['student_function']
        w, i = 32, 0
        side = schedule_func(i=i, w=w)
        assert isinstance(side, int)
        assert side == int((0.10 * (w ** 2)) ** 0.5)  # int(sqrt(0.1) * w) = 10

    def test_single_halving(self):
        """At i hitting a half-point (e.g., 10), p halves once."""
        schedule_func = _test_config['student_function']
        w, i = 32, 10  # halves once from 0.10 -> 0.05
        expected = int((0.05 * (w ** 2)) ** 0.5)  # int(sqrt(0.05) * w) = 7
        side = schedule_func(i=i, w=w)
        assert side == expected

    def test_minimum_enforced(self):
        """For large i and small p, returns at least 2."""
        schedule_func = _test_config['student_function']
        w, i = 32, 10000  # many halvings
        side = schedule_func(i=i, w=w)
        assert side == 2

class TestTask4:
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
                
        

class TestTask5:
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


class TestTask6:
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


class TestTask7:
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

# Notebook interface functions
def task1(student_function: Callable):
    """Run Task 1 tests: get_rand_square_coordinates."""
    _test_config['student_function'] = student_function
    result = _run_pytest_with_capture(TestTask1)
    _print_test_summary(result, "Task 1: get_rand_square_coordinates")
    return result

def task2(student_function: Callable):
    """Run Task 2 tests: l_inf_dist."""
    _test_config['student_function'] = student_function
    result = _run_pytest_with_capture(TestTask2)
    _print_test_summary(result, "Task 2: l_inf_dist")
    return result

def task3(student_function: Callable):
    """Run Task 3 tests: square_schedule."""
    _test_config['student_function'] = student_function
    result = _run_pytest_with_capture(TestTask3)
    _print_test_summary(result, "Task 3: square_schedule")
    return result

def task4(model, student_function: Callable):
    """Run Task 4 tests: l_inf_square_attack."""
    _test_config['model'] = model
    _test_config['student_function'] = student_function
    result = _run_pytest_with_capture(TestTask4)
    _print_test_summary(result, "Task 4: l_inf_square_attack")
    return result

def task5(student_function: Callable):
    """Run Task 5 tests: M function."""
    _test_config['student_function'] = student_function
    result = _run_pytest_with_capture(TestTask5)
    _print_test_summary(result, "Task 5: M function")
    return result

def task6(student_function: Callable):
    """Run Task 6 tests: eta function."""
    _test_config['student_function'] = student_function
    result = _run_pytest_with_capture(TestTask6)
    _print_test_summary(result, "Task 6: eta function")
    return result

def task7(student_function: Callable):
    """Run Task 7 tests: l_2_dist function."""
    _test_config['student_function'] = student_function
    result = _run_pytest_with_capture(TestTask7)
    _print_test_summary(result, "Task 7: l_2_dist function")
    return result