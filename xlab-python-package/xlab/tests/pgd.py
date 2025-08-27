"""
Tests for section 1.1 of the AI Security course.
"""

import pickle
import torch
import os
import pytest
import numpy as np
import xlab
from importlib import resources
from .. import utils


# Global variables to store test parameters passed from notebook
_test_config = {
    'model': None,
    'student_function': None,
    'loss_fn': None,
    'label': None,
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

# # Task 1 Tests
# class TestTask1:    
#     def test_model_prediction(self):
#         """Test that the model achieves the same predictions as the correct model"""
#         model = _test_config['model']
#         model.eval()
        
#         x, y = get_100_examples()
#         assert model(x) is not None, "Model not configured for testing"
#         with torch.no_grad():

#             logits = model(x)
#             predictions = torch.argmax(logits, axis=1)
    
#             cnn = xlab.utils.SimpleCNN()

#             cnn_path = resources.files("xlab.data").joinpath("CNN_weights.pth")
#             cnn.load_state_dict(torch.load(cnn_path))        

#             cnn.eval()
#             logits = cnn(x)
#             cnn_pred = torch.argmax(logits, axis=1)

#             assert torch.equal(predictions, cnn_pred)


#Task 2 Tests
class TestTask2:    
    def test_valid_output(self):
        """Tests successful processing of frog image."""

        process_image = _test_config['student_function']
        img_path = resources.files("xlab.data").joinpath("cat.jpg")
        
        # Execution
        result_tensor = process_image(img_path)
        
        # Assertions
        assert isinstance(result_tensor, torch.Tensor)
        assert result_tensor.shape == (1, 3, 32, 32)
        assert result_tensor.min() >= 0.0 and result_tensor.max() <= 1.0


    def test_correct_output(self):
        """Tests successful processing of cat image."""

        process_image = _test_config['student_function']
        img_path = resources.files("xlab.data").joinpath("cat.jpg")
        
        # Execution
        result_tensor = process_image(img_path)
        correct_tensor = utils.process_image(img_path)
        
        # Assertions
        assert torch.allclose(result_tensor, correct_tensor)


#Task 3 Tests
class TestTask3:
    
    # Test 1: Validate the fundamental properties of the output
    def test_output_properties(self):
        """
        Tests that the output is a valid image tensor: correct shape, type, and value range.
        """
        # Setup
        model = _test_config['model']
        student_FGSM = _test_config['student_function']
        loss_fn = torch.nn.CrossEntropyLoss()
        epsilon = 8/255
        img_path = resources.files("xlab.data").joinpath("cat.jpg")
        y_target = torch.tensor([6]) 
        adv_tensor = student_FGSM(model, loss_fn, img_path, y_target, epsilon)
        
        # Assertions
        assert isinstance(adv_tensor, torch.Tensor), "Output should be a torch.Tensor"
        assert adv_tensor.shape == (1, 3, 32, 32), "Output shape is incorrect"
        assert adv_tensor.min() >= 0.0 and adv_tensor.max() <= 1.0, "Output values must be clamped between 0 and 1"

    # Test 2: Verify that a non-zero epsilon actually changes the image
    def test_perturbation_is_applied(self):
        """
        Tests that for a non-zero epsilon, the output image is different from the input.
        """
        # Setup
        model = _test_config['model']
        student_FGSM = _test_config['student_function']
        loss_fn = torch.nn.CrossEntropyLoss()
        epsilon = 0.1
        img_path = resources.files("xlab.data").joinpath("cat.jpg")
        y_target = torch.tensor([6]) 
        adv_tensor = student_FGSM(model, loss_fn, img_path, y_target, epsilon)
        original_img = utils.process_image(img_path)
        
        # Assertions
        # The original and adversarial images should not be identical
        assert not torch.equal(original_img, adv_tensor), "Image was not perturbed"

    # Test 3: Verify the logic of the epsilon parameter
    def test_epsilon_zero_produces_no_change(self):
        """
        Tests the edge case where epsilon is 0. The output should be identical to the original image.
        """
        # Setup
        model = _test_config['model']
        student_FGSM = _test_config['student_function']
        loss_fn = torch.nn.CrossEntropyLoss()
        epsilon = 0
        img_path = resources.files("xlab.data").joinpath("cat.jpg")
        y_target = torch.tensor([6]) 
        adv_tensor = student_FGSM(model, loss_fn, img_path, y_target, epsilon)
        original_img = utils.process_image(img_path)
        
        # Assertions
        # With epsilon=0, the perturbation term is zero, so the images should be identical.
        assert torch.equal(original_img, adv_tensor), "Image should not change when epsilon is 0"

    # Test 4: Verify that perturbation magnitude stays within epsilon bound
    def test_perturbation_within_epsilon_bound(self):
        """
        Tests that the L∞ distance between original and adversarial images is at most epsilon.
        """
        # Setup
        model = _test_config['model']
        student_FGSM = _test_config['student_function']
        loss_fn = torch.nn.CrossEntropyLoss()
        epsilon = 8/255
        img_path = resources.files("xlab.data").joinpath("cat.jpg")
        y_target = torch.tensor([6]) 
        adv_tensor = student_FGSM(model, loss_fn, img_path, y_target, epsilon)
        original_img = utils.process_image(img_path)
        
        # Calculate L∞ distance (maximum absolute difference)
        perturbation = torch.abs(adv_tensor - original_img)
        max_perturbation = torch.max(perturbation)
        
        # Assertions
        # The maximum perturbation should not exceed epsilon
        assert max_perturbation <= epsilon + 1e-6, f"Perturbation magnitude {max_perturbation} exceeds epsilon {epsilon}"

    def test_output_adversarial(self):
        """
        Tests whether function produces adversarial output
        """
        # Setup
        model = _test_config['model']
        student_FGSM = _test_config['student_function']
        loss_fn = torch.nn.CrossEntropyLoss()
        epsilon = 1/50
        img_path = resources.files("xlab.data").joinpath("cat.jpg")
        y_target = torch.tensor([6]) 
        adv_tensor = student_FGSM(model, loss_fn, img_path, y_target, epsilon)
        pred = xlab.utils.prediction(model, adv_tensor)[0]      
        
        # Assertions
        assert not torch.equal(y_target, pred), "Output should be adversarial"

class TestTask4:    
    def test_clips_beyond_epsilon_ball(self):
        """
        Tests that a perturbation larger than epsilon is correctly clipped back
        to the epsilon boundary.
        """
        clip = _test_config['student_function']
        # Arrange: original is 0.5, epsilon is 0.1.
        # The allowed range around original is [0.4, 0.6].
        x_original = torch.tensor([0.5])
        epsilon = 0.1
        
        # Act: We provide an input of 0.8, which is 0.3 away (far beyond epsilon).
        x_perturbed = torch.tensor([0.8])
        x_clipped = clip(x_perturbed, x_original, epsilon)
        
        # Assert: The result should be clipped back to the max allowed value: 0.5 + 0.1 = 0.6
        expected_result = torch.tensor([0.6])
        torch.testing.assert_close(x_clipped, expected_result)

    # Test 2: Verify the final value clipping (for values > 1)
    def test_clips_values_above_one(self):
        """
        Tests that a final value greater than 1.0 is correctly clipped to 1.0,
        even if the perturbation was within the epsilon limit.
        """
        clip = _test_config['student_function']
        # Arrange: original is 0.95, epsilon is 0.1.
        # The perturbation is 0.08, which is within the epsilon limit.
        x_original = torch.tensor([0.95])
        epsilon = 0.1
        
        # Act: The perturbation (0.08) is valid, but adding it to the original
        # results in 1.03, which is outside the valid image range.
        x_perturbed = x_original + 0.08  # This is 1.03
        x_clipped = clip(x_perturbed, x_original, epsilon)
        
        # Assert: The result should be clamped to the maximum image value, 1.0.
        expected_result = torch.tensor([1.0])
        torch.testing.assert_close(x_clipped, expected_result)

    # Test 3: Verify the final value clipping (for values < 0)
    def test_clips_values_below_zero(self):
        """
        Tests that a final value less than 0.0 is correctly clipped to 0.0.
        """
        clip = _test_config['student_function']
        # Arrange: original is 0.05, epsilon is 0.1.
        # The perturbation is -0.08, which is within the epsilon limit.
        x_original = torch.tensor([0.05])
        epsilon = 0.1
        
        # Act: The perturbation (-0.08) is valid, but adding it to the original
        # results in -0.03, which is outside the valid image range.
        x_perturbed = x_original - 0.08 # This is -0.03
        x_clipped = clip(x_perturbed, x_original, epsilon)
        
        # Assert: The result should be clamped to the minimum image value, 0.0.
        expected_result = torch.tensor([0.0])
        torch.testing.assert_close(x_clipped, expected_result)

    # Test 4: Verify that no clipping occurs when it's not needed
    def test_no_clipping_needed(self):
        """
        Tests that if the input is within both the epsilon-ball and the [0, 1] range,
        it remains unchanged.
        """
        clip = _test_config['student_function']
        # Arrange
        x_original = torch.tensor([0.5])
        epsilon = 0.2 # Epsilon-ball is [0.3, 0.7]
        
        # Act: The perturbed value is 0.6, which is within all boundaries.
        x_perturbed = torch.tensor([0.6])
        x_clipped = clip(x_perturbed, x_original, epsilon)

        # Assert: The output should be identical to the input perturbation.
        torch.testing.assert_close(x_clipped, x_perturbed)


class TestTask4a:
    
    # Test 1: Validate the fundamental properties of the output
    def test_output_properties(self):
        """
        Tests that the output is a valid image tensor: correct shape, type, and value range.
        """
        # Setup
        model = _test_config['model']
        student_IGSM = _test_config['student_function']
        loss_fn = torch.nn.CrossEntropyLoss()
        epsilon = 8/255
        alpha = 1/100
        num_iters = 8
        img_path = resources.files("xlab.data").joinpath("cat.jpg")
        y_target = torch.tensor([6]) 
        adv_tensor = student_IGSM(model, loss_fn, img_path, y_target, epsilon, alpha, num_iters)
        
        # Assertions
        assert isinstance(adv_tensor, torch.Tensor), "Output should be a torch.Tensor"
        assert adv_tensor.shape == (1, 3, 32, 32), "Output shape is incorrect"
        assert adv_tensor.min() >= 0.0 and adv_tensor.max() <= 1.0, "Output values must be clamped between 0 and 1"

    # Test 2: Verify that a non-zero epsilon actually changes the image
    def test_perturbation_is_applied(self):
        """
        Tests that for a non-zero epsilon, the output image is different from the input.
        """
        # Setup
        model = _test_config['model']
        student_IGSM = _test_config['student_function']
        loss_fn = torch.nn.CrossEntropyLoss()
        epsilon = 8/255
        alpha = 1/100
        num_iters = 8
        img_path = resources.files("xlab.data").joinpath("cat.jpg")
        y_target = torch.tensor([6]) 
        adv_tensor = student_IGSM(model, loss_fn, img_path, y_target, epsilon, alpha, num_iters)

        original_img = utils.process_image(img_path)
        
        # Assertions
        # The original and adversarial images should not be identical
        assert not torch.equal(original_img, adv_tensor), "Image was not perturbed"

    # Test 3: Verify the logic of the epsilon parameter
    def test_epsilon_zero_produces_no_change(self):
        """
        Tests the edge case where epsilon is 0. The output should be identical to the original image.
        """
        # Setup
        model = _test_config['model']
        student_IGSM = _test_config['student_function']
        loss_fn = torch.nn.CrossEntropyLoss()
        epsilon = 0
        alpha = 1/100
        num_iters = 6
        img_path = resources.files("xlab.data").joinpath("cat.jpg")
        y_target = torch.tensor([6]) 
        adv_tensor = student_IGSM(model, loss_fn, img_path, y_target, epsilon, alpha, num_iters)
        
        original_img = utils.process_image(img_path)
        
        # Assertions
        # With epsilon=0, the perturbation term is zero, so the images should be identical.
        assert torch.equal(original_img, adv_tensor), "Image should not change when epsilon is 0"


    # Test 2: Verify that output is adversarial
    def test_output_adversarial(self):
        """
        Tests that for a non-zero epsilon, the prediction is different from the input.
        """
        # Setup
        model = _test_config['model']
        student_IGSM = _test_config['student_function']
        loss_fn = torch.nn.CrossEntropyLoss()
        epsilon = 8/255
        alpha = 1/100
        num_iters = 8
        img_path = resources.files("xlab.data").joinpath("cat.jpg")
        y_target = torch.tensor([6]) 
        adv_tensor = student_IGSM(model, loss_fn, img_path, y_target, epsilon, alpha, num_iters)
        pred = xlab.utils.prediction(model, adv_tensor)[0]
        
        # Assertions
        # The original and adversarial images should not be identical
        assert not torch.equal(y_target, adv_tensor), "Output should be adversarial"



#Tests for Task 5

class TestTask5:
    
    # Test 1: Validate the fundamental properties of the output
    def test_output_properties(self):
        """
        Tests that the output is a valid image tensor: correct shape, type, and value range.
        """
        # Setup
        model = _test_config['model']
        student_PGD = _test_config['student_function']
        loss_fn = torch.nn.CrossEntropyLoss()
        epsilon = 8/255
        alpha = 1/100
        num_iters = 8
        img_path = resources.files("xlab.data").joinpath("cat.jpg")
        y_target = torch.tensor([6]) 
        adv_tensor = student_PGD(model, loss_fn, img_path, y_target, epsilon, alpha, num_iters)
        
        # Assertions
        assert isinstance(adv_tensor, torch.Tensor), "Output should be a torch.Tensor"
        assert adv_tensor.shape == (1, 3, 32, 32), "Output shape is incorrect"
        assert adv_tensor.min() >= 0.0 and adv_tensor.max() <= 1.0, "Output values must be clamped between 0 and 1"

    # Test 2: Verify that a non-zero epsilon actually changes the image
    def test_perturbation_is_applied(self):
        """
        Tests that for a non-zero epsilon, the output image is different from the input.
        """
        # Setup
        model = _test_config['model']
        student_PGD = _test_config['student_function']
        loss_fn = torch.nn.CrossEntropyLoss()
        epsilon = 8/255
        alpha = 1/100
        num_iters = 8
        img_path = resources.files("xlab.data").joinpath("cat.jpg")
        y_target = torch.tensor([6]) 
        adv_tensor = student_PGD(model, loss_fn, img_path, y_target, epsilon, alpha, num_iters)

        original_img = utils.process_image(img_path)
        
        # Assertions
        # The original and adversarial images should not be identical
        assert not torch.equal(original_img, adv_tensor), "Image was not perturbed"

    # Test 3: Verify the logic of the epsilon parameter
    def test_epsilon_zero_produces_change(self):
        """
        Tests the edge case where epsilon is 0. The output should no be identical to the original image.
        """
        # Setup
        model = _test_config['model']
        student_PGD = _test_config['student_function']
        loss_fn = torch.nn.CrossEntropyLoss()
        epsilon = 0.1
        alpha = 1/100
        num_iters = 6
        img_path = resources.files("xlab.data").joinpath("cat.jpg")
        y_target = torch.tensor([6]) 
        adv_tensor = student_PGD(model, loss_fn, img_path, y_target, epsilon, alpha, num_iters)
        
        original_img = utils.process_image(img_path)
        
        # Assertions
        # With epsilon=0, the perturbation term is zero, so the images should be identical.
        assert not torch.equal(original_img, adv_tensor), "Image should not change when epsilon is 0"


    # Test 2: Verify that output is adversarial
    def test_output_adversarial(self):
        """
        Tests that for a non-zero epsilon, the prediction is different from the input.
        """
        # Setup
        model = _test_config['model']
        student_PGD = _test_config['student_function']
        loss_fn = torch.nn.CrossEntropyLoss()
        epsilon = 8/255
        alpha = 1/100
        num_iters = 8
        img_path = resources.files("xlab.data").joinpath("cat.jpg")
        y_target = torch.tensor([6]) 
        adv_tensor = student_PGD(model, loss_fn, img_path, y_target, epsilon, alpha, num_iters)
        pred = xlab.utils.prediction(model, adv_tensor)[0]
        
        # Assertions
        # The original and adversarial images should not be identical
        assert not torch.equal(y_target, adv_tensor), "Output should be adversarial"

class TestTask6():
    def test_l1_norm(self):
        """
        Tests the L1 norm (p=1) with a simple vector.
        """
        # Arrange
        distance = _test_config['student_function']
        x1 = torch.tensor([1., 5., 2.])
        x2 = torch.tensor([2., 3., 4.])
        p = 1
        
        # Manually calculate expected result:
        expected_distance = 5.0

        # Act
        actual_distance = distance(x1, x2, p)
        
        # Assert
        assert np.round(expected_distance, decimals = 4) == np.round(actual_distance.item(), decimals = 4)

    def test_l2_norm(self):
        """
        Tests the L2 norm (p=2) with a classic 3-4-5 triangle.
        """
        # Arrange
        distance = _test_config['student_function']
        x1 = torch.tensor([1., 2.])
        x2 = torch.tensor([4., 6.])
        p = 2

        # Manually calculate expected result:
        # sqrt( (1-4)^2 + (2-6)^2 ) = sqrt( (-3)^2 + (-4)^2 ) = sqrt(9 + 16) = sqrt(25) = 5.0
        expected_distance = 5.0
        
        # Act
        actual_distance = distance(x1, x2, p)
        
        # Assert
        assert np.round(expected_distance, decimals = 4) == np.round(actual_distance.item(), decimals = 4)

        
        
    def test_multi_dimensional_tensor_summation(self):
        """
        Tests that the function correctly sums over all elements in a multi-dimensional tensor.
        """
        # Arrange
        distance = _test_config['student_function']
        x1 = torch.tensor([[1., 2.], [3., 4.]])
        x2 = torch.tensor([[2., 2.], [3., 5.]])
        p = 1

        # Manually calculate expected result:
        # abs(1-2) + abs(2-2) + abs(3-3) + abs(4-5) = 1 + 0 + 0 + 1 = 2.0
        expected_distance = 2.0
        
        # Act
        actual_distance = distance(x1, x2, p)

        # Assert
        assert np.round(expected_distance, decimals = 4) == np.round(actual_distance.item(), decimals = 4)



    
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
# def task1(model):
#     """
#     Run Task 1 tests using pytest.
    
#     Args:
#         model: The model to test with.
    
#     Returns:
#         dict: A summary dictionary with test results.
#     """
#     # Configure global test parameters
#     _test_config['model'] = model
    
#     # Run pytest tests
#     result = _run_pytest_with_capture(TestTask1)
#     _print_test_summary(result, "Task 1")
    
#     return result


def task2(student_function):
    """
    Run Task 2 tests using pytest.
    
    Args:
        student_function (function): The student's function to test.
    
    Returns:
        dict: A summary dictionary with test results.
    """
    # Configure global test parameters
    _test_config['student_function'] = student_function
    
    # Run pytest tests
    result = _run_pytest_with_capture(TestTask2)
    _print_test_summary(result, "Task 2")
    
    return result


def task3(model, student_function):
    """
    Run Task 3 tests using pytest.
    
    Args:
        student_function (function): The student's function to test.
        model: The model to test with.
    
    Returns:
        dict: A summary dictionary with test results.
    """
    # Configure global test parameters
    _test_config['model'] = model
    _test_config['student_function'] = student_function
    
    # Run pytest tests
    result = _run_pytest_with_capture(TestTask3)
    _print_test_summary(result, "Task 3")
    
    return result


def task4(student_function):
    """
    Run Task 4 tests using pytest.
    
    Args:
        student_function (function): The student's function to test.
    
    Returns:
        dict: A summary dictionary with test results.
    """
    # Configure global test parameters
    _test_config['student_function'] = student_function
    
    # Run pytest tests
    result = _run_pytest_with_capture(TestTask4)
    _print_test_summary(result, "Task 4")
    
    return result



def task4a(model, student_function):
    """
    Run Task 4a tests using pytest.
    
    Args:
        student_function (function): The student's function to test.
        model: The model to test with.
    
    Returns:
        dict: A summary dictionary with test results.
    """
    # Configure global test parameters
    _test_config['model'] = model
    _test_config['student_function'] = student_function
    
    # Run pytest tests
    result = _run_pytest_with_capture(TestTask4a)
    _print_test_summary(result, "Task 4a")
    
    return result


def task5(model, student_function):
    """
    Run Task 5 tests using pytest.
    
    Args:
        student_function (function): The student's function to test.
        model: The model to test with.
    
    Returns:
        dict: A summary dictionary with test results.
    """
    # Configure global test parameters
    _test_config['model'] = model
    _test_config['student_function'] = student_function
    
    # Run pytest tests
    result = _run_pytest_with_capture(TestTask5)
    _print_test_summary(result, "Task 5")
    
    return result


def task6(student_function):
    """
    Run Task 6 tests using pytest.
    
    Args:
        student_function (function): The student's function to test.
    
    Returns:
        dict: A summary dictionary with test results.
    """
    # Configure global test parameters
    _test_config['student_function'] = student_function
    
    # Run pytest tests
    result = _run_pytest_with_capture(TestTask6)
    _print_test_summary(result, "Task 6")
    
    return result

