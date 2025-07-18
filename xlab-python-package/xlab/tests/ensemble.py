import pytest
import torch
from torch import Tensor
from unittest.mock import patch

class TestTarget:
    func = None

target = TestTarget()

@pytest.fixture
def student_function():
    """A pytest fixture that provides the student's function to any test."""
    if target.func is None:
        pytest.skip("Student function not provided.")
    return target.func


# Expected loss values for each of the 10 MNIST classes
EXPECTED_LOSSES = [
    13.77052116394043, 15.317304611206055, 8.956832885742188, 7.667974948883057, 
    18.443864822387695, 14.074539184570312, 27.119890213012695, 0.0033627948723733425, 
    12.95663833618164, 8.222661018371582
]

# Global variable to store the actual losses for testing
_actual_losses = None

def set_losses_for_testing(losses):
    """Set the losses to be tested. Call this before running the tests."""
    global _actual_losses
    _actual_losses = losses
    if isinstance(_actual_losses, Tensor):
        _actual_losses = _actual_losses.tolist()

# Global variables to store test data for task 2
_test_images = None
_test_model = None

def set_images_and_model_for_testing(images, model):
    """Set the images and model to be tested. Call this before running task 2 tests."""
    global _test_images, _test_model
    _test_images = images
    _test_model = model

@pytest.mark.task1
def test_mnist_class_0_loss():
    """Test loss for MNIST class 0."""
    assert _actual_losses is not None, "No losses set for testing. Call set_losses_for_testing() first."
    assert len(_actual_losses) >= 1, "Not enough loss values provided."
    epsilon = 1e-4
    actual = _actual_losses[0]
    expected = EXPECTED_LOSSES[0]
    assert abs(actual - expected) < epsilon, \
        f"Loss for class 0 is {actual}, expected {expected} (difference: {abs(actual - expected)})"

@pytest.mark.task1
def test_mnist_class_1_loss():
    """Test loss for MNIST class 1."""
    assert _actual_losses is not None, "No losses set for testing. Call set_losses_for_testing() first."
    assert len(_actual_losses) >= 2, "Not enough loss values provided."
    epsilon = 1e-4
    actual = _actual_losses[1]
    expected = EXPECTED_LOSSES[1]
    assert abs(actual - expected) < epsilon, \
        f"Loss for class 1 is {actual}, expected {expected} (difference: {abs(actual - expected)})"

@pytest.mark.task1
def test_mnist_class_2_loss():
    """Test loss for MNIST class 2."""
    assert _actual_losses is not None, "No losses set for testing. Call set_losses_for_testing() first."
    assert len(_actual_losses) >= 3, "Not enough loss values provided."
    epsilon = 1e-4
    actual = _actual_losses[2]
    expected = EXPECTED_LOSSES[2]
    assert abs(actual - expected) < epsilon, \
        f"Loss for class 2 is {actual}, expected {expected} (difference: {abs(actual - expected)})"

@pytest.mark.task1
def test_mnist_class_3_loss():
    """Test loss for MNIST class 3."""
    assert _actual_losses is not None, "No losses set for testing. Call set_losses_for_testing() first."
    assert len(_actual_losses) >= 4, "Not enough loss values provided."
    epsilon = 1e-4
    actual = _actual_losses[3]
    expected = EXPECTED_LOSSES[3]
    assert abs(actual - expected) < epsilon, \
        f"Loss for class 3 is {actual}, expected {expected} (difference: {abs(actual - expected)})"

@pytest.mark.task1
def test_mnist_class_4_loss():
    """Test loss for MNIST class 4."""
    assert _actual_losses is not None, "No losses set for testing. Call set_losses_for_testing() first."
    assert len(_actual_losses) >= 5, "Not enough loss values provided."
    epsilon = 1e-4
    actual = _actual_losses[4]
    expected = EXPECTED_LOSSES[4]
    assert abs(actual - expected) < epsilon, \
        f"Loss for class 4 is {actual}, expected {expected} (difference: {abs(actual - expected)})"

@pytest.mark.task1
def test_mnist_class_5_loss():
    """Test loss for MNIST class 5."""
    assert _actual_losses is not None, "No losses set for testing. Call set_losses_for_testing() first."
    assert len(_actual_losses) >= 6, "Not enough loss values provided."
    epsilon = 1e-4
    actual = _actual_losses[5]
    expected = EXPECTED_LOSSES[5]
    assert abs(actual - expected) < epsilon, \
        f"Loss for class 5 is {actual}, expected {expected} (difference: {abs(actual - expected)})"

@pytest.mark.task1
def test_mnist_class_6_loss():
    """Test loss for MNIST class 6."""
    assert _actual_losses is not None, "No losses set for testing. Call set_losses_for_testing() first."
    assert len(_actual_losses) >= 7, "Not enough loss values provided."
    epsilon = 1e-4
    actual = _actual_losses[6]
    expected = EXPECTED_LOSSES[6]
    assert abs(actual - expected) < epsilon, \
        f"Loss for class 6 is {actual}, expected {expected} (difference: {abs(actual - expected)})"

@pytest.mark.task1
def test_mnist_class_7_loss():
    """Test loss for MNIST class 7."""
    assert _actual_losses is not None, "No losses set for testing. Call set_losses_for_testing() first."
    assert len(_actual_losses) >= 8, "Not enough loss values provided."
    epsilon = 1e-4
    actual = _actual_losses[7]
    expected = EXPECTED_LOSSES[7]
    assert abs(actual - expected) < epsilon, \
        f"Loss for class 7 is {actual}, expected {expected} (difference: {abs(actual - expected)})"

@pytest.mark.task1
def test_mnist_class_8_loss():
    """Test loss for MNIST class 8."""
    assert _actual_losses is not None, "No losses set for testing. Call set_losses_for_testing() first."
    assert len(_actual_losses) >= 9, "Not enough loss values provided."
    epsilon = 1e-4
    actual = _actual_losses[8]
    expected = EXPECTED_LOSSES[8]
    assert abs(actual - expected) < epsilon, \
        f"Loss for class 8 is {actual}, expected {expected} (difference: {abs(actual - expected)})"

@pytest.mark.task1
def test_mnist_class_9_loss():
    """Test loss for MNIST class 9."""
    assert _actual_losses is not None, "No losses set for testing. Call set_losses_for_testing() first."
    assert len(_actual_losses) >= 10, "Not enough loss values provided."
    epsilon = 1e-4
    actual = _actual_losses[9]
    expected = EXPECTED_LOSSES[9]
    assert abs(actual - expected) < epsilon, \
        f"Loss for class 9 is {actual}, expected {expected} (difference: {abs(actual - expected)})"

@pytest.mark.task1
def test_mnist_losses_count():
    """Test that exactly 10 loss values are provided."""
    assert _actual_losses is not None, "No losses set for testing. Call set_losses_for_testing() first."
    assert len(_actual_losses) == 10, f"Expected exactly 10 loss values (one per MNIST class), but got {len(_actual_losses)}"

@pytest.mark.task2
def test_mnist_images_count():
    """Test that exactly 5 MNIST images are provided."""
    assert _test_images is not None, "No images set for testing. Call set_images_and_model_for_testing() first."
    assert len(_test_images) == 5, f"Expected exactly 5 MNIST images, but got {len(_test_images)}"

@pytest.mark.task2
def test_mnist_images_shape():
    """Test that all images have the correct shape [1, 1, 28, 28]."""
    assert _test_images is not None, "No images set for testing. Call set_images_and_model_for_testing() first."
    expected_shape = torch.Size([1, 1, 28, 28])
    
    for i, img in enumerate(_test_images):
        assert isinstance(img, Tensor), f"Image {i} should be a torch.Tensor, but got {type(img)}"
        assert img.shape == expected_shape, f"Image {i} has shape {img.shape}, expected {expected_shape}"

@pytest.mark.task2
def test_model_provided():
    """Test that a model is provided."""
    assert _test_model is not None, "No model set for testing. Call set_images_and_model_for_testing() first."

@pytest.mark.task2
def test_mnist_classification_to_3():
    """Test that at least 4 out of 5 images classify as digit '3'."""
    assert _test_images is not None, "No images set for testing. Call set_images_and_model_for_testing() first."
    assert _test_model is not None, "No model set for testing. Call set_images_and_model_for_testing() first."
    
    predictions = []
    
    # Get predictions for each image
    with torch.no_grad():
        for i, img in enumerate(_test_images):
            try:
                output = _test_model.predict(img)
                predictions.append(output.item())
            except Exception as e:
                pytest.fail(f"Model failed to classify image {i}: {e}")
    
    # Count how many predictions are "3"
    count_of_3s = sum(1 for pred in predictions if pred == 3)
    
    assert count_of_3s >= 4, \
        f"Expected at least 4 out of 5 images to classify as digit '3', but only {count_of_3s} did. " \
        f"Predictions: {predictions}"

def task1(losses):
    """Runs all 'task1' tests against the provided losses array."""
    # Set the losses for the global tests
    set_losses_for_testing(losses)
    
    # Run pytest with the task1 marker
    result_code = pytest.main([__file__, "-v", "--no-header", "-m", "task1"])
    if result_code == pytest.ExitCode.OK:
        print("✅ All checks passed!")

def task2(images, model):
    """Runs all 'task2' tests against the provided images and model."""
    # Set the images and model for the global tests
    set_images_and_model_for_testing(images, model)
    
    # Run pytest with the task2 marker
    result_code = pytest.main([__file__, "-v", "--no-header", "-m", "task2"])
    if result_code == pytest.ExitCode.OK:
        print("✅ All checks passed!")
