import pytest
import torch
import torch.nn.functional as F
from torch import Tensor

class TestTarget:
    func = None

target = TestTarget()

@pytest.fixture
def student_function():
    """A pytest fixture that provides the student's function to any test."""
    if target.func is None:
        pytest.skip("Student function not provided.")
    return target.func


@pytest.mark.task1
def test_f2_output_shape_and_type(student_function):
    """Tests that f2 returns a scalar tensor."""
    logits = torch.randn(10)
    target_class = 3
    
    result = student_function(logits, target_class)
    
    assert isinstance(result, Tensor), f"Expected a Tensor, but got {type(result)}"
    assert result.shape == torch.Size([]), f"Expected a scalar tensor, but got shape {result.shape}"
    assert result.item() >= 0, "Output should be non-negative due to ReLU"


@pytest.mark.task1 
def test_f2_target_not_max(student_function):
    """Tests f2 when target is not the maximum class."""
    # Create logits where class 0 has highest prob, target is class 2
    logits = torch.tensor([2.0, 1.0, 0.5, 1.5])  
    target_class = 2
    
    result = student_function(logits, target_class)
    
    # Manual calculation
    softmax_probs = F.softmax(logits, dim=0)
    max_non_target = softmax_probs[0]  # class 0 is max
    target_prob = softmax_probs[target_class]
    expected = F.relu(max_non_target - target_prob)
    
    assert torch.allclose(result, expected, atol=1e-6), f"Expected {expected}, but got {result}"


@pytest.mark.task1
def test_f2_target_is_max(student_function):
    """Tests f2 when target is the maximum class (needs masking)."""
    # Create logits where target class has highest prob
    logits = torch.tensor([1.0, 0.5, 3.0, 1.5])  # class 2 is max
    target_class = 2
    
    result = student_function(logits, target_class)
    
    # Manual calculation
    softmax_probs = F.softmax(logits, dim=0)
    masked_probs = softmax_probs.clone()
    masked_probs[target_class] = float('-inf')
    second_max_idx = torch.argmax(masked_probs)
    expected = F.relu(softmax_probs[second_max_idx] - softmax_probs[target_class])
    
    assert torch.allclose(result, expected, atol=1e-6), f"Expected {expected}, but got {result}"


@pytest.mark.task1
def test_f2_large_confidence_gap(student_function):
    """Tests f2 with a large confidence gap."""
    # Target has very low probability compared to max
    logits = torch.tensor([5.0, 2.0, -2.0, 1.0])  # class 0 >> class 2
    target_class = 2
    
    result = student_function(logits, target_class)
    
    # Should return a positive value since max_prob >> target_prob
    assert result.item() > 0, "Should return positive value when max probability >> target probability"


def task1(student_func):
    """Runs all 'task1' tests against the provided student function."""
    target.func = student_func
    result_code = pytest.main([__file__, "-v", "--no-header", "-m", "task1"])
    if result_code == pytest.ExitCode.OK:
        print("✅ All checks passed!")


@pytest.mark.task2
def test_f4_output_shape_and_type(student_function):
    """Tests that f4 returns a scalar tensor."""
    logits = torch.randn(10)
    target_class = 3
    
    result = student_function(logits, target_class)
    
    assert isinstance(result, Tensor), f"Expected a Tensor, but got {type(result)}"
    assert result.shape == torch.Size([]), f"Expected a scalar tensor, but got shape {result.shape}"
    assert result.item() >= 0, "Output should be non-negative due to ReLU"


@pytest.mark.task2
def test_f4_target_prob_above_threshold(student_function):
    """Tests f4 when target probability is above 0.5 (should return 0)."""
    # Create logits where target class has high probability
    logits = torch.tensor([0.0, 0.0, 3.0, 0.0])  # class 2 will have prob > 0.5
    target_class = 2
    
    result = student_function(logits, target_class)
    
    # Manual calculation
    softmax_probs = F.softmax(logits, dim=0)
    expected = F.relu(0.5 - softmax_probs[target_class])
    
    assert torch.allclose(result, expected, atol=1e-6), f"Expected {expected}, but got {result}"
    assert result.item() == 0.0, "Should return 0 when target probability > 0.5"


@pytest.mark.task2
def test_f4_target_prob_below_threshold(student_function):
    """Tests f4 when target probability is below 0.5 (should return positive value)."""
    # Create uniform logits where each class has prob = 0.25 < 0.5
    logits = torch.tensor([1.0, 1.0, 1.0, 1.0])  # uniform distribution
    target_class = 2
    
    result = student_function(logits, target_class)
    
    # Manual calculation
    softmax_probs = F.softmax(logits, dim=0)
    expected = F.relu(0.5 - softmax_probs[target_class])
    
    assert torch.allclose(result, expected, atol=1e-6), f"Expected {expected}, but got {result}"
    assert result.item() > 0, "Should return positive value when target probability < 0.5"


@pytest.mark.task2
def test_f4_target_prob_near_threshold(student_function):
    """Tests f4 when target probability is very close to 0.5."""
    # Create logits that result in target prob ≈ 0.5
    logits = torch.tensor([0.0, 0.0, 0.0, 0.0])  # exactly 0.25 each
    target_class = 1
    
    result = student_function(logits, target_class)
    
    # Manual calculation  
    softmax_probs = F.softmax(logits, dim=0)
    target_prob = softmax_probs[target_class]
    expected = F.relu(0.5 - target_prob)
    
    assert torch.allclose(result, expected, atol=1e-6), f"Expected {expected}, but got {result}"
    # With uniform distribution, prob = 0.25, so 0.5 - 0.25 = 0.25
    assert abs(result.item() - 0.25) < 1e-6, f"Expected approximately 0.25, but got {result.item()}"


def task2(student_func):
    """Runs all 'task2' tests against the provided student function."""
    target.func = student_func
    result_code = pytest.main([__file__, "-v", "--no-header", "-m", "task2"])
    if result_code == pytest.ExitCode.OK:
        print("✅ All checks passed!")
