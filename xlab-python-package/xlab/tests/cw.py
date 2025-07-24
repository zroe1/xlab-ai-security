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


@pytest.mark.task3
def test_f6_output_shape_and_type(student_function):
    """Tests that f6 returns a scalar tensor."""
    logits = torch.randn(10)
    target_class = 3
    
    result = student_function(logits, target_class)
    
    assert isinstance(result, Tensor), f"Expected a Tensor, but got {type(result)}"
    assert result.shape == torch.Size([]), f"Expected a scalar tensor, but got shape {result.shape}"
    assert result.item() >= 0, "Output should be non-negative due to ReLU"


@pytest.mark.task3
def test_f6_target_not_max(student_function):
    """Tests f6 when target is not the maximum logit."""
    # Create logits where class 0 has highest logit, target is class 2
    logits = torch.tensor([3.0, 1.0, 0.5, 2.0])  
    target_class = 2
    
    result = student_function(logits, target_class)
    
    # Manual calculation
    max_logit_idx = torch.argmax(logits)  # should be class 0
    max_logit = logits[max_logit_idx]
    target_logit = logits[target_class]
    expected = F.relu(max_logit - target_logit)
    
    assert torch.allclose(result, expected, atol=1e-6), f"Expected {expected}, but got {result}"


@pytest.mark.task3
def test_f6_target_is_max(student_function):
    """Tests f6 when target is the maximum logit (needs masking)."""
    # Create logits where target class has highest logit
    logits = torch.tensor([1.0, 0.5, 4.0, 2.0])  # class 2 is max
    target_class = 2
    
    result = student_function(logits, target_class)
    
    # Manual calculation
    masked_logits = logits.clone()
    masked_logits[target_class] = float('-inf')
    second_max_idx = torch.argmax(masked_logits)  # should be class 3
    expected = F.relu(logits[second_max_idx] - logits[target_class])
    
    assert torch.allclose(result, expected, atol=1e-6), f"Expected {expected}, but got {result}"


@pytest.mark.task3
def test_f6_large_logit_difference(student_function):
    """Tests f6 with large logit differences."""
    # Target has much lower logit than max
    logits = torch.tensor([10.0, 2.0, -5.0, 1.0])  # class 0 >> class 2
    target_class = 2
    
    result = student_function(logits, target_class)
    
    # Should return a large positive value since max_logit >> target_logit
    expected_diff = 10.0 - (-5.0)  # 15.0
    assert abs(result.item() - expected_diff) < 1e-6, f"Expected {expected_diff}, but got {result.item()}"
    assert result.item() > 0, "Should return positive value when max logit >> target logit"


@pytest.mark.task3
def test_f6_target_higher_than_max(student_function):
    """Tests f6 when target logit is higher than max non-target (should return 0)."""
    # Target has highest logit by a good margin
    logits = torch.tensor([1.0, 2.0, 8.0, 3.0])  # class 2 is clearly highest
    target_class = 2
    
    result = student_function(logits, target_class)
    
    # Manual calculation: max non-target is class 3 with logit 3.0
    # target logit is 8.0, so 3.0 - 8.0 = -5.0, ReLU(-5.0) = 0
    assert result.item() == 0.0, f"Should return 0 when target logit > max non-target logit, but got {result.item()}"


def task3(student_func):
    """Runs all 'task3' tests against the provided student function."""
    target.func = student_func
    result_code = pytest.main([__file__, "-v", "--no-header", "-m", "task3"])
    if result_code == pytest.ExitCode.OK:
        print("✅ All checks passed!")


# Global variables to store test parameters for task4
_task4_config = {
    'array1': None,
    'array2': None,
    'array3': None,
    'target': None
}


@pytest.mark.task4
def test_arrays_have_equal_length(student_function):
    """Tests that all arrays have equal length."""
    array1 = _task4_config['array1']
    array2 = _task4_config['array2']
    array3 = _task4_config['array3']
    
    assert array1 is not None, "Array1 not configured for testing"
    assert array2 is not None, "Array2 not configured for testing"
    assert array3 is not None, "Array3 not configured for testing"
    
    len1, len2, len3 = len(array1), len(array2), len(array3)
    assert len1 == len2 == len3, f"Arrays must have equal length for valid comparison. " \
                                f"Got lengths: {len1}, {len2}, {len3}"


@pytest.mark.task4
def test_third_array_has_most_occurrences(student_function):
    """Tests that the third array has the most occurrences of the target value."""
    array1 = _task4_config['array1']
    array2 = _task4_config['array2']
    array3 = _task4_config['array3']
    target = _task4_config['target']
    
    assert array1 is not None, "Array1 not configured for testing"
    assert array2 is not None, "Array2 not configured for testing"
    assert array3 is not None, "Array3 not configured for testing"
    assert target is not None, "Target value not configured for testing"
    
    # Count occurrences in each array
    count1 = array1.count(target) if hasattr(array1, 'count') else sum(1 for x in array1 if x == target)
    count2 = array2.count(target) if hasattr(array2, 'count') else sum(1 for x in array2 if x == target)
    count3 = array3.count(target) if hasattr(array3, 'count') else sum(1 for x in array3 if x == target)
    
    assert count3 > count1 and count3 > count2, \
        f"Third array does not have the most occurrences of target {target}. " \
        f"Counts: Array1={count1}, Array2={count2}, Array3={count3}"


def task4(array1, array2, array3, target):
    """
    Run Task 4 tests using pytest.
    
    Tests that the third array has the most occurrences of the target value
    compared to the first two arrays.

    Args:
        array1 (list): First array to compare.
        array2 (list): Second array to compare.
        array3 (list): Third array to compare.
        target (int): The target value to count in each array.

    Returns:
        dict: A summary dictionary with the total number of tests, the
              number passed/failed, and a final score.
    """
    # Configure global test parameters
    _task4_config['array1'] = array1
    _task4_config['array2'] = array2
    _task4_config['array3'] = array3
    _task4_config['target'] = target
    
    # Run pytest tests
    result_code = pytest.main([__file__, "-v", "--no-header", "-m", "task4"])
    if result_code == pytest.ExitCode.OK:
        print("✅ All checks passed!")
    
    return {"exit_code": result_code}


@pytest.mark.task6
def test_confident_f6_output_shape_and_type(student_function):
    """Tests that confident_f6 returns a scalar tensor."""
    logits = torch.randn(10)
    target_class = 3
    tau = 2.0
    
    result = student_function(logits, target_class, tau)
    
    assert isinstance(result, Tensor), f"Expected a Tensor, but got {type(result)}"
    assert result.shape == torch.Size([]), f"Expected a scalar tensor, but got shape {result.shape}"


@pytest.mark.task6
def test_confident_f6_target_not_max(student_function):
    """Tests confident_f6 when target is not the maximum logit."""
    # Create logits where class 0 has highest logit, target is class 2
    logits = torch.tensor([3.0, 1.0, 0.5, 2.0])  
    target_class = 2
    tau = 1.0
    
    result = student_function(logits, target_class, tau)
    
    # Manual calculation
    max_logit_idx = torch.argmax(logits)  # should be class 0
    max_logit = logits[max_logit_idx]
    target_logit = logits[target_class]
    diff = max_logit - target_logit  # 3.0 - 0.5 = 2.5
    expected = torch.max(diff, -torch.tensor(tau))
    
    assert torch.allclose(result, expected, atol=1e-6), f"Expected {expected}, but got {result}"


@pytest.mark.task6
def test_confident_f6_target_is_max(student_function):
    """Tests confident_f6 when target is the maximum logit (needs masking)."""
    # Create logits where target class has highest logit
    logits = torch.tensor([1.0, 0.5, 4.0, 2.0])  # class 2 is max
    target_class = 2
    tau = 0.5
    
    result = student_function(logits, target_class, tau)
    
    # Manual calculation
    masked_logits = logits.clone()
    masked_logits[target_class] = float('-inf')
    second_max_idx = torch.argmax(masked_logits)  # should be class 3
    diff = logits[second_max_idx] - logits[target_class]  # 2.0 - 4.0 = -2.0
    expected = torch.max(diff, -torch.tensor(tau))  # max(-2.0, -0.5) = -0.5
    
    assert torch.allclose(result, expected, atol=1e-6), f"Expected {expected}, but got {result}"


@pytest.mark.task6
def test_confident_f6_tau_clamping(student_function):
    """Tests tau clamping when difference would be very negative."""
    # Target has much higher logit than others
    logits = torch.tensor([1.0, 2.0, 8.0, 3.0])  # class 2 dominates
    target_class = 2
    tau = 1.0
    
    result = student_function(logits, target_class, tau)
    
    # Difference would be 3.0 - 8.0 = -5.0, but clamped to -tau = -1.0
    expected = torch.tensor(-tau)
    assert torch.allclose(result, expected, atol=1e-6), f"Expected {expected}, but got {result}"


@pytest.mark.task6
def test_confident_f6_no_clamping_needed(student_function):
    """Tests when no tau clamping is needed."""
    # Create case where difference > -tau
    logits = torch.tensor([2.0, 1.0, 1.5, 0.5])  # class 0 is max
    target_class = 2
    tau = 2.0
    
    result = student_function(logits, target_class, tau)
    
    # Difference is 2.0 - 1.5 = 0.5, which is > -tau = -2.0
    # So result should be 0.5, not clamped
    expected_diff = 2.0 - 1.5
    assert abs(result.item() - expected_diff) < 1e-6, f"Expected {expected_diff}, but got {result.item()}"


@pytest.mark.task6  
def test_confident_f6_different_tau_values(student_function):
    """Tests confident_f6 with different tau values."""
    logits = torch.tensor([1.0, 2.0, 5.0, 1.5])  # class 2 is max
    target_class = 2
    
    # With larger tau, should clamp to larger negative value
    tau_large = 3.0
    result_large = student_function(logits, target_class, tau_large)
    
    # With smaller tau, should clamp to smaller negative value  
    tau_small = 0.5
    result_small = student_function(logits, target_class, tau_small)
    
    # Both should be negative and clamped, with large tau giving more negative result
    assert result_large.item() < 0, "Result with large tau should be negative"
    assert result_small.item() < 0, "Result with small tau should be negative"
    assert result_large.item() < result_small.item(), "Larger tau should give more negative result"


def task6(student_func):
    """Runs all 'task6' tests against the provided student function."""
    target.func = student_func
    result_code = pytest.main([__file__, "-v", "--no-header", "-m", "task6"])
    if result_code == pytest.ExitCode.OK:
        print("✅ All checks passed!")


@pytest.mark.task7
def test_get_adv_from_w_output_shape_and_type(student_function):
    """Tests that get_adv_from_w preserves shape and type."""
    w = torch.randn(3, 224, 224)
    
    result = student_function(w)
    
    assert isinstance(result, Tensor), f"Expected a Tensor, but got {type(result)}"
    assert result.shape == w.shape, f"Expected shape {w.shape}, but got {result.shape}"
    assert result.dtype == w.dtype, f"Expected dtype {w.dtype}, but got {result.dtype}"


@pytest.mark.task7
def test_get_adv_from_w_value_range(student_function):
    """Tests that output is in range [0, 1]."""
    # Test with various input ranges
    w_large_pos = torch.tensor([10.0, 5.0, 100.0])
    w_large_neg = torch.tensor([-10.0, -5.0, -100.0])
    w_mixed = torch.tensor([-2.0, 0.0, 2.0])
    
    for w in [w_large_pos, w_large_neg, w_mixed]:
        result = student_function(w)
        assert torch.all(result >= 0), f"All values should be >= 0, but got min {result.min()}"
        assert torch.all(result <= 1), f"All values should be <= 1, but got max {result.max()}"


@pytest.mark.task7
def test_get_adv_from_w_specific_values(student_function):
    """Tests specific input/output mappings."""
    # w = 0 should give 0.5
    w_zero = torch.tensor([0.0])
    result_zero = student_function(w_zero)
    expected_zero = 0.5
    assert abs(result_zero.item() - expected_zero) < 1e-6, f"Expected {expected_zero} for w=0, got {result_zero.item()}"
    
    # Large positive w should approach 1
    w_large_pos = torch.tensor([10.0])
    result_large_pos = student_function(w_large_pos)
    assert result_large_pos.item() > 0.99, f"Large positive w should give value close to 1, got {result_large_pos.item()}"
    
    # Large negative w should approach 0
    w_large_neg = torch.tensor([-10.0])
    result_large_neg = student_function(w_large_neg)
    assert result_large_neg.item() < 0.01, f"Large negative w should give value close to 0, got {result_large_neg.item()}"


@pytest.mark.task7
def test_get_adv_from_w_gradient_preservation(student_function):
    """Tests that gradients are preserved when input requires grad."""
    w = torch.randn(2, 3, requires_grad=True)
    
    result = student_function(w)
    
    assert result.requires_grad, "Output should require gradients when input does"
    
    # Test that gradients can flow through
    loss = result.sum()
    loss.backward()
    assert w.grad is not None, "Gradients should flow back to input"
    assert not torch.all(w.grad == 0), "Gradients should be non-zero"


def task7(student_func):
    """Runs all 'task7' tests against the provided student function."""
    target.func = student_func
    result_code = pytest.main([__file__, "-v", "--no-header", "-m", "task7"])
    if result_code == pytest.ExitCode.OK:
        print("✅ All checks passed!")


@pytest.mark.task8
def test_get_delta_output_shape_and_type(student_function):
    """Tests that get_delta returns correct shape and type."""
    w = torch.randn(3, 32, 32)
    x = torch.randn(3, 32, 32)
    
    result = student_function(w, x)
    
    assert isinstance(result, Tensor), f"Expected a Tensor, but got {type(result)}"
    assert result.shape == w.shape, f"Expected shape {w.shape}, but got {result.shape}"
    assert result.shape == x.shape, f"Expected shape {x.shape}, but got {result.shape}"


@pytest.mark.task8
def test_get_delta_computation(student_function):
    """Tests the mathematical relationship delta = adv - x."""
    w = torch.tensor([0.0, 2.0, -2.0])
    x = torch.tensor([0.3, 0.7, 0.1])
    
    result = student_function(w, x)
    
    # Manual calculation
    adv = 0.5 * (torch.tanh(w) + 1)
    expected_delta = adv - x
    
    assert torch.allclose(result, expected_delta, atol=1e-6), f"Expected {expected_delta}, but got {result}"


@pytest.mark.task8
def test_get_delta_zero_case(student_function):
    """Tests when w corresponds to x (delta should be close to zero)."""
    # Choose x values, then find w such that 0.5 * (tanh(w) + 1) ≈ x
    x = torch.tensor([0.2, 0.5, 0.8])
    # Inverse: w = atanh(2*x - 1)
    w = torch.atanh(2 * x - 1)
    
    result = student_function(w, x)
    
    # Delta should be very close to zero
    assert torch.allclose(result, torch.zeros_like(result), atol=1e-5), \
        f"Delta should be close to zero when w corresponds to x, but got {result}"


@pytest.mark.task8
def test_get_delta_gradient_preservation(student_function):
    """Tests that gradients flow through both w and x."""
    w = torch.randn(2, 3, requires_grad=True)
    x = torch.randn(2, 3, requires_grad=True)
    
    result = student_function(w, x)
    
    assert result.requires_grad, "Output should require gradients when inputs do"
    
    # Test gradient flow
    loss = result.sum()
    loss.backward()
    
    assert w.grad is not None, "Gradients should flow back to w"
    assert x.grad is not None, "Gradients should flow back to x"
    assert not torch.all(w.grad == 0), "w gradients should be non-zero"
    assert torch.allclose(x.grad, -torch.ones_like(x.grad)), "x gradient should be -1"


@pytest.mark.task8
def test_get_delta_different_shapes(student_function):
    """Tests get_delta with different tensor shapes."""
    shapes = [(10,), (3, 5), (2, 3, 4), (1, 28, 28)]
    
    for shape in shapes:
        w = torch.randn(shape)
        x = torch.randn(shape)
        
        result = student_function(w, x)
        assert result.shape == shape, f"Failed for shape {shape}: expected {shape}, got {result.shape}"


def task8(student_func):
    """Runs all 'task8' tests against the provided student function."""
    target.func = student_func
    result_code = pytest.main([__file__, "-v", "--no-header", "-m", "task8"])
    if result_code == pytest.ExitCode.OK:
        print("✅ All checks passed!")
